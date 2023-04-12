import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from timm.utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoProcessor,
                          get_cosine_schedule_with_warmup)

import wandb


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = self.processor(
            images=item["image"],
            text=item["text"],
            padding="max_length",
            return_tensors="pt",
        )

        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding


def evaluate(valid_dataloader, model):
    model.eval()
    data_loader_tqdm = tqdm(valid_dataloader, file=sys.stdout)

    val_meters = {
        "loss": AverageMeter(),
    }

    with torch.no_grad():
        for idx, batch in enumerate(data_loader_tqdm):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
            )
            loss = outputs.loss

            val_meters["loss"].update(loss.item(), n=input_ids.size(0))

            data_loader_tqdm.set_description(f"Epoch {epoch}, loss: {loss.item()}")

    model.train()

    return val_meters["loss"].avg


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1 python run_train.py
    # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/GIT/Fine_tune_GIT_on_an_image_captioning_dataset.ipynb

    wandb.login()

    model_name = "microsoft/git-base"
    epochs = 10
    batch_size = 32
    grad_accum_steps = 16
    valid_batch_size = 32
    learning_rate = 2.5e-5
    valid_steps = 100
    warmup_ratio = 0.05
    use_amp = True
    seed = 42
    memo = f"git-model-{seed}s-{epochs}ep-{model_name}-on-v6"

    wandb.init(
        name=memo,
        project="prompts-to-image",
        config={
            "epochs": epochs,
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "valid_batch_size": valid_batch_size,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
        },
    )

    output_path = f"output_{memo}"
    os.makedirs(output_path, exist_ok=True)

    seed_everything(seed)

    # https://huggingface.co/docs/datasets/image_load#imagefolder
    # https://huggingface.co/docs/datasets/image_dataset
    train_dataset = load_dataset(
        "imagefolder",
        data_dir="./diffusion/image-to-prompt-train-valid-split-v6",
        split="train",
        num_proc=8,
    )
    valid_dataset = load_dataset(
        "imagefolder",
        data_dir="./diffusion/image-to-prompt-train-valid-split-v6",
        split="validation",
        num_proc=8,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    train_dataset = ImageCaptioningDataset(train_dataset, processor)
    valid_dataset = ImageCaptioningDataset(valid_dataset, processor)
    # input_ids [512], attention_mask [512], pixel_values [3, 224, 224]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    total_steps = len(train_dataloader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=use_amp)

    best_score = 9999
    logging_loss = torch.tensor(0.0).cuda()
    step, accumulated_steps = 0, 0
    for epoch in range(epochs):
        model.train()

        data_loader_tqdm = tqdm(train_dataloader, file=sys.stdout)
        for idx, batch in enumerate(data_loader_tqdm):
            accumulated_steps += 1

            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            with autocast(enabled=use_amp, dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
                )

            loss = outputs.loss
            loss /= grad_accum_steps

            logging_loss += loss.detach()

            scaler.scale(loss).backward()

            if accumulated_steps < grad_accum_steps:
                continue

            accumulated_steps = 0
            step += 1

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                mean_loss = logging_loss / 10
                mean_loss = mean_loss.item()

                wandb.log({"train_loss": mean_loss})
                data_loader_tqdm.set_description(f"Epoch {epoch}, loss: {mean_loss}")

                logging_loss = torch.tensor(0.0).cuda()

            if step > 0 and step % valid_steps == 0:
                valid_score = evaluate(valid_dataloader, model)
                wandb.log({"valid_loss": valid_score})

                if valid_score < best_score:
                    best_score = valid_score

                    torch.save(
                        model.state_dict(),
                        os.path.join(output_path, f"best_model.pth"),
                    )

        torch.save(
            model.state_dict(), os.path.join(output_path, f"last_model_ep_{epoch}.pth")
        )
