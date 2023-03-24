import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from timm.utils import AverageMeter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

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

    wandb.login()

    epochs = 50
    batch_size = 32
    valid_batch_size = 32
    learning_rate = 5e-5
    valid_steps = 1000
    warmup_ratio = 0.05
    seed = 42

    wandb.init(
        name="git-model",
        project="learning-equality",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "valid_batch_size": valid_batch_size,
            "warmup_ratio": warmup_ratio,
            # "use_fp16": use_fp16,
            "seed": seed,
        },
    )

    seed_everything(seed)

    train_dataset = load_dataset("imagefolder", data_dir="./diffusion", split="train")
    valid_dataset = load_dataset(
        "imagefolder", data_dir="./diffusion", split="validation"
    )

    processor = AutoProcessor.from_pretrained("microsoft/git-base")

    train_dataset = ImageCaptioningDataset(train_dataset, processor)
    valid_dataset = ImageCaptioningDataset(valid_dataset, processor)
    # input_ids [512], attention_mask [512], pixel_values [3, 224, 224]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)

    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ttl_iters = epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)

    model.train()

    best_score = -1.0
    for epoch in range(epochs):
        data_loader_tqdm = tqdm(train_dataloader, file=sys.stdout)
        for idx, batch in enumerate(data_loader_tqdm):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=input_ids
            )
            loss = outputs.loss

            wandb.log({"train_loss": loss.item()})
            data_loader_tqdm.set_description(f"Epoch {epoch}, loss: {loss.item()}")

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if idx > 0 and idx % valid_steps == 0:
                valid_score = evaluate(valid_dataloader, model)
                wandb.log({"valid_loss": valid_score})

                if valid_score > best_score:
                    best_score = valid_score
                    os.makedirs(f"outputs_{epoch}ep/", exist_ok=True)
                    torch.save(model.state_dict(), f"outputs_{epoch}ep/best_model.pth")

        torch.save(model.state_dict(), f"outputs_{epoch}ep/last_model.pth")
