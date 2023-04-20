import json
import logging
import os
import random

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    GPT2Config,
    GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
)

from models.dataset import ImageTextDataset, get_transformation_for_train
from models.model import EncoderDecoder, LanguageModel, VisionModel
from common.utils import TQDM_FORMAT, create_logger, LayerwiseDecayAdamW


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)

    log_level = logging.INFO
    logger = create_logger(__name__, "logs")

    # Model setting
    print(f"[+] Load VLM model")

    vm_name = "microsoft/swin-large-patch4-window12-384-in22k"  # "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"  # "openai/clip-vit-large-patch14"
    lm_name = "gpt2-medium"  # "Gustavosta/MagicPrompt-Stable-Diffusion"
    n_image_token = 4
    logger.info(f"vm_name:{vm_name}\nlm_name:{lm_name}\nn_image_token:{n_image_token}")

    config = GPT2Config.from_pretrained(lm_name)
    tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
    lm = LanguageModel.from_pretrained(lm_name, config=config)
    vm = VisionModel(model_name=vm_name, out_features=config.n_embd * n_image_token)

    # Add the special PAD token
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<IMG>"])
    # Resize the model's token embeddings to account for the new token(s)
    lm.resize_token_embeddings(len(tokenizer))

    model = EncoderDecoder(
        encoder=vm,
        decoder=lm,
        config={
            "encoder_frozen": False,
            "decoder_frozen": True,
        },
    )

    # Make Dataset
    print(f"[+] Build train/validation dataset")
    image_size = (384, 384)
    max_seq_length = 32
    train_data_path = "datasets/v6/train/metadata.jsonl"
    with open(train_data_path) as f:
        train_data = [json.loads(d) for d in f.readlines()]
    train_transform = get_transformation_for_train(image_size)
    train_dataset = ImageTextDataset(
        train_data,
        tokenizer,
        max_seq_length=max_seq_length,
        n_image_token=n_image_token,
        transform=train_transform,
        image_size=image_size,
        root_path="datasets/v6/train",
    )

    validation_data_path = "datasets/v6/validation/metadata.jsonl"
    with open(validation_data_path) as f:
        validation_data = [json.loads(d) for d in f.readlines()]
    validation_dataset = ImageTextDataset(
        validation_data,
        tokenizer,
        max_seq_length=max_seq_length,
        n_image_token=n_image_token,
        image_size=image_size,
        root_path="datasets/v6/validation",
    )

    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=16,
        num_workers=0,
    )
    logger.info(f"max_seq_length:{max_seq_length}\nbatch_size:{batch_size}")
    # Make train code
    print(f"[+] Start to train")
    num_epochs = 20
    learning_rate = 1e-4
    best_loss = 1000.0
    grad_accum_steps = 16
    step, accumulated_steps = 0, 0
    logger.info(
        f"num_epochs:{num_epochs}\nlearning_rate:{learning_rate}\nbest_loss:{best_loss}\ngrad_accum_steps:{grad_accum_steps}"
    )

    num_training_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * num_training_steps)
    validation_step = len(train_dataloader) // (4 * grad_accum_steps)
    train_loss_check_step = len(train_dataloader) // (10 * grad_accum_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = LayerwiseDecayAdamW(model, base_lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    # use amp
    use_amp = True
    scaler = GradScaler(enabled=use_amp)
    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        train_loss = torch.tensor(0.0).cuda()
        optimizer.zero_grad()
        model.train()
        logger.info(f"[{epoch:02d}Epoch] Train Start")
        for batch in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"[Epoch]:{epoch}",
            bar_format=TQDM_FORMAT,
        ):
            accumulated_steps += 1

            input_ids = batch["input_ids"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            image_token_mask = batch["image_token_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_ids"].to(device)

            # Forward pass
            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                output = model(
                    input_ids=input_ids,
                    # attention_mask=attention_mask,
                    image_token_mask=image_token_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = output.loss
                loss /= grad_accum_steps

            train_loss += loss.detach()
            scaler.scale(loss).backward()

            if accumulated_steps < grad_accum_steps:
                continue

            accumulated_steps = 0
            step += 1

            # Unscale gradient first, then clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update accumulated gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % train_loss_check_step == 0:
                mean_loss = train_loss / train_loss_check_step
                mean_loss = mean_loss.item()

                logger.info(
                    f"Steps: {step * grad_accum_steps:03d}\tTrin Loss: {mean_loss}"
                )
                train_loss = torch.tensor(0.0).cuda()

            # Perform validation
            if step % validation_step == 0:
                model.eval()
                val_loss = 0
                num_val_batches = 0
                torch.cuda.empty_cache()

                with torch.no_grad():
                    for val_batch in tqdm(
                        validation_dataloader,
                        total=len(validation_dataloader),
                        desc=f"[Epoch]:{epoch}",
                        bar_format=TQDM_FORMAT,
                    ):
                        num_val_batches += 1
                        input_ids = val_batch["input_ids"].to(device)
                        # attention_mask = val_batch["attention_mask"].to(device)
                        image_token_mask = val_batch["image_token_mask"].to(device)
                        pixel_values = val_batch["pixel_values"].to(device)
                        labels = val_batch["label_ids"].to(device)

                        outputs = model(
                            input_ids=input_ids,
                            # attention_mask=attention_mask,
                            image_token_mask=image_token_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                        )
                        val_loss += outputs.loss.item()

                val_loss /= num_val_batches
                logger.info(f"Validation Loss: {val_loss}")

                if best_loss > val_loss:
                    # save model
                    print(f"[+] Save best model")
                    torch.save(
                        model.state_dict(),
                        f"results/swin-0418-384-large-v6/best-model.ckpt",
                    )
                    best_loss = val_loss

                model.train()
                torch.cuda.empty_cache()

        torch.save(
            model.state_dict(),
            f"results/swin-0418-384-large-v6/{epoch}epoch-model.ckpt",
        )

    print(f"[+] Finished training")
