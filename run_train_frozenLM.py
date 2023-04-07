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
    get_linear_schedule_with_warmup,
)

from models.dataset import ImageTextDataset, get_transformation_for_train
from models.model import EncoderDecoder, LanguageModel, VisionModel
from common.utils import TQDM_FORMAT, create_logger


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
    logger = create_logger(__name__)

    # Model setting
    print(f"[+] Load VLM model")

    vm_name = "openai/clip-vit-large-patch14"
    lm_name = "Gustavosta/MagicPrompt-Stable-Diffusion"

    config = GPT2Config.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
    tokenizer = GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
    lm = LanguageModel.from_pretrained(lm_name, config=config)
    vm = VisionModel(model_name=vm_name, out_features=config.n_embd)

    # Add the special PAD token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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
    image_size = (224, 224)
    max_seq_length = 128
    train_data_path = "datasets/train/metadata.jsonl"
    with open(train_data_path) as f:
        train_data = [json.loads(d) for d in f.readlines()]
    train_transform = get_transformation_for_train(image_size)
    train_dataset = ImageTextDataset(
        train_data,
        tokenizer,
        max_seq_length=max_seq_length,
        transform=train_transform,
        image_size=image_size,
        root_path="datasets/train",
    )

    validation_data_path = "datasets/validation/metadata.jsonl"
    with open(validation_data_path) as f:
        validation_data = [json.loads(d) for d in f.readlines()]
    validation_dataset = ImageTextDataset(
        validation_data,
        tokenizer,
        max_seq_length=max_seq_length,
        image_size=image_size,
        root_path="datasets/validation",
    )

    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, batch_size=batch_size
    )

    # Make train code
    print(f"[+] Start to train")
    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * num_training_steps)
    validation_step = 200  # num_training_steps //10
    train_loss_check_step = 50  # len(train_dataloader) //5
    best_loss = 1000.0
    step = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    # use amp
    use_amp = True
    scaler = GradScaler(enabled=use_amp)

    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch in tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"[Epoch]:{epoch}",
            bar_format=TQDM_FORMAT,
        ):
            step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_token_mask = batch["image_token_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["input_ids"].to(device)

            # Forward pass
            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_token_mask=image_token_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
            loss = output.loss
            scaler.scale(loss).backward()

            # Unscale gradient first, then clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update accumulated gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            if step % train_loss_check_step == 0:
                logger.info(
                    f"Steps: {step:03d}\tTrin Loss: {train_loss/(train_loss_check_step)}"
                )
                train_loss = 0.0

            # Perform validation
            if step % validation_step == 0:
                model.eval()
                val_loss = 0
                num_val_batches = 0

                with torch.no_grad():
                    for val_batch in tqdm(
                        validation_dataloader,
                        total=len(validation_dataloader),
                        desc=f"[Epoch]:{epoch}",
                        bar_format=TQDM_FORMAT,
                    ):
                        num_val_batches += 1
                        input_ids = val_batch["input_ids"].to(device)
                        attention_mask = val_batch["attention_mask"].to(device)
                        pixel_values = val_batch["pixel_values"].to(device)
                        labels = val_batch["input_ids"].to(device)

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            image_token_mask=image_token_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                        )
                        val_loss += outputs.loss.item()

                val_loss /= num_val_batches
                logger.info(f"Validation Loss: {val_loss}")

                if best_loss > val_loss:
                    # save model
                    torch.save(model.state_dict(), "best.ckpt")

                model.train()

    print(f"[+] Finished training")
