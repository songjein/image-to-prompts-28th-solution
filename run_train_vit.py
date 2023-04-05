import json
import os
import random
import shutil
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
from scipy import spatial
from timm.utils import AverageMeter
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from dataset import get_dataloaders

warnings.filterwarnings("ignore")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def cosine_similarity(y_trues, y_preds):
    return np.mean(
        [
            1 - spatial.distance.cosine(y_true, y_pred)
            for y_true, y_pred in zip(y_trues, y_preds)
        ]
    )


def train(
    train_df,
    valid_df,
    model_name,
    input_size,
    batch_size,
    num_epochs,
    lr,
    lr_scaling_factor,
    dropout_rate,
    output_path,
    scheduler,
    warmup_steps,
    use_aug,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(train_df, valid_df, input_size, batch_size, use_aug)

    model = timm.create_model(model_name, pretrained=True, num_classes=384)
    if dropout_rate > 0.0:
        model.haed = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(model.head.in_features, model.head.in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(model.head.in_features, 384),
        )

    fp_log = open(os.path.join(output_path, "logs.txt"), "w", encoding="utf-8")

    model.set_grad_checkpointing()
    model.to(device)

    if lr_scaling_factor is not None:
        lr_dict = dict()
        for name, _ in model.named_parameters():
            if "head" in name:
                lr_dict["head"] = lr
            else:
                lr_dict["backbone"] = lr * lr_scaling_factor

        optimizer_params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "head" in name
                ],
                "lr": lr_dict["head"],
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "head" not in name
                ],
                "lr": lr_dict["backbone"],
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        print("set cosine annealing lr scheduler")
        fp_log.write("set cosine annealing lr scheduler\n")
    elif scheduler == "CosineSchedulerWithWarmup":
        steps_per_epoch = len(dataloaders["train"])
        num_training_steps = num_epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=warmup_steps,
        )
        print("set cosine scheduler with warmup lr scheduler")
        fp_log.write("set cosine scheduler with warmup lr scheduler\n")

    criterion = nn.CosineEmbeddingLoss()

    best_score = -1.0

    for epoch in range(num_epochs):
        train_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter(),
        }
        model.train()
        for X, y in tqdm(dataloaders["train"]):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            X_out = model(X)
            target = torch.ones(X.size(0)).to(device)
            loss = criterion(X_out, y, target)
            loss.backward()

            optimizer.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(), y.detach().cpu().numpy()
            )

            train_meters["loss"].update(trn_loss, n=X.size(0))
            train_meters["cos"].update(trn_cos, n=X.size(0))

        scheduler.step()

        log = "Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}".format(
            epoch + 1, train_meters["loss"].avg, train_meters["cos"].avg
        )
        print(log)
        fp_log.write(log + "\n")

        val_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter(),
        }
        model.eval()
        for X, y in tqdm(dataloaders["val"]):
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)

                val_loss = loss.item()
                val_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(), y.detach().cpu().numpy()
                )

            val_meters["loss"].update(val_loss, n=X.size(0))
            val_meters["cos"].update(val_cos, n=X.size(0))

        log = "Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}".format(
            epoch + 1, val_meters["loss"].avg, val_meters["cos"].avg
        )
        print(log)
        fp_log.write(log + "\n")

        if val_meters["cos"].avg > best_score:
            best_score = val_meters["cos"].avg
            torch.save(model.state_dict(), f"{output_path}/{model_name}_best.pth")

        torch.save(model.state_dict(), f"{output_path}/{model_name}_{epoch}ep.pth")


if __name__ == "__main__":

    class CFG:
        model_name = "vit_large_patch14_clip_224.openai_ft_in12k_in1k"
        input_size = (224, 224)
        batch_size = 256
        num_epochs = 5
        lr = 1e-4
        seed = 42
        lr_scaling_factor: Optional[float] = None
        dropout_rate = 0.1
        scheduler = "CosineAnnealingLR"
        warmup_steps = 200
        use_aug = True

        output_path = f"{model_name}_on_v5_aug_do01_2fc"
        train_metadata_file = "metadata.jsonl"
        valid_metadata_file = "metadata.jsonl"

        train_dir = "./diffusion/image-to-prompt-train-valid-split-v5/train"
        valid_dir = "./diffusion/image-to-prompt-train-valid-split-v5/validation"

    assert CFG.scheduler in ["CosineSchedulerWithWarmup", "CosineAnnealingLR"]

    seed_everything(CFG.seed)

    os.makedirs(CFG.output_path, exist_ok=True)
    shutil.copy("./run_train_vit.py", os.path.join(CFG.output_path, "run_train_vit.py"))
    shutil.copy("./dataset.py", os.path.join(CFG.output_path, "dataset.py"))
    with open(
        os.path.join(CFG.output_path, "train_conf.json"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps(vars(CFG())))

    with open(os.path.join(CFG.train_dir, CFG.train_metadata_file)) as f:
        train_data = {
            "filepath": [],
            "prompt": [],
        }
        for line in f:
            item = json.loads(line)
            train_data["filepath"].append(
                os.path.join(CFG.train_dir, item["file_name"])
            )
            train_data["prompt"].append(item["text"])

        train_df = pd.DataFrame.from_dict(train_data)

    with open(os.path.join(CFG.valid_dir, CFG.valid_metadata_file)) as f:
        validation_data = {
            "filepath": [],
            "prompt": [],
        }
        for line in f:
            item = json.loads(line)
            validation_data["filepath"].append(
                os.path.join(CFG.valid_dir, item["file_name"])
            )
            validation_data["prompt"].append(item["text"])

        valid_df = pd.DataFrame.from_dict(validation_data)

    train(
        train_df,
        valid_df,
        CFG.model_name,
        CFG.input_size,
        CFG.batch_size,
        CFG.num_epochs,
        CFG.lr,
        CFG.lr_scaling_factor,
        CFG.dropout_rate,
        CFG.output_path,
        CFG.scheduler,
        CFG.warmup_steps,
        CFG.use_aug,
    )
