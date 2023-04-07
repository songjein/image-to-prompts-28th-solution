import json
import os
import random
import shutil
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
from pydantic import BaseModel
from scipy import spatial
from timm.utils import AverageMeter
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModel, get_cosine_schedule_with_warmup

from dataset import get_dataloaders

warnings.filterwarnings("ignore")


class HFVitModel(nn.Module):
    def __init__(
        self, model_path_or_name, hidden_size=1024, dropout_rate=0.1, activation="relu"
    ):
        super(HFVitModel, self).__init__()

        clip = AutoModel.from_pretrained(model_path_or_name)
        clip.gradient_checkpointing_enable()
        self.vision = clip.vision_model

        if dropout_rate > 0.0:
            self.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU() if activation == "relu" else torch.nn.GELU(),
                torch.nn.Linear(hidden_size, 384),
            )
        else:
            self.fc = torch.nn.Linear(hidden_size, 384)

    def forward(self, x):
        out = self.vision(x)["pooler_output"]
        return self.fc(out)


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
    image_size,
    batch_size,
    num_epochs,
    lr,
    lr_scaling_factor,
    dropout_rate,
    output_path,
    scheduler,
    warmup_steps,
    use_aug,
    use_amp,
    use_hf_model,
    image_mean,
    image_std,
    activation,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(
        train_df,
        valid_df,
        image_size,
        batch_size,
        use_aug,
        image_mean,
        image_std,
    )

    if use_hf_model:
        model = HFVitModel(model_name, dropout_rate=dropout_rate, activation=activation)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=384)
        if dropout_rate > 0.0:
            print("apply dropout", dropout_rate)
            model.haed = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(model.head.in_features, model.head.in_features),
                torch.nn.ReLU() if activation == "relu" else torch.nn.GELU(),
                torch.nn.Linear(model.head.in_features, 384),
            )
        model.set_grad_checkpointing()

    model.to(device)
    fp_log = open(os.path.join(output_path, "logs.txt"), "w", encoding="utf-8")

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

    scaler = GradScaler(enabled=use_amp)

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

            with autocast(enabled=use_amp, dtype=torch.float16):
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            torch.save(
                model.state_dict(),
                f"{output_path}/{model_name.replace('/', '-')}_best.pth",
            )

        torch.save(
            model.state_dict(),
            f"{output_path}/{model_name.replace('/', '-')}_{epoch}ep.pth",
        )


if __name__ == "__main__":

    class Config(BaseModel):
        seed: int = 42

        memo = "on_v5_aug_2fc_1e5"
        model_name: str = "vit_huge_patch14_224_clip_laion2b"

        #: True로 설정시 laion/CLIP-ViT-H-14-laion2B-s32B-b79K 와 같은 허깅페이스 호환모델 전달
        use_hf_model: bool = True
        if use_hf_model:
            model_name = "openai/clip-vit-large-patch14-336"  # laion/CLIP-ViT-H-14-laion2B-s32B-b79K
        image_size: Tuple[int, int] = (336, 336)
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        batch_size: int = 128
        num_epochs: int = 5
        lr: float = 1e-5  # large: 1e-4, huge: 1e-5
        lr_scaling_factor: Optional[float] = None
        dropout_rate: float = -0.1
        scheduler: str = "CosineAnnealingLR"
        warmup_steps: int = 200
        use_aug: bool = True
        use_amp: bool = True
        activation = "gelu"

        output_path: str = f"{model_name.replace('/', '-')}_{memo}"
        train_metadata_file: str = "metadata.jsonl"
        valid_metadata_file: str = "metadata.jsonl"

        train_dir: str = "./diffusion/image-to-prompt-train-valid-split-v5/train"
        valid_dir: str = "./diffusion/image-to-prompt-train-valid-split-v5/validation"

    config = Config()

    assert config.scheduler in ["CosineSchedulerWithWarmup", "CosineAnnealingLR"]

    seed_everything(config.seed)

    os.makedirs(config.output_path, exist_ok=True)
    assert os.path.exists(config.output_path)
    shutil.copy(
        "./run_train_vit.py", os.path.join(config.output_path, "run_train_vit.py")
    )
    shutil.copy("./dataset.py", os.path.join(config.output_path, "dataset.py"))
    with open(
        os.path.join(config.output_path, "train_conf.json"), "w", encoding="utf-8"
    ) as f:
        cfg_str = json.dumps(vars(config), indent=4)
        print(cfg_str)
        f.write(cfg_str)

    with open(os.path.join(config.train_dir, config.train_metadata_file)) as f:
        train_data = {
            "filepath": [],
            "prompt": [],
        }
        for line in f:
            item = json.loads(line)
            train_data["filepath"].append(
                os.path.join(config.train_dir, item["file_name"])
            )
            train_data["prompt"].append(item["text"])

        train_df = pd.DataFrame.from_dict(train_data)

    with open(os.path.join(config.valid_dir, config.valid_metadata_file)) as f:
        validation_data = {
            "filepath": [],
            "prompt": [],
        }
        for line in f:
            item = json.loads(line)
            validation_data["filepath"].append(
                os.path.join(config.valid_dir, item["file_name"])
            )
            validation_data["prompt"].append(item["text"])

        valid_df = pd.DataFrame.from_dict(validation_data)

    train(
        train_df,
        valid_df,
        config.model_name,
        config.image_size,
        config.batch_size,
        config.num_epochs,
        config.lr,
        config.lr_scaling_factor,
        config.dropout_rate,
        config.output_path,
        config.scheduler,
        config.warmup_steps,
        config.use_aug,
        config.use_amp,
        config.use_hf_model,
        config.image_mean,
        config.image_std,
        config.activation,
    )
