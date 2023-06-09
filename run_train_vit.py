import json
import os
import random
import shutil
import sys
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel
from scipy import spatial
from timm.utils import AverageMeter
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoModel, get_cosine_schedule_with_warmup

from dataset import get_dataloaders

warnings.filterwarnings("ignore")


class LayerwiseDecayAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        model,
        base_lr,
        min_lr=1e-8,  # 1e-6으로도 해보고 싶긴 함
        backbone_weight_decay=1e-3,
        head_weight_decay=1e-5,
        head_lr_factor=10.0,
    ):
        params = []

        layers = [model.vision.embeddings] + list(model.vision.encoder.layers)
        learning_rates = np.linspace(base_lr, min_lr, len(layers))
        for idx, layer in enumerate(reversed(layers)):
            lr = learning_rates[idx]
            params += [
                {
                    "params": layer.parameters(),
                    "lr": lr,
                    "weight_decay": backbone_weight_decay,
                }
            ]

        head_lr = base_lr * head_lr_factor
        params += [
            {
                "params": model.fc.parameters(),
                "lr": head_lr,
                "weight_decay": head_weight_decay,
            }
        ]

        super(LayerwiseDecayAdamW, self).__init__(
            params, defaults=dict(weight_decay=backbone_weight_decay)
        )
        self._optimizer = torch.optim.AdamW(self.param_groups)

    def step(self, closure=None):
        return self._optimizer.step(closure=closure)

    def zero_grad(self, set_to_none=False):
        return self._optimizer.zero_grad(set_to_none=set_to_none)


class MultisampledDropout(nn.Module):
    def __init__(self):
        super(MultisampledDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout((i + 1) * 0.1) for i in range(5)])

    def forward(self, x, module):
        return torch.mean(
            torch.stack([module(dropout(x)) for dropout in self.dropouts], dim=0), dim=0
        )


class HFVitModel(nn.Module):
    def __init__(
        self,
        model_path_or_name,
        hidden_size=1024,
        dropout_rate=0.1,
        use_complex_head=False,
        use_ms_dropout=False,
    ):
        super(HFVitModel, self).__init__()

        if "vit" in model_path_or_name.lower() and "clip" in model_path_or_name.lower():
            clip = AutoModel.from_pretrained(model_path_or_name)
            clip.gradient_checkpointing_enable()
            self.vision = clip.vision_model
        else:
            self.vision = AutoModel.from_pretrained(model_path_or_name)
            self.vision.gradient_checkpointing_enable()

        # NOTE: temp
        # for param in self.vision.parameters():
        #     param.requires_grad = False

        if use_complex_head:
            print("w complex head")
            self.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size, 384),
            )
        else:
            print("w simple head")
            self.fc = torch.nn.Linear(hidden_size, 384)

        self.use_ms_dropout = use_ms_dropout
        if use_ms_dropout:
            self.dropout = MultisampledDropout()

    def forward(self, x):
        out = self.vision(x)["pooler_output"]

        if self.use_ms_dropout:
            return self.dropout(out, self.fc)

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


def valid(model, dataloaders, device, criterion, epoch, fp_log):
    val_meters = {
        "loss": AverageMeter(),
        "cos": AverageMeter(),
    }
    model.eval()
    for X, y in tqdm(dataloaders):
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

    torch.cuda.empty_cache()

    return val_meters["loss"].avg, val_meters["cos"].avg


def train(
    train_df,
    valid_df,
    model_name,
    image_size,
    batch_size,
    grad_accum_steps,
    num_epochs,
    lr,
    use_layerwise_lr_decay,
    output_path,
    scheduler_type,
    warmup_steps,
    valid_steps,
    use_aug,
    use_amp,
    image_mean,
    image_std,
    hidden_size,
    dropout_rate,
    use_complex_head=False,
    use_ms_dropout=False,
    weight_decay=1e-4,
    backbone_weight_decay=5e-2,
    head_weight_decay=1e-5,
    milestones=[],
    mix_embeds=False,
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
        mix_embeds=mix_embeds,
    )

    model = HFVitModel(
        model_name,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        use_complex_head=use_complex_head,
        use_ms_dropout=use_ms_dropout,
    )

    # NOTE: temp
    # model.load_state_dict(
    #     torch.load(
    #         "./openai-clip-vit-large-patch14-336_on_v7e_extra_gpt_data/openai-clip-vit-large-patch14-336_1ep.pth"
    #     )
    # )

    model.to(device)
    fp_log = open(os.path.join(output_path, "logs.txt"), "w", encoding="utf-8")

    if use_layerwise_lr_decay:
        if scheduler_type == "MultiStepLR":
            raise Exception("MultiStepLR은 layerwise lr decay 구현 안되어 있음")
        else:
            optimizer = LayerwiseDecayAdamW(
                model,
                base_lr=lr,
                backbone_weight_decay=backbone_weight_decay,
                head_weight_decay=head_weight_decay,
            )
            print("Layerwise LR Decay with AdamW")
    else:
        if scheduler_type == "MultiStepLR":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        print("set cosine annealing lr scheduler")
        fp_log.write("set cosine annealing lr scheduler\n")
    elif scheduler_type == "CosineSchedulerWithWarmup":
        steps_per_epoch = len(dataloaders["train"]) // grad_accum_steps
        num_training_steps = num_epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=warmup_steps,
        )
        print("set cosine scheduler with warmup lr scheduler")
        fp_log.write("set cosine scheduler with warmup lr scheduler\n")
    elif scheduler_type == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False
        )
    else:
        raise Exception("Invalid scheduler")

    criterion = nn.CosineEmbeddingLoss()

    scaler = GradScaler(enabled=use_amp)

    best_score = -1.0

    logging_loss = torch.tensor(0.0).cuda()
    step, accumulated_steps = 0, 0
    for epoch in range(num_epochs):
        train_meters = {
            "loss": AverageMeter(),
            "cos": AverageMeter(),
        }

        data_loader_tqdm = tqdm(dataloaders["train"], file=sys.stdout)
        for X, y in data_loader_tqdm:
            model.train()

            accumulated_steps += 1
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                X_out = model(X)
                target = torch.ones(X.size(0)).to(device)
                loss = criterion(X_out, y, target)

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

            if scheduler_type not in ["MultiStepLR", "CosineAnnealingLR"]:
                scheduler.step()

            if step > 0 and step % 10 == 0:
                mean_loss = logging_loss / 10
                mean_loss = mean_loss.item()
                trn_cos = cosine_similarity(
                    X_out.detach().cpu().numpy(), y.detach().cpu().numpy()
                )
                data_loader_tqdm.set_description(f"Epoch {epoch}, loss: {mean_loss}")

                train_meters["loss"].update(mean_loss, n=X.size(0))
                train_meters["cos"].update(trn_cos, n=X.size(0))
                logging_loss = torch.tensor(0.0).cuda()

            if valid_steps > 0 and step % valid_steps == 0:
                val_loss, val_cos = valid(
                    model, dataloaders["val"], device, criterion, epoch, fp_log
                )

                if val_cos > best_score:
                    best_score = val_cos
                    torch.save(
                        model.state_dict(),
                        f"{output_path}/{model_name.replace('/', '-')}_best.pth",
                    )

        if scheduler_type in ["MultiStepLR", "CosineAnnealingLR"]:
            scheduler.step()

        log = "Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}".format(
            epoch + 1, train_meters["loss"].avg, train_meters["cos"].avg
        )
        print(log)
        fp_log.write(log + "\n")

        val_loss, val_cos = valid(
            model, dataloaders["val"], device, criterion, epoch, fp_log
        )

        if val_cos > best_score:
            best_score = val_cos
            torch.save(
                model.state_dict(),
                f"{output_path}/{model_name.replace('/', '-')}_best.pth",
            )

        torch.save(
            model.state_dict(),
            f"{output_path}/{model_name.replace('/', '-')}_{epoch}ep.pth",
        )


def bulid_dataframe(
    images_dir,
    metadata_fn="metadata.jsonl",
    target_label="text",
):
    metadata_path = os.path.join(images_dir, metadata_fn)
    print(f"[build_dataframe] process metadata_path: {metadata_path}")

    with open(metadata_path) as f:
        data_dict = {
            "filepath": [],
            "prompt": [],
            "orig_prompt": [],
        }
        for idx, line in enumerate(f):
            item = json.loads(line)
            if "hdcd" in item["file_name"]:
                continue
            # 0.5 확률로 파일명에 "gpt"를 포함하면 continue
            # if (
            #     "train" in images_dir
            #     and "gpt" in item["file_name"]
            #     and random.random() < 0.5
            # ):
            #     continue
            data_dict["filepath"].append(os.path.join(images_dir, item["file_name"]))
            data_dict["prompt"].append(item[target_label])
            data_dict["orig_prompt"].append(item["orig_text"])
        df = pd.DataFrame.from_dict(data_dict)

    return df


if __name__ == "__main__":

    class Config(BaseModel):
        seed: int = 42

        memo = "on_v7_coco_half_gpt"
        model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

        hidden_size = 768
        image_size: Tuple[int, int] = (224, 224)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        if model_name == "openai/clip-vit-large-patch14-336":
            hidden_size = 1024
            image_size = (336, 336)
        elif model_name == "laion/CLIP-ViT-H-14-laion2B-s32B-b79K":
            hidden_size = 1280
        elif model_name == "microsoft/swin-large-patch4-window7-224-in22k":
            hidden_size = 1536
            image_size = (224, 224)
        elif model_name == "microsoft/swinv2-large-patch4-window12-192-22k":
            hidden_size = 1536
            image_size = (192, 192)
        elif model_name == "microsoft/swin-large-patch4-window12-384-in22k":
            hidden_size = 1536
            image_size = (384, 384)
        if model_name == "laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
            hidden_size = 1024

        if model_name in [
            "vit_large_patch14_224_clip_laion2b",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        ]:
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]
        elif (
            model_name == "vit_huge_patch14_224_clip_laion2b"
            or model_name == "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            or model_name == "openai/clip-vit-large-patch14-336"
            or model_name == "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        ):
            image_mean = [0.48145466, 0.4578275, 0.40821073]
            image_std = [0.26862954, 0.26130258, 0.27577711]

        dropout_rate: float = 0.1
        use_ms_dropout: bool = False
        use_complex_head: bool = False

        batch_size: int = 256
        grad_accum_steps = 1
        num_epochs: int = 3
        lr: float = 1e-4
        use_layerwise_lr_decay: bool = True
        scheduler_type: str = "CosineSchedulerWithWarmup"
        weight_decay = 1e-4  # SGD용
        backbone_weight_decay = 1e-4  # 1e-3  # AdamW for 백본 일반화
        head_weight_decay = 1e-5  # AdamW for 헤드 자유도
        milestones = [num_epochs // 3, 2 * num_epochs // 3]  # MultiStepLR용
        warmup_steps: int = 500
        valid_steps: int = 1000

        target_label_name = "text"

        mix_embeds = False

        use_aug: bool = True
        use_amp: bool = True

        output_path: str = f"{model_name.replace('/', '-')}_{memo}"

        train_dir: str = "./diffusion/image-to-prompt-train-valid-split-v7-a/train"
        valid_dir: str = "./diffusion/image-to-prompt-train-valid-split-v7-a/validation"

        extra_train_dirs = [
            "./diffusion/image-to-prompt-extra-v1/train",
            "./diffusion/image-to-prompt-extra-v2/train",
            "./diffusion/diffusiondb-extra/images",
            "./diffusion/openprompts-extra/images",
            "./diffusion/gpt-generated-sd2-v6-v7/images",
            "./diffusion/gpt-generated-sd2-v8",
            "./diffusion/gpt-generated-sd2-v9",  # gpt gen도 sub sample해서 하면 좋을 듯
            "./diffusion/laion/images",
            "./diffusion/cc3m-77-100/images",
            "./diffusion/coco-caps/images",
        ]
        extra_valid_dirs = ["./diffusion/image-to-prompt-extra-v1/validation"]

    config = Config()

    assert config.scheduler_type in [
        "CosineSchedulerWithWarmup",
        "CosineAnnealingLR",
        "MultiStepLR",
    ]

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

    train_df = bulid_dataframe(config.train_dir, target_label=config.target_label_name)

    extra_train_dfs = [train_df]
    for data_dir in config.extra_train_dirs:
        extra_train_dfs.append(
            bulid_dataframe(data_dir, target_label=config.target_label_name)
        )

    train_df = pd.concat(extra_train_dfs).reset_index(drop=True)

    valid_df = bulid_dataframe(config.valid_dir)

    extra_valid_dfs = [valid_df]
    for data_dir in config.extra_valid_dirs:
        extra_valid_dfs.append(bulid_dataframe(data_dir))

    valid_df = pd.concat(extra_valid_dfs).reset_index(drop=True)

    print("train len", len(train_df))
    print("valid len", len(valid_df))
    print(config.memo)

    train(
        train_df,
        valid_df,
        config.model_name,
        config.image_size,
        config.batch_size,
        config.grad_accum_steps,
        config.num_epochs,
        config.lr,
        config.use_layerwise_lr_decay,
        config.output_path,
        config.scheduler_type,
        config.warmup_steps,
        config.valid_steps,
        config.use_aug,
        config.use_amp,
        config.image_mean,
        config.image_std,
        config.hidden_size,
        config.dropout_rate,
        use_complex_head=config.use_complex_head,
        use_ms_dropout=config.use_ms_dropout,
        weight_decay=config.weight_decay,
        backbone_weight_decay=config.backbone_weight_decay,
        head_weight_decay=config.head_weight_decay,
        milestones=config.milestones,
        mix_embeds=config.mix_embeds,
    )
