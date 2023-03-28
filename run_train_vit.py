import os
import random
import warnings

import numpy as np
import pandas as pd
import timm
import torch
from scipy import spatial
from sklearn.model_selection import train_test_split
from timm.utils import AverageMeter
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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


def train(train_df, valid_df, model_name, input_size, batch_size, num_epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_dataloaders(train_df, valid_df, input_size, batch_size)

    model = timm.create_model(model_name, pretrained=True, num_classes=384)
    model.set_grad_checkpointing()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ttl_iters = num_epochs * len(dataloaders["train"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)
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
            scheduler.step()

            trn_loss = loss.item()
            trn_cos = cosine_similarity(
                X_out.detach().cpu().numpy(), y.detach().cpu().numpy()
            )

            train_meters["loss"].update(trn_loss, n=X.size(0))
            train_meters["cos"].update(trn_cos, n=X.size(0))

        print(
            "Epoch {:d} / trn/loss={:.4f}, trn/cos={:.4f}".format(
                epoch + 1, train_meters["loss"].avg, train_meters["cos"].avg
            )
        )

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

        print(
            "Epoch {:d} / val/loss={:.4f}, val/cos={:.4f}".format(
                epoch + 1, val_meters["loss"].avg, val_meters["cos"].avg
            )
        )

        if val_meters["cos"].avg > best_score:
            best_score = val_meters["cos"].avg
            torch.save(model.state_dict(), f"outputs_{num_epochs}ep/{model_name}.pth")


if __name__ == "__main__":

    class CFG:
        model_name = "vit_base_patch16_224"
        input_size = 224
        batch_size = 64
        num_epochs = 5
        lr = 1e-4
        seed = 42

    seed_everything(CFG.seed)

    df = pd.read_csv("./data/diffusiondb.csv")
    for index, row in df.iterrows():
        df.at[index, "filepath"] = df.at[index, "filepath"].replace(
            "/kaggle/input/", "./data/"
        )

    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=CFG.seed)

    os.makedirs(f"outputs_{CFG.num_epochs}ep", exist_ok=True)
    train(
        train_df,
        valid_df,
        CFG.model_name,
        CFG.input_size,
        CFG.batch_size,
        CFG.num_epochs,
        CFG.lr,
    )
