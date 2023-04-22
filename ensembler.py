import json
import os
import random

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from run_train_vit import HFVitModel, cosine_similarity, seed_everything


def get_transformation_for_test(image_size) -> A.Compose:
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.0),  # NOTE 임시 수정
        ]
    )
    return transform


class DiffusionTestDataset(Dataset):
    def __init__(
        self,
        file_paths,
        transform,
        image_size=(224, 224),
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    ):
        self.file_paths = file_paths
        self.transform = transform
        self.img_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = str(self.file_paths[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        image = image / 255.0
        image = (image - self.image_mean) / self.image_std
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float)
        return image


def predict(
    file_paths,
    model_path,
    model_name,
    input_size,
    image_mean,
    image_std,
    batch_size,
    dropout_rate,
    activation,
    hidden_size=-1,
    use_hf_model=True,
    use_layernorm=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = (input_size, input_size)
    transform = get_transformation_for_test(img_size)
    dataset = DiffusionTestDataset(
        file_paths, transform, img_size, image_mean, image_std
    )
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
    )

    if use_hf_model:
        print("hf model")
        model = HFVitModel(
            model_name,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            activation=activation,
            use_layernorm=use_layernorm,
        )
    else:
        print("timm model")
        model = timm.create_model(model_name, pretrained=False, num_classes=384)
        if dropout_rate > 0.0:
            print("w head", dropout_rate, activation, "use layer norm", use_layernorm)
            if use_layernorm:
                model.haed = torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout_rate),
                    torch.nn.Linear(model.head.in_features, model.head.in_features),
                    torch.nn.ReLU() if activation == "relu" else torch.nn.GELU(),
                    torch.nn.Linear(model.head.in_features, 384),
                    torch.nn.LayerNorm(384),
                )
            else:
                model.haed = torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout_rate),
                    torch.nn.Linear(model.head.in_features, model.head.in_features),
                    torch.nn.ReLU() if activation == "relu" else torch.nn.GELU(),
                    torch.nn.Linear(model.head.in_features, 384),
                )
        else:
            print("wo head")

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tta_preds = []
    for _ in range(1):
        preds = []
        for X in tqdm(dataloader):
            X = X.to(device)

            with torch.no_grad():
                X_out = model(X)
                preds.append(X_out.cpu())

        preds = torch.cat(preds, dim=0)
        tta_preds.append(preds)

    del model, dataloader, dataset
    torch.cuda.empty_cache()

    return torch.cat(tta_preds, dim=0)


#     tta_preds = None
#     for _ in range(2):
#         preds = []
#         for X in tqdm(dataloader, leave=False):
#             X = X.to(device)

#             with torch.no_grad():
#                 X_out = model(X)
#                 preds.append(X_out.cpu().numpy())

#         if tta_preds is None:
#             tta_preds = np.vstack(preds).flatten()
#         else:
#             tta_preds += np.vstack(preds).flatten()

#     return tta_preds / 2


def make_vit_embeddings(
    items,
    model_path,
    model_name,
    input_size,
    image_mean,
    image_std,
    dropout_rate,
    batch_size=64,
    activation="relu",
    hidden_size=-1,
    use_hf_model=False,
    use_layernorm=False,
):
    root_path = "./diffusion/image-to-prompt-train-valid-split-v7/validation/"

    prompt_embeddings = predict(
        [os.path.join(root_path, item["file_name"]) for item in items],
        model_path,
        model_name,
        input_size,
        image_mean,
        image_std,
        batch_size,
        dropout_rate,
        activation,
        hidden_size,
        use_hf_model,
        use_layernorm=use_layernorm,
    )

    return prompt_embeddings


def parse_config(path):
    with open(path) as f:
        config = json.loads(f.read())
    return config


def normalize(embeds):
    embeds = embeds.reshape(-1, 384)
    return embeds / np.linalg.norm(embeds, ord=2, axis=1, keepdims=True)


if __name__ == "__main__":
    seed_everything(42)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    items = []
    with open(
        "./diffusion/image-to-prompt-train-valid-split-v7/validation/metadata.jsonl"
    ) as f:
        for line in f:
            items.append(json.loads(line))

    random.shuffle(items)
    items = items[-2000:]

    # [1] huge model 3ep
    model_root = "laion-CLIP-ViT-H-14-laion2B-s32B-b79K_on_v7_no_head/tmp-3ep/"
    model_path = f"{model_root}/laion-CLIP-ViT-H-14-laion2B-s32B-b79K_best.pth"
    config = parse_config(f"{model_root}/train_conf.json")
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    input_size = config["image_size"][0]
    image_mean = config["image_mean"]
    image_std = config["image_std"]
    dropout_rate = config["dropout_rate"]
    hidden_size = config["hidden_size"]

    embeddings1 = make_vit_embeddings(
        items,
        model_path,
        model_name,
        input_size,
        image_mean,
        image_std,
        dropout_rate,
        hidden_size=hidden_size,
        use_hf_model=True,
    )

    # [2] large model
    model_root = "laion-CLIP-ViT-L-14-laion2B-s32B-b82K_on_v7_no_head/tmp/"
    model_path = f"{model_root}/laion-CLIP-ViT-L-14-laion2B-s32B-b82K_best.pth"
    config = parse_config(f"{model_root}/train_conf.json")
    model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    input_size = config["image_size"][0]
    image_mean = config["image_mean"]
    image_std = config["image_std"]
    dropout_rate = config["dropout_rate"]
    hidden_size = config["hidden_size"]

    embeddings2 = make_vit_embeddings(
        items,
        model_path,
        model_name,
        input_size,
        image_mean,
        image_std,
        dropout_rate,
        hidden_size=hidden_size,
        use_hf_model=True,
    )

    # [3] large 336 model
    model_root = "openai-clip-vit-large-patch14-336_on_v7_wo_head/tmp/"
    model_path = f"{model_root}/openai-clip-vit-large-patch14-336_3ep.pth"
    config = parse_config(f"{model_root}/train_conf.json")
    model_name = "openai/clip-vit-large-patch14-336"
    input_size = config["image_size"][0]
    image_mean = config["image_mean"]
    image_std = config["image_std"]
    dropout_rate = config["dropout_rate"]
    hidden_size = config["hidden_size"]

    embeddings3 = make_vit_embeddings(
        items,
        model_path,
        model_name,
        input_size,
        image_mean,
        image_std,
        dropout_rate,
        hidden_size=hidden_size,
        use_hf_model=True,
    )

    # [4] huge with head
    model_root = "laion-CLIP-ViT-H-14-laion2B-s32B-b79K_on_v7_w_head"
    model_path = f"{model_root}/laion-CLIP-ViT-H-14-laion2B-s32B-b79K_best.pth"
    config = parse_config(f"{model_root}/train_conf.json")
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    input_size = config["image_size"][0]
    image_mean = config["image_mean"]
    image_std = config["image_std"]
    dropout_rate = config["dropout_rate"]
    hidden_size = config["hidden_size"]

    embeddings4 = make_vit_embeddings(
        items,
        model_path,
        model_name,
        input_size,
        image_mean,
        image_std,
        dropout_rate,
        hidden_size=hidden_size,
        use_hf_model=True,
    )

    apply_normalize = False

    outputs = [embeddings1, embeddings2, embeddings3, embeddings4]

    if apply_normalize:
        outputs = [normalize(output) for output in outputs]

    gt_texts = [item["text"] for item in items]
    gt_embeddings = torch.tensor(st_model.encode(gt_texts))

    best_score = -1
    best_weights = None
    for i in range(10000):
        weights = []
        for i in range(len(outputs)):
            r = random.uniform(0, 1)
            weights.append(r)

        total = sum(weights)
        weights = [x / total for x in weights]

        ensemble_embeddings = sum([o * w for o, w in zip(outputs, weights)])

        scores = cosim(ensemble_embeddings, gt_embeddings)

        score = sum(scores) / len(scores)

        if best_score < score:
            best_score = score
            best_weights = weights
            print(best_score, "with", weights)

    # TODO: 부스팅 학습
