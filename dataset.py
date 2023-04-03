import albumentations as A
import cv2
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN_RGB = [0.485, 0.456, 0.406]
IMAGENET_STD_RGB = [0.229, 0.224, 0.225]


def get_transformation_for_train(image_size) -> A.Compose:
    """이미지 augmentation 로직을 만들어주는 함수입니다.
    이미지의 내용을 해치지 않는 선에서 밝기, 대조, 블러, 회전 등에 대한 augmentation 을 수행합니다.
    :return: 이미지 변환을 수행해주는 Compose class
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5
            ),
            A.Resize(image_size[0], image_size[1]),
            A.Cutout(
                max_h_size=int(image_size[0] * 0.1),
                max_w_size=int(image_size[1] * 0.1),
                num_holes=1,
                p=0.5,
            ),
        ]
    )
    return transform


class DiffusionDataset(Dataset):
    def __init__(self, df, transform, image_size=(224, 224)):
        self.df = df
        self.transform = transform
        self.img_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # read image using cv2
        image = cv2.imread(row["filepath"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        image = image / 255.0
        image = (image - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float)

        prompt = row["prompt"]
        return image, prompt


class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, show_progress_bar=False, convert_to_tensor=True
        )
        return images, prompt_embeddings


def get_dataloaders(trn_df, val_df, image_size, batch_size, use_aug):
    train_transform = None
    if use_aug:
        train_transform = get_transformation_for_train(image_size)
    trn_dataset = DiffusionDataset(trn_df, train_transform, image_size)
    val_dataset = DiffusionDataset(val_df, None, image_size)
    collator = DiffusionCollator()

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collator,
    )
    dataloaders["val"] = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        collate_fn=collator,
    )
    return dataloaders
