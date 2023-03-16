import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DiffusionDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"])
        image = self.transform(image)
        prompt = row["prompt"]
        return image, prompt


class DiffusionCollator:
    def __init__(self):
        self.st_model = SentenceTransformer(
            "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2", device="cpu"
        )

    def __call__(self, batch):
        images, prompts = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(
            prompts, show_progress_bar=False, convert_to_tensor=True
        )
        return images, prompt_embeddings


def get_dataloaders(train_df, valid_df, input_size, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = DiffusionDataset(train_df, transform)
    valid_dataset = DiffusionDataset(valid_df, transform)
    collator = DiffusionCollator()

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
        collate_fn=collator,
    )
    dataloaders["val"] = DataLoader(
        dataset=valid_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False,
        collate_fn=collator,
    )
    return dataloaders
