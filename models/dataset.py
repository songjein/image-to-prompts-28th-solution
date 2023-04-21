import os

import albumentations as A
import cv2
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

IMAGENET_MEAN_RGB = [0.485, 0.456, 0.406]
IMAGENET_STD_RGB = [0.229, 0.224, 0.225]


def get_transformation_for_train(image_size) -> A.Compose:
    """이미지 augmentation 로직을 만들어주는 함수입니다.
    이미지의 내용을 해치지 않는 선에서 밝기, 대조, 블러, 회전 등에 대한 augmentation 을 수행합니다.
    :return: 이미지 변환을 수행해주는 Compose class
    """
    transform = A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            # A.ImageCompression(quality_lower=99, quality_upper=100),
            # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            # A.ShiftScaleRotate(
            #    shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5
            # ),
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


class ImageTextDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_length,
        n_image_token,
        transform=None,
        image_size=(224, 224),
        root_path=None,
    ):
        super(ImageTextDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.n_image_token = n_image_token
        self.transform = transform
        self.image_size = image_size
        self.root_path = root_path

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<IMG>")
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load and preprocess the image
        row = self.data[idx]
        if self.root_path is not None:
            image_path = os.path.join(self.root_path, row["file_name"])
        else:
            image_path = row["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        image = image / 255.0
        image = (image - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
        image = image.transpose(2, 0, 1)
        pixel_values = torch.tensor(image, dtype=torch.float)

        # Load and preprocess the text
        # [<IMG> * n_image_token, prompt_texet, EOS]
        text = row["text"]
        prompt_embeddings = self.st_model.encode(text, convert_to_tensor=True)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length - 1 - self.n_image_token,
        )
        input_ids = (
            [self.image_token_id] * self.n_image_token
            + inputs["input_ids"]
            + [self.eos_token_id]
        )
        image_token_mask = [1] * self.n_image_token + [0] * (
            len(input_ids) - self.n_image_token
        )

        # padding ids
        num_padded = self.max_seq_length - len(input_ids)

        # right padding
        padded_input_ids = input_ids + [self.pad_token_id] * num_padded
        padded_label_ids = (
            [-100] * (self.n_image_token)
            + input_ids[self.n_image_token :]
            + [-100] * num_padded
        )
        padded_image_token_mask = image_token_mask + [0] * num_padded

        # Convert input_ids and attention_mask to tensors
        padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        padded_label_ids = torch.tensor(padded_label_ids, dtype=torch.long)
        padded_image_token_mask = torch.tensor(
            padded_image_token_mask, dtype=torch.long
        )

        return {
            "input_ids": padded_input_ids,
            "label_ids": padded_label_ids,
            "image_token_mask": padded_image_token_mask,
            "pixel_values": pixel_values,
            "prompt_embeddings": prompt_embeddings,
        }
