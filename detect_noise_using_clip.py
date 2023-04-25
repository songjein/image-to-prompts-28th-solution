import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

if __name__ == "__main__":
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model = CLIPModel.from_pretrained(model_name).cuda()
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    root = "diffusion/image-to-prompt-train-valid-split-v7/train"

    items = []
    with open(os.path.join(root, "metadata.jsonl")) as f:
        for line in f:
            items.append(json.loads(line))

    batch_size = 16

    with open("./resources/dissim_pairs_025_v7_train.txt", "w", encoding="utf-8") as f:
        for start_index in range(0, len(items), batch_size):
            pixel_values = None
            texts = []
            for item in items[start_index : start_index + batch_size]:
                texts.append(item["text"])
                path = os.path.join(root, item["file_name"])
                image = Image.open(path)
                pv = processor(text=None, images=image, return_tensors="pt")[
                    "pixel_values"
                ]

                if pixel_values is None:
                    pixel_values = pv
                else:
                    pixel_values = torch.vstack((pixel_values, pv))

            pixel_values = pixel_values.cuda()
            image_embeds = model.get_image_features(pixel_values)

            text_inputs = tokenizer(
                texts, return_tensors="pt", max_length=77, truncation=True, padding=True
            )
            text_inputs = {key: val.cuda() for key, val in text_inputs.items()}
            text_embeds = model.get_text_features(**text_inputs)

            scores = cosim(image_embeds, text_embeds)

            for in_batch_index, score in enumerate(scores):
                index = start_index + in_batch_index

                if score < 0.25:
                    score = float(score)
                    item = items[index]
                    text = item["text"]
                    file_path = item["file_name"]
                    record = f"{index}\t{score}\t{text}\t{file_path}\n"
                    f.write(record)
                    print(record)
