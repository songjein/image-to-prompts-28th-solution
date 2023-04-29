import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

if __name__ == "__main__":
    batch_size = 16
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model = CLIPModel.from_pretrained(model_name).cuda()
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    root = "diffusion/image-to-prompt-train-valid-split-v7/validation"
    clip_score_path = "./resources/dissim_pairs_all_retry-valid.txt"

    meta_path = os.path.join(root, "metadata.jsonl")

    items = []
    with open(meta_path) as f:
        for line in f:
            items.append(json.loads(line))

    lines = []
    with open(clip_score_path, "w", encoding="utf-8") as f:
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

                score = float(score)
                item = items[index]
                text = item["text"]
                file_name = item["file_name"]
                record = f"{index}\t{score}\t{text}\t{file_name}\n"
                f.write(record)
                lines.append(record)
                print(record)

    # 같은 순서로 만들어짐
    assert len(items) == len(lines)

    # 굳이 읽을 필요 없는데 일단 별도로 저장 후 읽음
    filtered_items = []
    with open(clip_score_path) as f:
        for idx, line in enumerate(f):
            item = items[idx]
            index, score, text, file_name = line.strip().split("\t")
            assert item["file_name"] == file_name
            item["clip_score"] = float(score)

            if float(score) > 0.3:
                filtered_items.append(item)

    with open(meta_path, "w", encoding="utf-8") as f:
        for item in filtered_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("orig samples:", len(items))
    print("filtered samples:", len(filtered_items))
