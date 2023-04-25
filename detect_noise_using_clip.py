
import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

if __name__ == "__main__":

    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # model_name = "openai/clip-vit-base-patch32"
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

    with open("./resources/dissim_pairs.txt", "w", encoding="utf-8") as f:
        for i in range(0, len(items), batch_size):

            pixel_values = None
            texts = []
            for item in items[i: i+batch_size]:
                texts.append(item["text"])
                path = os.path.join(root, item["file_name"])
                image = Image.open(path)
                pv = processor(text=None, images=image, return_tensors="pt", padding=True)["pixel_values"]

                if pixel_values is None:
                    pixel_values = pv
                else:
                    pixel_values = torch.vstack((pixel_values, pv))
            
            pixel_values = pixel_values.cuda()

            image_embeds = model.get_image_features(pixel_values)

            text_inputs = tokenizer(texts, return_tensors="pt", padding=True)
            text_inputs = {key: val.cuda() for key, val in text_inputs.items()}
            text_embeds = model.get_text_features(**text_inputs)

            scores = cosim(image_embeds, text_embeds)

            for j, score in enumerate(scores):
                base = batch_size * i
                index = base + j
                
                if score < 0.25:
                    score = float(score)
                    item = items[index]
                    text = item["text"]
                    file_path = item["file_name"]
                    record = f"{index}\t{score}\t{text}\t{file_path}\n"
                    f.write(record)
                    print(record)
