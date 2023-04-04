import json
import os

from langdetect import detect
from tqdm import tqdm

if __name__ == "__main__":
    input_path = "./resources/openprompts_dedup_075_filtered_cross_dedup.txt"
    output_path = "./diffusion/openprompts_images/metadata.jsonl"
    image_dir_path = "./diffusion/openprompts_images"

    captions = []
    with open(input_path) as f:
        for idx, line in enumerate(tqdm(f)):
            file_name = f"openprompts_{idx:08d}.jpg"
            path = os.path.join(image_dir_path, file_name)
            if not os.path.exists(path):
                continue
            prompt = line.strip()

            try:
                if detect(prompt) != "en":
                    continue
            except:
                print("exception at", prompt)

            captions.append(
                {
                    "file_name": file_name,
                    "text": prompt,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")
