import json
import os

from langdetect import detect

if __name__ == "__main__":
    captions = []
    with open("./diffusion/prompts-large.txt") as f:
        for idx, line in enumerate(f):
            file_name = f"{idx:08d}.jpg"
            path = os.path.join("./diffusion/images/", file_name)
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

    with open("./diffusion/captions.jsonl", "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")
