import json
import os
import random
import shutil

if __name__ == "__main__":
    root = "./diffusion/images/"

    captions = []
    with open("./diffusion/captions.jsonl") as f:
        for line in f:
            captions.append(json.loads(line))

    random.seed(42)
    random.shuffle(captions)

    train_captions = captions[: int(len(captions) * 0.95)]
    valid_captions = captions[int(len(captions) * 0.95) :]

    os.makedirs("./diffusion/train/", exist_ok=True)
    os.makedirs("./diffusion/validation/", exist_ok=True)

    for caption in train_captions:
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join("./diffusion/train", caption["file_name"])
        shutil.move(_from, _to)

    for caption in valid_captions:
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join("./diffusion/validation", caption["file_name"])
        shutil.move(_from, _to)

    with open("./diffusion/train/metadata.jsonl", "w") as f:
        for caption in train_captions:
            caption["file_name"] = caption["file_name"]
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")

    with open("./diffusion/validation/metadata.jsonl", "w") as f:
        for caption in valid_captions:
            caption["file_name"] = caption["file_name"]
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")
