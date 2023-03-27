import json
import os
import random
import shutil

if __name__ == "__main__":
    root = "./diffusion/images/"
    move_file = False

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

    visited = set()
    tlens = []
    for caption in train_captions:
        visited.add(caption["text"].strip())
        tlens.append(len(caption["text"].strip()))
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join("./diffusion/train", caption["file_name"])
        if move_file:
            shutil.move(_from, _to)

    dup_cnt = 0
    vlens = []
    for caption in valid_captions:
        if caption["text"] in visited:
            print(caption)
            dup_cnt += 1
            continue
        vlens.append(len(caption["text"].strip()))
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join("./diffusion/validation", caption["file_name"])
        if move_file:
            shutil.move(_from, _to)

    print(dup_cnt)
    print("tlens", sum(tlens) / len(tlens))
    print("vlens", sum(vlens) / len(vlens))
    print("num train samples", len(tlens))
    print("num valid samples", len(vlens))

    if move_file:
        with open("./diffusion/train/metadata.jsonl", "w") as f:
            for caption in train_captions:
                caption["file_name"] = caption["file_name"]
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")

        with open("./diffusion/validation/metadata.jsonl", "w") as f:
            for caption in valid_captions:
                caption["file_name"] = caption["file_name"]
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")
