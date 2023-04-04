import json
import os
import random
import shutil

from tqdm import tqdm


if __name__ == "__main__":
    root = "./diffusion/chatgpt-images/"
    copy_file = False
    make_meta = False

    captions = []
    with open("./diffusion/chatgpt_0330_captions.jsonl") as f:
        for line in f:
            captions.append(json.loads(line))

    random.seed(42)
    random.shuffle(captions)

    train_captions = captions[: int(len(captions) * 0.95)]
    valid_captions = captions[int(len(captions) * 0.95) :]

    train_dir = "./diffusion/chatgpt_0330_train/"
    valid_dir = "./diffusion/chatgpt_0330_validation/"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    visited = set()
    train_lens = []
    _train_captions = []
    for caption in tqdm(train_captions):
        visited.add(caption["text"].strip())
        train_lens.append(len(caption["text"].strip()))
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join(train_dir, caption["file_name"])
        if copy_file:
            shutil.copy(_from, _to)
        _train_captions.append(caption)

    train_captions = _train_captions

    dup_cnt = 0
    valid_lens = []
    _valid_captions = []
    for caption in tqdm(valid_captions):
        if caption["text"] in visited:
            print(caption)
            dup_cnt += 1
            continue
        valid_lens.append(len(caption["text"].strip()))
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join(valid_dir, caption["file_name"])
        if copy_file:
            shutil.copy(_from, _to)
        _valid_captions.append(caption)

    valid_captions = _valid_captions

    print(dup_cnt)
    print("train_lens", sum(train_lens) / len(train_lens))
    print("valid_lens", sum(valid_lens) / len(valid_lens))
    print("num train samples", len(train_lens))
    print("num valid samples", len(valid_lens))

    if make_meta:
        with open(os.path.join(train_dir, "metadata.jsonl"), "w") as f:
            for caption in train_captions:
                caption["file_name"] = caption["file_name"]
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")

        with open(os.path.join(valid_dir, "metadata.jsonl"), "w") as f:
            for caption in valid_captions:
                caption["file_name"] = caption["file_name"]
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")
