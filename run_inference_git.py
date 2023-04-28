import json
import os
import time
from glob import glob

import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

if __name__ == "__main__":
    beam_size = 2
    num_example = 500

    processor = AutoProcessor.from_pretrained(f"microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained(f"microsoft/git-base").cuda()
    model.load_state_dict(
        torch.load(
            f"./output_git-model-42s-5ep-microsoft/git-base-on-v7/best_model.pth"
        )
    )

    data_root = "./diffusion/image-to-prompt-train-valid-split-v7/"

    id2text = dict()
    with open(os.path.join(data_root, "validation/metadata.jsonl")) as f:
        for line in f:
            item = json.loads(line)
            id2text[item["file_name"]] = item["text"]

    filepaths = sorted(glob(os.path.join(data_root, "validation/*.jpg")))

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    print("start to inf")
    stime = time.time()
    batch_size = 128
    pred_texts = []
    for idx in tqdm(range(0, len(filepaths), batch_size)):
        pixel_values = None
        for idx, filepath in enumerate(filepaths[idx : idx + batch_size]):
            filename = filepath.split("/")[-1]
            image = Image.open(filepath)
            pv = processor(images=image, return_tensors="pt").pixel_values

            if pixel_values is None:
                pixel_values = pv
            else:
                pixel_values = torch.vstack((pixel_values, pv))

        pixel_values = pixel_values.cuda()

        # 길이 77로 했을 때  0.4346
        # 길이 16-32로 했을 때 0.439
        # 길이 32-48 했을 때 0.442
        if True:
            generated_ids = model.generate(
                pixel_values=pixel_values,
                min_length=32,
                max_length=64,
                do_sample=True,
                top_p=0.05,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )
        else:
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=77,
                num_beams=beam_size,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )

        _pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        pred_texts.extend(_pred_texts)

    fin_time = time.time() - stime

    gt_texts = [id2text[fp.split("/")[-1]] for fp in filepaths]
    pred_embeddings = torch.tensor(st_model.encode(pred_texts))
    gt_embeddings = torch.tensor(st_model.encode(gt_texts))

    scores = cosim(pred_embeddings, gt_embeddings)

    for pred, gt in zip(pred_texts, gt_texts):
        print("A:", pred)
        print("B:", gt)
        print("----")

    print(f"i: {idx}, avg score: {sum(scores)/len(scores)}")
    print("fin inf: ", fin_time)
