import json
import time
from glob import glob

import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoProcessor

if __name__ == "__main__":
    size = "large"
    memo = f"{size}_beam5"
    top_p = 0.1
    num_beams = 5
    num_example = 1000
    processor = AutoProcessor.from_pretrained(f"microsoft/git-{size}")
    model = AutoModelForCausalLM.from_pretrained(f"microsoft/git-{size}").cuda()

    print(memo)
    time.sleep(3)

    model.load_state_dict(
        torch.load(f"./output_git-model-42s-3ep-microsoft/git-{size}/best_model.pth")
    )

    id2text = dict()
    with open("./diffusion/validation/metadata.jsonl") as f:
        for line in f:
            item = json.loads(line)
            id2text[item["file_name"]] = item["text"]

    filepaths = sorted(glob("./diffusion/validation/*.jpg"))[:num_example]

    output_path = f"./inf_{memo}.csv"
    output = {
        "filename": [],
        "pred": [],
        "gt": [],
        "score": [],
        "time_diff": [],
    }

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    scores = []
    for filepath in filepaths:
        filename = filepath.split("/")[-1]
        gt_text = id2text[filename]
        path = filepath
        image = Image.open(path)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.cuda()

        stime = time.time()
        # generated_ids = model.generate(
        #     pixel_values=pixel_values,
        #     max_length=77,
        #     top_p=top_p,
        #     no_repeat_ngram_size=2,
        #     do_sample=True,
        # )
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=77,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
        )
        time_diff = time.time() - stime
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        prompt_embedding = torch.tensor(st_model.encode([pred_text]))
        gt_embedding = torch.tensor(st_model.encode([gt_text]))
        score = cosim(prompt_embedding, gt_embedding)[0].item()
        scores.append(score)

        print(filename)
        print("pred:", pred_text)
        print("gt:", gt_text)
        print("score:", score)
        print("time:", time_diff)
        print()

        output["filename"].append(filename)
        output["pred"].append(pred_text)
        output["gt"].append(gt_text)
        output["score"].append(score)
        output["time_diff"].append(time_diff)

    pd.DataFrame.from_dict(output).to_csv(output_path, index=False)

    print(f"avg score: {sum(scores)/len(scores)}")
