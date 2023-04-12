import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if __name__ == "__main__":
    input_path = "resources/prompts_chatgpt_0330.txt"
    output_path = "resources/prompts_chatgpt_0330_dedup_085.txt"
    output_del_path = "./resources/del.txt"
    thres = 0.85

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    prompts = []

    n_train = 0

    if False:
        import json

        with open(
            "./diffusion/image-to-prompt-train-valid-split-v5/train/metadata.jsonl"
        ) as f:
            for line in f:
                prompts.append(json.loads(line)["text"])
        n_train = len(prompts)
        print("train len:", len(prompts))

    with open(input_path) as f:
        for idx, line in enumerate(f):
            prompts.append(line.strip())

    embeds = torch.tensor(st_model.encode(prompts)).cuda()
    print("embeds created")

    deleted_ids = set()

    chunk_size = 1000
    for i in tqdm(range(0, embeds.size(0), chunk_size)):
        start = i
        end = i + chunk_size
        chunk = embeds[i:end]

        _embeds = embeds[start:end]

        similarities = torch.mm(_embeds, embeds.T)
        indices = torch.nonzero(similarities > thres, as_tuple=True)

        pairs = []
        for x, y in zip(indices[0].tolist(), indices[1].tolist()):
            _x = x + start
            if _x == y:
                continue

            if n_train > 0 and y < n_train:
                continue

            pairs.append((_x, y))

        deleted_ids.update(set([p[1] for p in pairs]))

    print("deleted ids:", len(deleted_ids))

    with open(output_del_path, "w", encoding="utf-8") as f:
        results = []
        for idx, prompt in enumerate(prompts):
            if n_train > 0 and idx < n_train:
                continue

            if idx in deleted_ids:
                f.write(prompt + "\n")
                continue
            else:
                results.append(prompt + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, line in enumerate(results):
                f.write(line)
