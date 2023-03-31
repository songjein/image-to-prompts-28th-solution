import json

import torch
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    input_path = "./diffusion/gustavosta_eval_images/metadata.jsonl"
    output_path = "./diffusion/gustavosta_eval_images/metadata_dedup.jsonl"

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    items = []
    prompts = []
    with open(input_path) as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            items.append(item)
            prompts.append(item["text"])

    embeds = torch.tensor(st_model.encode(prompts)).cuda()
    print("embeds created")

    deleted_ids = set()

    chunk_size = 1000
    for i in range(0, embeds.size(0), chunk_size):
        start = i
        end = i + chunk_size
        chunk = embeds[i:end]

        _embeds = embeds[start:end]

        similarities = torch.mm(_embeds, embeds.T)
        indices = torch.nonzero(similarities > 0.85, as_tuple=True)

        pairs = []
        for x, y in zip(indices[0].tolist(), indices[1].tolist()):
            _x = x + start
            if _x == y:
                continue
            pairs.append((_x, y))

        deleted_ids.update(set([p[1] for p in pairs]))

    print("deleted ids:", len(deleted_ids))

    results = []
    for idx, item in enumerate(items):
        if idx in deleted_ids:
            continue
        else:
            results.append(json.dumps(item, ensure_ascii=False) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, line in enumerate(results):
            f.write(line)
