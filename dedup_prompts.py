import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

if __name__ == "__main__":
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    prompts = []
    embeds_dict = dict()

    with open("./data/openprompts_filtered_16_256.txt") as f:
        for idx, line in tqdm(enumerate(f)):
            prompts.append(line.strip())
            embeds_dict[idx] = torch.tensor(st_model.encode([line.strip()]))

    deleted_indices = set()
    target_index = 0
    while target_index < len(prompts):
        if target_index in deleted_indices:
            target_index += 1
            continue

        target_emb = embeds_dict[target_index]

        for idx in embeds_dict.keys():
            if idx == target_index:
                continue

            if idx in deleted_indices:
                continue

            comp_emb = embeds_dict[idx]
            if cosim(target_emb, comp_emb) > 0.75:
                deleted_indices.add(idx)

                print("target", target_index, "deleted", idx)
                print("A:", prompts[target_index])
                print("B:", prompts[idx])
                print()

        target_index += 1

    with open("./data/openprompts_dedup.txt", "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            if idx in deleted_indices:
                continue
            f.write(prompt + "\n")
