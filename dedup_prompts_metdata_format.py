import json

import torch
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    cross_split_dedup = True
    thres = 0.85
    chunk_size = 1000

    if not cross_split_dedup:
        input_path = "./diffusion/train/metadata_dedup_wo_chatgpt.jsonl"
        output_path = "test.jsonl"

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

        deleted_indices = set()

        for i in range(0, embeds.size(0), chunk_size):
            start = i
            end = i + chunk_size
            _embeds = embeds[start:end]

            similarities = torch.mm(_embeds, embeds.T)
            indices = torch.nonzero(similarities > thres, as_tuple=True)

            pairs = []
            for x, y in zip(indices[0].tolist(), indices[1].tolist()):
                _x = x + start
                if _x == y:
                    continue
                pairs.append((_x, y))

            deleted_indices.update(set([p[1] for p in pairs]))

        print("deleted ids:", len(deleted_indices))

        results = []
        for idx, item in enumerate(items):
            if idx in deleted_indices:
                continue
            else:
                results.append(json.dumps(item, ensure_ascii=False) + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, line in enumerate(results):
                f.write(line)

    else:
        # train 기준 valid 의 dup을 제거
        train_input_path = "./diffusion/train/metadata_dedup_wo_chatgpt.jsonl"
        valid_input_path = "./diffusion/validation/metadata_dedup_wo_chatgpt.jsonl"
        file_name = valid_input_path.split("/")[-1]
        output_path = valid_input_path.replace(
            file_name, file_name.split(".")[0] + "_split_dedup.jsonl"
        )
        st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()
        cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

        # 한 곳으로 합쳐서 한 번에 계산 후, 인덱스로 구분
        items = []
        prompts = []
        with open(train_input_path) as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                items.append(item)
                prompts.append(item["text"])

        #: 학습 데이터 마지막 인덱스
        end_of_train_index = len(prompts) - 1

        with open(valid_input_path) as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                items.append(item)
                prompts.append(item["text"])

        #: 학습 + 벨리데이션 데이터셋에 대한 임베딩
        embeds = torch.tensor(st_model.encode(prompts)).cuda()
        print("embeds created")

        #: 삭제할 인덱스
        deleted_indices = set()
        for i in range(0, embeds.size(0), chunk_size):
            start = i
            end = i + chunk_size
            _embeds = embeds[start:end]

            similarities = torch.mm(_embeds, embeds.T)
            indices = torch.nonzero(similarities > thres, as_tuple=True)

            pairs = []
            for x, y in zip(indices[0].tolist(), indices[1].tolist()):
                #: 청크 인덱스가 반영된 시작점 (y는 관계 없음)
                _x = x + start

                # 기준점은 학습 인덱스만, 비교 대상은 벨리데이션 인덱스만
                if _x > end_of_train_index or y <= end_of_train_index:
                    continue

                if _x == y:
                    continue
                pairs.append((_x, y))

            deleted_indices.update(set([p[1] for p in pairs]))

        print("deleted ids:", len(deleted_indices))

        results = []
        for idx, item in enumerate(items):
            if idx in deleted_indices:
                continue
            elif idx <= end_of_train_index:
                continue
            else:
                results.append(json.dumps(item, ensure_ascii=False) + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, line in enumerate(results):
                f.write(line)
