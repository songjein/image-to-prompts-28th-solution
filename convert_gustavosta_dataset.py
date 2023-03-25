import json
import os
import shutil
from glob import glob

import pandas as pd

if __name__ == "__main__":
    """gustavosta의 8만건 데이터를 기존 학습 데이터와 통합하기 위해 전처리합니다.

    .. note::
        기존 파일명과 충돌하지 않도록 prefix를 붙여주고, 메타데이터 jsonl을 생성

    .. note::
        이후에 train/metadta.jsonl, validation/metadata.jsonl에 각 메타데이터를 concat하고,
        # cat gustavosta_train_images/metadata.jsonl >> train/metadata.jsonl
        # cat gustavosta_eval_images/metadata.jsonl >> validation/metadata.jsonl
    """

    convert = False

    if convert:
        train_df = pd.read_csv("./diffusion/train.csv")

        captions = []
        for idx, row in train_df.iterrows():
            prefix = "gustavosta-train-"
            src_file = row.image_path.split("/")[-1]
            dst_file = prefix + src_file
            src_path = os.path.join("./diffusion/gustavosta_train_images", src_file)
            dst_path = os.path.join("./diffusion/gustavosta_train_images", dst_file)
            shutil.move(src_path, dst_path)
            captions.append(
                {
                    "file_name": dst_file,
                    "text": row.Prompt,
                }
            )
        with open(
            "./diffusion/gustavosta_train_images/metadata.jsonl", "w", encoding="utf-8"
        ) as f:
            for caption in captions:
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")

        eval_df = pd.read_csv("./diffusion/eval.csv")

        captions = []
        for idx, row in eval_df.iterrows():
            prefix = "gustavosta-eval-"
            src_file = row.image_path.split("/")[-1]
            dst_file = prefix + src_file
            src_path = os.path.join("./diffusion/gustavosta_eval_images", src_file)
            dst_path = os.path.join("./diffusion/gustavosta_eval_images", dst_file)
            shutil.move(src_path, dst_path)
            captions.append(
                {
                    "file_name": dst_file,
                    "text": row.Prompt,
                }
            )

        with open(
            "./diffusion/gustavosta_eval_images/metadata.jsonl", "w", encoding="utf-8"
        ) as f:
            for caption in captions:
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")

    # gustavosta_train_images/* -> train/*
    # gustavosta_eval_images/* -> validation/*

    train_files = glob("./diffusion/gustavosta_train_images/*.jpg")
    eval_files = glob("./diffusion/gustavosta_eval_images/*.jpg")

    for train_file in train_files:
        shutil.copy(train_file, train_file.replace("gustavosta_train_images", "train"))

    for eval_file in eval_files:
        shutil.copy(
            eval_file, eval_file.replace("gustavosta_eval_images", "validation")
        )

    # 메타데이터 concat은 별도로 직접 수행
    # cat gustavosta_train_images/metadata.jsonl >> train/metadata.jsonl
    # cat gustavosta_eval_images/metadata.jsonl >> validation/metadata.jsonl
