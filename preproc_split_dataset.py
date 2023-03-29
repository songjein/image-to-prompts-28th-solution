import json
import os
import random
import re
import shutil
from typing import Optional

from langdetect import detect
from tqdm import tqdm


def preprocess(text: str) -> Optional[str]:
    text = text.strip()
    orig_text = text

    # 빈칸으로 쪼갰을 때 각 단어의 평균 길이가 3 이하라면 필터링
    len_sub_words = [len(word) for word in text.split()]
    if sum(len_sub_words) / len(len_sub_words) < 3:
        return None

    concat_text = text.replace(" ", "")
    if "http" in concat_text:
        return None

    if "fps" in concat_text:
        return None

    stopwords = [
        "unreal engine",
        "award winning",
        "award-winning",
        "sec shutter",
        "second shutter",
        "shutter speed",
        "1/60",
        "1/160",
        "1/1000",
        "1/2000",
        "cannon",
        "canon",
        "kodak",
        "portra",
        " eos",
        "leica summilux",
        "focal length",
        "degrees angle",
        "exposure time",
        "flash on",
        "DOF",
        "dof:",
        "blur:",
        "fzr",
    ]
    for stopword in stopwords:
        if stopword in text.lower():
            return None

    # 숫자 사이의 빈칸 제거
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    # 연속된 스페이스 하나로 통일
    text = re.sub(r"\s+", " ", text)

    # 반복된 특수문자 1개로 통일
    text = re.sub(r"([^\w\s])\1+", r"\1", text)

    # n - 5, n 9 패턴 제거
    text = re.sub(r"\bn\s*-?\s*\d+\b", "", text)

    # n과 숫자 사이의 빈칸 제거
    text = re.sub(r"\sn\s*(\d)", r" n\1", text)

    # k와 숫자 사이의 빈칸 제거
    text = re.sub(r"\b(\d+)\s*k\b", r"\1k", text)

    # h, w과 숫자 사이의 빈칸 제거
    text = re.sub(r"(\s[hw])\s*(\d)", r"\1\2", text)

    # n9 케이스 빈칸으로 치환
    text = re.sub(r"\b(n\d+)\b", "", text)
    text = re.sub(r"(?i)\b(n-\d+)\b", "", text)

    # h300, w4000 케이스 빈칸으로 치환
    text = re.sub(r"[hw]\d+", "", text)
    text = re.sub(r"(?i)[hw]-\d+", "", text)

    # f/5.6, f2. 8, f / 2. 4, f 2. 8, f/1. 96
    text = re.sub(r"(?i)\bf\s*/?\s*\d\.\s*\d\b", "", text)
    # f/20, f/8
    text = re.sub(r"(?i)\bf\s*/\s*\d\b", "", text)

    # photo 85mm, 12314mm, 50mm dslr, 100 mm lens
    text = re.sub(
        r"\b(sigma\s)?(photo\s)?\d+\s*mm(\sdslr)?(\slens)?(\sfilm)?\b", "", text
    )

    # uhd, fhd, ... 패턴
    text = re.sub(r"(?i)\b(ultra hd|fullhd|ultrahd|fhd|uhd|hd|hq|hdr)\b", "", text)

    # 4k, 8k, shuttershock 없애기
    text = (
        text.replace("8k", "")
        .replace("8K", "")
        .replace("4k", "")
        .replace("4K", "")
        .replace("shuttershock", "")
    )

    # high resolution 제거
    text = re.sub(r"high-? resolution", "", text, flags=re.IGNORECASE)

    # ar 숫자:숫자 패턴 제거
    text = re.sub(r"ar\s+\d+\s*:\s*\d+", "", text)

    # 구두점 직전의 공백 없애기
    text = re.sub(r"\s*([^\w\s,\.!\?])(?:\s+|$)", r"\1", text)

    # 숫자 iso 패턴
    text = re.sub(r"(?i)\b\d+\s*iso\b", "", text)

    # iso 숫자 패턴
    text = re.sub(r"(?i)\biso\s*\d+\b", "", text)

    # 숫자mm lens 패턴 없애기
    text = re.sub(r"\b(using )?\d+mm lens\b", "", text)

    # 연속된 스페이스 하나로 통일
    text = re.sub(r"\s+", " ", text)

    # 연속된 , , 없애기
    text = re.sub(r",\s,\s,", ", ", text)
    text = re.sub(r",\s,", ", ", text)

    # 연속된 . . 없애기
    text = re.sub(r"\.\s\.\s\.", ". ", text)
    text = re.sub(r"\.\s\.", ". ", text)

    # 연속된 ; ; 없애기
    text = re.sub(r";\s;\s;", "; ", text)
    text = re.sub(r";\s;", "; ", text)

    # 연속된 스페이스 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    # 마지막 구두점 제거
    text = re.sub(r"[^\w\s]+$", "", text)

    # 문장 시작 구두점 제거
    text = re.sub(r"^\W+", "", text)

    try:
        if detect(text) != "en":
            return None
    except:
        print("exception at", orig_text)
        return None

    return text

    # NOTE: salchenwursage. 35 mm, , cinelux asa 100
    # NOTE: 그렇게 많이 안날아가는데, 사진 관련된 애들 (ex. mm포함) 다 날려버려도 될 수도?


if __name__ == "__main__":
    root = "./diffusion/images/"
    move_file = not False

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
    _train_captions = []
    for caption in tqdm(train_captions):
        text = preprocess(caption["text"])
        if text is None:
            continue
        caption["text"] = text
        visited.add(caption["text"].strip())
        tlens.append(len(caption["text"].strip()))
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join("./diffusion/train", caption["file_name"])
        if move_file:
            shutil.copy(_from, _to)
        _train_captions.append(caption)

    train_captions = _train_captions

    dup_cnt = 0
    vlens = []
    _valid_captions = []
    for caption in tqdm(valid_captions):
        text = preprocess(caption["text"])
        if text is None:
            continue
        caption["text"] = text
        if caption["text"] in visited:
            print(caption)
            dup_cnt += 1
            continue
        vlens.append(len(caption["text"].strip()))
        _from = os.path.join(root, caption["file_name"])
        _to = os.path.join("./diffusion/validation", caption["file_name"])
        if move_file:
            shutil.copy(_from, _to)
        _valid_captions.append(caption)

    valid_captions = _valid_captions

    print(dup_cnt)
    print("tlens", sum(tlens) / len(tlens))
    print("vlens", sum(vlens) / len(vlens))
    print("num train samples", len(tlens))
    print("num valid samples", len(vlens))

    with open("./diffusion/train/metadata.jsonl", "w") as f:
        for caption in train_captions:
            caption["file_name"] = caption["file_name"]
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")

    with open("./diffusion/validation/metadata.jsonl", "w") as f:
        for caption in valid_captions:
            caption["file_name"] = caption["file_name"]
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")
