import json
import os
import re
from typing import Optional

from langdetect import detect
from tqdm import tqdm


def preprocess(text: str) -> Optional[str]:
    text = text.strip()
    orig_text = text

    concat_text = text.replace(" ", "")
    if "http" in concat_text:
        return None

    # NOTE: 기존에 있던 스탑워드 필터링 제거함 > 추후에 필요하다면 replace로 대체

    # [1] 전처리
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

    # high resolution, unreal engine, award winning 제거
    text = re.sub(r"high-? resolution", "", text, flags=re.IGNORECASE)
    text = re.sub(r"unreal\s?engine", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"award\s?-?winning(\sphotography)?(\sphoto)?", "", text, flags=re.IGNORECASE
    )

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

    # [2] 필터링

    # 길이 기준 필터링 (하위 5%?, 상위 5%)
    if len(text) < 16 or len(text) > 256:
        return None

    words = text.split()
    uniq_words = set(words)

    # 유니크 단어 개수 기준 필터링 (반복 케이스 잡기 위해 set 적용)
    num_uniq_words = len(uniq_words)
    if num_uniq_words <= 3:
        return None

    # 반복 단어 개수 비율
    diff_uniq_words_ratio = (len(words) - len(uniq_words)) / len(words)
    if diff_uniq_words_ratio > 0.55:
        return None

    # 빈칸으로 쪼갰을 때 각 단어의 평균 길이
    len_sub_words = [len(word) for word in words]
    if sum(len_sub_words) / len(len_sub_words) < 2.5:
        return None

    # 자기 생각 적어놓은 류
    patterns = r"(?i)\b(i'll|i'm|i am|if you|isn't|i need|help me|i like|i had|you know|i have)\b"
    if re.search(patterns, text, re.IGNORECASE):
        return None

    return text


if __name__ == "__main__":
    prefix = "openprompts_"
    input_path = "./resources/openprompts_dedup_075_filtered_cross_dedup.txt"
    output_path = "./diffusion/openprompts_images/metadata.jsonl"  # 기존엔 168166
    image_dir_path = "./diffusion/openprompts_images"

    skip_cnt = 0

    captions = []
    with open(input_path) as f:
        for idx, line in enumerate(tqdm(f)):
            file_name = f"{prefix}{idx:08d}.jpg"
            path = os.path.join(image_dir_path, file_name)
            if not os.path.exists(path):
                continue
            prompt = line.strip()

            try:
                if detect(prompt) != "en":
                    skip_cnt += 1
                    continue
            except:
                print("exception at", prompt)

            text = preprocess(prompt)
            if text is None:
                skip_cnt += 1
                continue

            captions.append(
                {
                    "file_name": file_name,
                    "text": text,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")

    print(f"skip_count: {skip_cnt}")
