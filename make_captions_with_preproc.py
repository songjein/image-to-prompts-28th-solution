import json
import os
import re
from typing import Optional

from langdetect import detect
from tqdm import tqdm


def preprocess(text: str) -> Optional[str]:
    text = text.strip()

    concat_text = text.replace(" ", "")
    if "http" in concat_text:
        return None

    # [0] 사전 필터링

    words = text.split()
    uniq_words = set(words)

    # [순서 중요] 반복 단어 개수 비율
    diff_uniq_words_ratio = (len(words) - len(uniq_words)) / len(words)
    if diff_uniq_words_ratio > 0.55:
        return None

    # [순서 중요] 빈칸으로 쪼갰을 때 각 단어의 평균 길이
    len_sub_words = [len(word) for word in words]
    if sum(len_sub_words) / len(len_sub_words) < 2.5:
        return None

    # [1] 전처리

    # 알수 없는 문자 제거
    text = text.replace("ï¿½", "")
    text = text.replace(";", ", ")

    # 콤마 좌우에 빈칸 없는 경우
    text = re.sub(r"(?<=[^ ]),(?=[^ ])", ", ", text)

    # 숫자 사이의 빈칸 제거
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    # 연속된 스페이스 하나로 통일 1
    text = re.sub(r"\s+", " ", text)

    # 반복된 특수문자 1개로 통일
    text = re.sub(r"([^\w\s])\1+", r"\1", text)

    # n - 5, n 9 패턴 제거
    text = re.sub(r"\bn\s*-?\s*\d+\b", "", text)

    # n과 숫자 사이의 빈칸 제거 ?
    text = re.sub(r"\sn\s*(\d)", r" n\1", text)

    # k와 숫자 사이의 빈칸 제거 ?
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

    # trending on + word 패턴
    text = re.sub(r"\b(?:trending|featured)\b on \b\w+\b", "", text)

    # high resolution, unreal engine, award winning 제거
    text = re.sub(r"high-? resolution", "", text, flags=re.IGNORECASE)
    text = re.sub(r"unreal\s?engine", "", text, flags=re.IGNORECASE)
    text = re.sub(r"cry\s?engine", "", text, flags=re.IGNORECASE)
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

    # 연속된 스페이스 하나로 통일 2
    text = re.sub(r"\s+", " ", text)

    # 연속된 , , 없애기 [1회 반복]
    text = re.sub(r",\s,\s,", ", ", text)
    text = re.sub(r",\s,", ", ", text)

    # 연속된 . . 없애기 [1회 반복]
    text = re.sub(r"\.\s\.\s\.", ". ", text)
    text = re.sub(r"\.\s\.", ". ", text)

    # 연속된 ; ; 없애기 [1회 반복]
    text = re.sub(r";\s;\s;", "; ", text)
    text = re.sub(r";\s;", "; ", text)

    # seed 숫자 제거
    text = re.sub(r"\bseed\b \d+", "", text).strip()

    # [2] 필터링 + 치환

    # [순서 중요] 콤마가 10번 이상 등장
    if text.count(",") >= 10:
        return None

    # [순서 중요] by 뒤에 콤마가 3번 이상 등장
    by_pattern = " by "
    by_fp_pattern = "ed by "  # consumed, surrounded, scorched, squished, ignited, ...
    and_pattern = " and "
    if by_pattern in text and by_fp_pattern not in text:
        index = text.index(by_pattern)
        if by_pattern in text and text[index + 2 :].count(",") >= 3:
            text = text[:index].strip()
            if text[-1] == ",":
                text = text[:-1]

    # [순서 중요] by 뒤에 (and 개수 + by 개수) 두개 이상 등장
    if by_pattern in text and by_fp_pattern not in text:
        index = text.index(by_pattern)
        if (
            by_pattern in text
            and (
                text[index + 2 :].count(and_pattern)
                + text[index + 2 :].count(by_pattern)
            )
            >= 2
        ):
            text = text[:index].strip()
            if text[-1] == ",":
                text = text[:-1]

    # [순서 중요] by 부터 가장 가까운 콤마까지 빈칸 치환
    if by_fp_pattern not in text:
        text = re.sub(r"\bby\b.*?,", "", text)

    # 추가 스탑워드 패턴 제거
    # todo: station art trending on artstation by art station at his art station
    stop_patterns = [
        "by art station",
        "highly detailed digital",
        "greg rutkowski and",
        "artstation concept art",
        "and greg rutkowski",
        "digital painting artstation",
        "painting artstation concept",
        "hyperdetailed artstation cgsociety",
        "by artstation",
        "octane render",
        "highly detailed",
        "contest winner",
        "artstation",
        "art station",
        "cgsociety",
        "deviantart",
        "pinterest",
        "shuttershock",
        "/r/",
        "8k",
        "4k",
    ]
    for pattern in stop_patterns:
        regex_pattern = re.compile(
            r"\b" + re.escape(pattern) + r"\b", flags=re.IGNORECASE
        )
        text = regex_pattern.sub("", text).strip()

    # 연속된 , , 없애기 [2회 반복]
    text = re.sub(r",\s,\s,", ", ", text)
    text = re.sub(r",\s,", ", ", text)

    # 연속된 . . 없애기 [2회 반복]
    text = re.sub(r"\.\s\.\s\.", ". ", text)
    text = re.sub(r"\.\s\.", ". ", text)

    # 연속된 ; ; 없애기 [2회 반복]
    text = re.sub(r";\s;\s;", "; ", text)
    text = re.sub(r";\s;", "; ", text)

    # 길이 기준 필터링 (하위 5%?, 상위 5%)
    if len(text) < 16 or len(text) > 256:
        return None

    #: 현재 상태에서 단어 분리
    words = text.split()
    uniq_words = set(words)

    # 유니크 단어 개수 기준 필터링 (반복 케이스 잡기 위해 set 적용)
    num_uniq_words = len(uniq_words)
    if num_uniq_words <= 3:
        return None

    # [순서 중요] uni-gram 반복 교정
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE).strip()

    # [순서 중요] bi-gram 반복 교정
    text = re.sub(r"\b(\w+\s+\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE).strip()

    # 자기 생각 적어놓은 류
    patterns = r"(?i)\b(i'll|i'm|i am|if you|if my|isn't|i need|help me|i like|i had|you know|i have)\b"
    if re.search(patterns, text, re.IGNORECASE):
        return None

    # 연속된 스페이스 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    # 마지막 구두점 제거
    text = re.sub(r"[^\w\s]+$", "", text).strip()

    # 문장 시작 구두점 제거
    text = re.sub(r"^\W+", "", text).strip()

    return text


if __name__ == "__main__":
    prefix = "chatgpt_0330_"
    input_path = "./resources/prompts_chatgpt_0330_dedup.txt"
    output_path = "./diffusion/chatgpt-images/metadata.jsonl"
    image_dir_path = "./diffusion/chatgpt-images"

    #: TODO 임시
    # input_path = "./resources/900k_dedup_08_split_dedup_wt_wv.txt"
    # output_path = "./resources/tmp.txt"
    # image_dir_path = None

    skip_cnt = 0

    if image_dir_path is not None:
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
                        "orig_text": prompt,
                    }
                )

        with open(output_path, "w", encoding="utf-8") as f:
            for caption in captions:
                f.write(json.dumps(caption, ensure_ascii=False) + "\n")

    else:
        captions = []

        with open(input_path) as f:
            for idx, line in enumerate(tqdm(f)):
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

                captions.append(prompt)

        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in captions:
                f.write(prompt + "\n")

    print(f"skip_count: {skip_cnt}")
