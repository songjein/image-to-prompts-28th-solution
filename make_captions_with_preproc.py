import json
import os
import re
from collections import defaultdict
from typing import Optional

import spacy
from langdetect import detect
from tqdm import tqdm

# NOTE: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

STAT = defaultdict(int)


def preprocess(text: str) -> Optional[str]:
    text = text.strip()

    # [0] 사전 필터링
    concat_text = text.replace(" ", "")
    if "http" in concat_text:
        STAT["http"] += 1
        return None

    # - 남용
    if text.count("-") > 5:
        STAT["5x-"] += 1
        return None

    # by 남용
    if text.count(" by ") > 3:
        STAT["byx3"] += 1
        return None

    # by 혹은 style을 포함하는 경우 사람이 너무 많다면(3명 초과)
    # NOTE: 생성 때는 2명 이상이면 날림
    if " by " in text or " style " in text:
        doc = nlp(text)
        people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if len(people) > 3:
            STAT["too many poople"] += 1
            return None

    # 무조건 날려야 하는 애들
    ban_words = [
        "#",
        "1/",
        "jpg",
        "png",
        "jpeg",
        "1/160s",
        "XF IQ4",
        "EOS 1D",
        "EOS-1D",
        "EOS R3",
        "EOS-R3",
        "150MP",
        "150 MP",
        "apeture",
        "sony a 7",
        "in-frame",
        "rendered with",
        "resolution-W",
    ]
    for word in ban_words:
        if word.lower() in text.lower():
            STAT["stopwords"] += 1
            return None

    words = text.split()
    uniq_words = set(words)

    # [순서 중요] 반복 단어 개수 비율
    diff_uniq_words_ratio = (len(words) - len(uniq_words)) / len(words)
    if diff_uniq_words_ratio > 0.55:
        STAT["repeat words ratio > 0.55"] += 1
        return None

    # [순서 중요] 빈칸으로 쪼갰을 때 각 단어의 평균 길이
    len_sub_words = [len(word) for word in words]
    if sum(len_sub_words) / len(len_sub_words) < 2.5:
        STAT["split subword length"] += 1
        return None

    # [1] 전처리

    # 비율 패턴 지우기 16:9
    text = re.sub(r"\d+:\d+", "", text)

    # 3/4
    text = re.sub(r"\d/\d\s*(view)?", "", text)

    # 반복 특수 문자 한 번으로 교정
    text = re.sub(r"([()\[\]{}])\1+", r"\1", text)

    # 알수 없는 문자 제거
    text = text.replace("ï¿½", "")
    text = text.replace(";", ", ")
    text = text.replace("”", "")
    text = text.replace("!", ".")
    text = text.replace("_", " ")
    text = text.replace(":::", " ")
    text = text.replace("::", " ")
    text = text.replace(":", " ")
    text = text.replace("—", "-")

    # 콤마 좌우에 빈칸 없는 경우
    text = re.sub(r"(?<=[^ ]),(?=[^ ])", ", ", text)

    # 숫자 사이의 빈칸 제거
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    # 연속된 스페이스 하나로 통일 1
    text = re.sub(r"\s+", " ", text)

    # 반복된 특수문자 1개로 통일
    text = re.sub(r"([^\w\s])\1+", r"\1", text)

    # n - 5, n 9 패턴 제거
    text = re.sub(r"n\s*-?\s*\d+", "", text, flags=re.IGNORECASE)

    # 숫자 + 공백 + 캐릭터
    text = re.sub(r"\b(\d+)\s+([a-zA-Z])\b", r"\1\2", text, flags=re.IGNORECASE)

    # 숫자 + t/k
    text = re.sub(r"\d+\s*[tks]", "", text, flags=re.IGNORECASE)

    # n과 숫자 사이의 빈칸 제거 ? (50 n)
    text = re.sub(r"\sn\s*(\d)", r" n\1", text, flags=re.IGNORECASE)

    # k와 숫자 사이의 빈칸 제거 ? (20 k)
    text = re.sub(r"\b(\d+)\s*k\b", r"\1k", text, flags=re.IGNORECASE)

    # h, w과 숫자 사이의 빈칸 제거
    text = re.sub(r"(\s[hw])\s*(\d)", r"\1\2", text, flags=re.IGNORECASE)

    # n9 케이스 빈칸으로 치환
    text = re.sub(r"\b(n\d+)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(n-\d+)\b", "", text, flags=re.IGNORECASE)

    # h300, w4000 케이스 빈칸으로 치환
    text = re.sub(r"[hw]\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[hw]-\d+", "", text, flags=re.IGNORECASE)

    # 768H 640W 케이스
    text = re.sub(r"(\d)\s*([hw])", r"\1\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+[hw]\b", "", text, flags=re.IGNORECASE)

    # -w 1024, -h 2028, -n 10 패턴
    text = re.sub(r"-[whn]\s+\d+", "", text, flags=re.IGNORECASE)
    # w 1024, h 2028, n 10 패턴
    text = re.sub(r"\b[whn]\s+\d+", "", text, flags=re.IGNORECASE)
    # -height 1024 -width 1024 패턴
    text = re.sub(r"-*(height|width)\s+\d+", "", text, flags=re.IGNORECASE)
    # --steps 50
    text = re.sub(r"\-*steps?\s\d+", "", text, flags=re.IGNORECASE)
    # [순서 중요] ---i 숫자, -s 숫자, --c 숫자
    text = re.sub(r"\-+[isc]{1,3}\s\d+", "", text, flags=re.IGNORECASE)
    # [순서 중요] ---i -s -c (ex. k-pop, sci-fi 조심)
    text = re.sub(r"-+[isc]", "", text, flags=re.IGNORECASE)
    # v. a. b. e.
    text = re.sub(r"\b[a-z]\.\s", "", text, flags=re.IGNORECASE)
    # 10-i, 100i
    text = re.sub(r"\b\d+-?i\b", "", text, flags=re.IGNORECASE)

    # 1. 8f
    text = re.sub(r"\b\d*\.\s*\d*\s*f\b", "", text)
    # f/5.6, f2. 8, f / 2. 4, f 2. 8, f/1. 96
    text = re.sub(r"\bf\s*/?\s*\d\.\s*\d\b", "", text, flags=re.IGNORECASE)
    # f/20, f/8
    text = re.sub(r"\bf\s*/\s*\d\b", "", text, flags=re.IGNORECASE)
    # f. 14
    text = re.sub(r"f\.\s*\d+", "", text, flags=re.IGNORECASE)
    # f 숫자
    text = re.sub(r"f\s*\d+", "", text, flags=re.IGNORECASE)
    # 숫자 f
    text = re.sub(r"\d+\s*f", "", text, flags=re.IGNORECASE)
    # -숫자 n
    text = re.sub(r"-?\d+\s*n", "", text)
    # -숫자
    text = re.sub(r"\s-\d+", "", text)

    # |숫자 패턴
    text = re.sub(r"\|\d+", "", text)

    # 숫자 px 패턴
    text = re.sub(r"\d+\s*px\b", "", text, flags=re.IGNORECASE)
    text = text.replace(" px ", "")
    text = text.replace(" PX ", "")

    # 숫자 mpx 패턴
    text = re.sub(r"\b\d+\s*(?:mpx)\b", "", text)

    # photo 85mm, 12314mm, 50mm dslr, 100 mm lens
    text = re.sub(
        r"\b(sigma\s)?(photo\s)?\d+\s*mm(\sdslr)?(\slens)?(\sfilm)?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # uhd, fhd, ... 패턴
    text = re.sub(
        r"(?i)\b(ultra hd|full hd|ultra hd|fullhd|ultrahd|fhd|uhd|hd|hq|hdr)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # trending on + word 패턴 (trending on artstation)
    text = re.sub(
        r"\b(?:trending|featured)\s?(on|in|at)?\s?\b\w+\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # high resolution, unreal engine, award winning 제거
    text = re.sub(
        r"(low|ultra|high|hi|super|hyper)?-?\s?resolution",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"unreal\s?engine", "", text, flags=re.IGNORECASE)
    text = re.sub(r"cry\s?engine", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bue\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(award|prize)\s?-?winning(\sphotography)?(\sphoto)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # ar 숫자:숫자 패턴 제거
    text = re.sub(r"ar\s+\d+\s*:\s*\d+", "", text)

    # 숫자 iso 패턴
    text = re.sub(r"(?i)\b\d+\s*iso\b", "", text)

    # iso 숫자 패턴
    text = re.sub(r"(?i)\biso\s*\d+\b", "", text)

    # 숫자mm lens 패턴 없애기
    text = re.sub(r"(using )?\d+mm lens\b", "", text)

    # 연속된 스페이스 하나로 통일 2
    text = re.sub(r"\s+", " ", text)

    # 연속된 , , 없애기 [1회 반복]
    text = re.sub(r",\s,\s,", ", ", text)
    text = re.sub(r",\s,", ", ", text)
    text = re.sub(r",\s\.", ", ", text)
    text = re.sub(r"\.\s,", ", ", text)

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
        STAT["comma > 10"] += 1
        return None

    # 추가 스탑워드 패턴 제거
    # todo: station art trending on artstation by art station at his art station
    stop_patterns = [
        "artgerm and greg rutkowski and alphonse mucha",
        "hyperdetailed artstation cgsociety",
        "digital painting artstation",
        "painting artstation concept",
        "artstation concept art",
        "greg rutkowski and",
        "and greg rutkowski",
        "drawn by artgerm",
        "contest winner",
        "contest-winning",
        "contest-winner",
        "award-winning",
        "award-winner",
        "octane render",
        "rtx",
        "/r/",
        "8k",
        "4k",
    ]
    for pattern in stop_patterns:
        regex_pattern = re.compile(
            r"\b" + re.escape(pattern) + r"\b", flags=re.IGNORECASE
        )
        text = regex_pattern.sub("", text).strip()

    artists = [
        "greg rutkowski",
        "wlop",
        "ilya kuvshinov",
        "alphonse mucha",
        "rossdraws",
        "craig mullins",
        "james jean",
        "rhads",
    ]
    for artist in artists:
        regex_pattern = re.compile(
            r"\b" + re.escape("by " + artist) + r"\b", flags=re.IGNORECASE
        )
        text = regex_pattern.sub("", text).strip()

        regex_pattern = re.compile(
            r"\b" + re.escape(artist) + r"\b", flags=re.IGNORECASE
        )
        text = regex_pattern.sub("", text).strip()

    trendings = [
        "national geographic",
        "artstationHD",
        "artstation",
        "art station",
        "cgsociety",
        "deviantart",
        "pinterest",
        "shuttershock",
        "behance",
        "instagram",
        "zbrush central",
        "cg society",
        "polycount",
        "shutterstock",
        "artgerm",
        "redshift",
        "octane",
    ]
    for trending in trendings:
        regex_pattern = re.compile(
            r"\b" + re.escape("by " + trending) + r"\b", flags=re.IGNORECASE
        )
        text = regex_pattern.sub("", text).strip()

        regex_pattern = re.compile(
            r"\b" + re.escape(trending) + r"\b", flags=re.IGNORECASE
        )
        text = regex_pattern.sub("", text).strip()

    # [digital art] 케이스
    text = re.sub(r"\[\s*digital art\s*\]", "", text)

    # 연속된 , , 없애기 [2회 반복]
    text = re.sub(r",\s,\s,", ", ", text)
    text = re.sub(r",\s,", ", ", text)
    text = re.sub(r",\s\.", ", ", text)
    text = re.sub(r"\.\s,", ", ", text)

    # 연속된 . . 없애기 [2회 반복]
    text = re.sub(r"\.\s\.\s\.", ". ", text)
    text = re.sub(r"\.\s\.", ". ", text)

    # 연속된 ; ; 없애기 [2회 반복]
    text = re.sub(r";\s;\s;", "; ", text)
    text = re.sub(r";\s;", "; ", text)

    # 길이 기준 필터링 (하위 5%?, 상위 5%)
    if len(text) < 16 or len(text) > 256:
        STAT["16 < len < 256"] += 1
        return None

    #: 현재 상태에서 단어 분리
    words = text.split()
    uniq_words = set(words)

    # 유니크 단어 개수 기준 필터링 (반복 케이스 잡기 위해 set 적용)
    num_uniq_words = len(uniq_words)
    if num_uniq_words <= 3:
        STAT["uniq words"] += 1
        return None

    # [순서 중요] uni-gram 반복 교정
    text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE).strip()

    # [순서 중요] bi-gram 반복 교정
    text = re.sub(r"\b(\w+\s+\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE).strip()

    # 자기 생각 적어놓은 류
    patterns = r"(?i)\b(i'll|i'm|i am|if you|if my|isn't|i need|help me|i like|i had|you know|i have)\b"
    if re.search(patterns, text, re.IGNORECASE):
        STAT["thought"] += 1
        return None

    # 연속된 스페이스 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    # 구두점 좌우에 스페이스가 있다면 오른쪽만 남기기
    text = re.sub(r"\s+([.,;!?])", r"\1", text)

    # (년도) 패턴
    text = re.sub(r"\(\s*\d+\s*\)", "", text)
    # (년도-년도) 패턴
    text = re.sub(r"\(\d{4}-\d{4}\)", "", text)
    # (년도 패턴
    text = re.sub(r"\(\s*\d+\s*", "", text)
    # (년도-년도 패턴
    text = re.sub(r"\(\d{4}-\d{4}", "", text)

    # 연속된 , , 없애기 [1회 반복]
    text = re.sub(r",\s,\s,", ", ", text)
    text = re.sub(r",\s,", ", ", text)
    text = re.sub(r",\s\.", ", ", text)
    text = re.sub(r"\.\s,", ", ", text)

    # 연속된 . . 없애기 [1회 반복]
    text = re.sub(r"\.\s\.\s\.", ". ", text)
    text = re.sub(r"\.\s\.", ". ", text)

    # 연속된 ; ; 없애기 [1회 반복]
    text = re.sub(r";\s;\s;", "; ", text)
    text = re.sub(r";\s;", "; ", text)

    # (by 단어 단어) 패턴 지우기
    text = re.sub(r"\((by [^)]+)\)", "", text, flags=re.IGNORECASE)

    # 연속된 ,. 조합
    text = re.sub(r"[.,]{2,}", " ", text)

    # 문장 시작 구두점 제거
    text = re.sub(r"^\W+", "", text).strip()

    # 기타
    text = text.replace(" _ ", " ")

    # ||| 반복 패턴 빈칸 치환
    text = re.sub(r"\|+", " ", text)

    # ())(()), ))(())
    text = re.sub(r"[()]{2,}", "", text)

    # 5자리 이상 숫자 제거
    text = re.sub(r"\b\d{5,}\b", "", text)

    # 괄호 안 빈칸 제거
    text = re.sub(r"\(\s*(\S+)\s*\)", "", text)

    # 연속된 스페이스 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    # 마지막에 공백+and로 끝나는 경우
    if text[-4:] == " and":
        text = text[:-3]

    return text


if __name__ == "__main__":
    prefix = "dbd4_"
    input_path = "./resources/dbd4_prompts.txt"
    output_path = "./diffusion/dbd4/metadata.jsonl"
    image_dir_path = "./diffusion/dbd4"

    #: TODO 임시
    input_path = "./resources/v6_dbd3_dbd4_080_pprc.txt"
    output_path = "./resources/v6_dbd3_dbd4_080_pprc_.txt"
    image_dir_path = None

    if False:
        skip = 0
        items = []
        with open(
            "./diffusion/image-to-prompt-train-valid-split-v7/train/metadata.jsonl"
        ) as f:
            for line in tqdm(f):
                item = json.loads(line)
                text = preprocess(item["orig_text"])
                if text is None:
                    skip += 1
                    continue
                item["text"] = text
                items.append(item)

        with open(
            "./diffusion/image-to-prompt-train-valid-split-v7/train/metadata_.jsonl"
        ) as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print("skip:", skip)
        print(STAT)
        exit()

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

                # try:
                #     if detect(prompt) != "en":
                #         skip_cnt += 1
                #         continue
                # except:
                #     print("exception at", prompt)

                text = preprocess(prompt)
                if text is None:
                    skip_cnt += 1
                    continue

                # captions.append(prompt)
                captions.append(text)  # 전처리된 것으로 저장 (gpt)

        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in captions:
                f.write(prompt + "\n")
    print(STAT)
    print(f"skip_count: {skip_cnt}")
