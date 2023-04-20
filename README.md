## dataset 만드는 방법

1. generate_sd2_images.py로 이미지 생성
2. make_captions_with_preproc.py로 캡션(메타데이터) 생성 및 전처리
3. (optional) split_dataset.py로 train/validation 스플릿(5%)

## chatgpt augmentation

- 4.0 수작업으로 3000 개 생성

  ```
  Examples
  - hyper realistic photo of very friendly and dystopian crater
  - ramen carved out of fractal rose ebony, in the style of hudson river school
  - ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus
  - a thundering retro robot crane inks on parchment with a droopy french bulldog
  - portrait painting of a shimmering greek hero, next to a loud frill-necked lizard
  - an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms
  - Kaggle employee Phil at a donut shop ordering all the best donuts, with a speech bubble that proclaims "Donuts. It's what's for dinner!"

  These are the examples of prompts for generating images.

  Make other prompts in the similar style as many as possible, with variety of random scenes, and range from fairly simple to complex with multiple objects and modifiers. Different words from examples are preferred.

  Not preferred topics: [mushroom, jellyfish, giraffe, forest, underwater, street, garden, neon, cityscape, castle, dragon, landscape, sea, coral, tea, trees, cars, fish, mountain, mischievous, mermaid, animals, unicorn, schools, space, rainbow, ancient, skyscrapers, flowers, cabin, beach, fairy, market]
  ```

- 4.0에서 랜덤 10개 가져오고, 학습 프롬프트에서 10개 가져와서 생성
  - prompt_inf.py
  - 결과: resources/\*.txt

## v3 데이터 구성 (03/31)

- 기존의 [diffusionDB 데이터](https://www.kaggle.com/datasets/jeinsong/sd2-images-211238)에 적용 및 dedup 스크립트 포함 시켜 놓음
  - [train/validation split](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v2)에도 적용 해 놓음(metadata_dedup.jsonl)
    - 학습 데이터 11010 개 제거 (155813 개)
      - train/metadata_dedup.jsonl (중복제거된 메타데이터)
    - 벨리데이션 데이터 33 개 제거 (8765 개)
      - validation/metadata_dedup.jsonl (중복제거된 메타데이터)
- 3/30일 까지 모았던 45126 chatgpt prompts에 dedup 적용
  - 25707 개 (./resources/prompts_chatgpt_0330_dedup.txt)
  - [chatgpt images ~ 3/30](https://www.kaggle.com/datasets/jeinsong/chatgpt-images-dedup-0330)
    - [train/validation split](https://www.kaggle.com/datasets/jeinsong/chatgpt-images-dedup-0330-split)
    - train/validation 각각을 DiffusionDB split으로 복사하고, 아래 명령어를 통해 메타데이터 통합
      - `cat chatgpt_0330_train/metadata.jsonl >> train/metadata_dedup.jsonl`
        - 23830 개
      - `cat chatgpt_0330_validation/metadata.jsonl >> validation/metadata_dedup.jsonl`
        - 1248 개
- gustavosta 데이터 통합
  - 데이터 [링크](https://www.kaggle.com/datasets/motono0223/gustavosta-stable-diffusion-prompts-sd2-v2)
    - 이미 스플릿되어 공개된 데이터셋
  - convert_gustavosta_dataset.py를 실행하여 데이터 형태 변환 및 Diffusion DB split로의 복사
  - dedup_prompts_metadata_format.py를 통해 각 gustavosta_train_images/gustavosta_eval_images split 각각이 갖고 있는 metadata dedup
    - `gustavosta_train_images/metadata_dedup.jsonl`
      - 14351 개 (59367 개 제거)
    - `gustavosta_eval_images/metadata_dedup.jsonl`
      - 4920 개 (3272 개 제거)
  - 메타데이터 통합
    - `cat gustavosta_train_images/metadata_dedup.jsonl >> train/metadata_dedup.jsonl`
    - `cat gustavosta_eval_images/metadata_dedup.jsonl >> validation/metadata_dedup.jsonl`
- 여기까지! 통합된 세 가지 데이터를 합쳐서 [v3 데이터](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v3)로 만듬
  - 학습 데이터 193994 개
  - 벨리데이션 데이터 14933 개
  - 참고
    - [v2 데이터](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v2)
      - [DiffusionDB 데이터](https://www.kaggle.com/datasets/jeinsong/sd2-images-211238) 대상으로 03/31기준 preproc_split_dataset.py 실행
- 3/31일 open_prompts 10G(천만건) 데이터에 dedup적용(dedup_prompts.py)
  - 232187 개 (./resources/openprompts_dedup_075.txt)
  - filter_before_generate_image.py 적용하여 전처리를 미리 적용해 본 후 필터링
    - 192744 개 (./resources/openprompts_dedup_075_filtered.txt)

## v4 데이터 구성 (04/03)

- 앞서 v2에 dedup해 놓은 [train/validation split](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v2)을 활용
  - 학습 데이터 11010 개 제거 (155813 개)
    - train/metadata_dedup.jsonl (중복제거된 메타데이터)
  - 벨리데이션 데이터 33 개 제거 (8765 개)
    - validation/metadata_dedup.jsonl (중복제거된 메타데이터)
- 중간에 확인해보니 open_prompts 데이터가 train과 많이 겹치고 있었음...
  - 현재까지 처리된 12만 개는 제외하고, 12만개 이후에선 중복 제거(22643개)하여 이어서 돌림
  - resources/openprompts_dedup_075_filtered_cross_dedup.txt
    - 앞으로도 데이터 합친 이후엔 반드시 dedup 적용(self)
- v4에선 gustavosta 데이터를 validation에서만 활용할 예정
  - v2 validation + gustavosta data(train + valid)
- v4에선 아래 데이터셋을 추가
  - openprompts 12만 개에서, 중복 제거하고 추가
    - make_captions로 메타데이터 부터 만들고 학습데이터와 중복인 부분 제거
    - [필터링 된 데이터](https://www.kaggle.com/datasets/jeinsong/openprompts-images-120k-dedup)
      - metadata_dedup.jsonl (81955 개)
- v4 데이터 생성 결과
  - 4/3일
    - train 237768 개 (v2 + openprompts 120k dedup)
      - train/metadata_concat.jsonl
    - validation 23116 개 (v2 + gustavosta train/eval)
      - 이후에 dedup_prompts_metadata_format.py 실행 통해 split dedup (3901 개 제거)
      - validation/metadata_concat_split_dedup.jsonl
  - 4/4일 170k dedup 합친 버전 (jeinsong/image-to-prompt-train-valid-split-v4-a)
    - 총 286698 개 (학습셋만 업데이트)
      - make_captions > dedup_prompts_metadata_format(cross split) > image-to-prompt-train-valid-split-v4/prepare 실행
    - _추가 데이터에 대해 preproc이 적용 안된 문제 발견_ > image-to-prompt-train-valid-split-v4/filter_by_preprocy.py 반복 실행 > v4-b로 재배포
      - train 286670 개 (train/metadata_concat.jsonl)
      - validation 13324 개 (validation/metadata_concat_split_dedup.jsonl)

## v5 데이터 데이터 구성 (04/04)

- ddb 데이터 다시 정제(전처리에서 스탑워드 필터링 일괄 제거)
  - [168916건 데이터 확보](https://www.kaggle.com/datasets/jeinsong/ddb-sd2-168916)
    - jeinsong/sd2-images-211238를 재가공
- openprompts 데이터 다시 정제
  - 168212 개
- 중복 제거
  - dedup_prompts_metadata_format.py 한 번 해주기 > train/metadata_concat_dedup.jsonl (82982 제거되어 254146개) > train/metadata.jsonl로 이름 변경
  - validation은 v4-b 가져와서 필터 0개 될 때 까지, filter_by_preproc.py 실행 후 8802 개 남음 (gustavosta 아닌 것 제거 하고, 전처리 필터링 적용) > validation/metadata.jsonl로 이름 변경
- v6 or v5+ 에선 아래 데이터셋을 추가 고려
  - https://www.kaggle.com/datasets/xiaozhouwang/sd2hardcode
  - https://www.kaggle.com/datasets/xiaozhouwang/sd2gpt2
  - (optional) https://www.kaggle.com/datasets/jeinsong/chatgpt-images-dedup-0330-split

## v6 데이터셋 구성 (04/11)

- ddb 데이터 원본 다시 가져와서 놓쳤던 데이터 대거 보충
- validation split도 다시 만듦
- train: 363269 개
- valid: 11262 개

## v7 데이터셋 구성 (04/17)

- ddb 데이터 추가 보충 (111k 개에서 9844개 삭제 됨) > 101672 개 > v6 train(0.85)/validation(0.75) dedup > 96635 개
- chatgpt 데이터 보충 > v6 train(0.85)/validation(0.75) dedup > 23385 개
- dedup_prompts_metdata_format.py > train / validation 각각에 self-dedup 적용 (0.85)
  - train -1022 > 477267 개
  - valid -39 > 16223 개
- dedup_prompts_metdata_format.py > train / validation cross-dedup 적용 (0.75)
  - valid -2988 > 13235 개
- 이후에 metadata.jsonl 기준으로 파일 삭제 (prepare.py)
- (04/18) 성능 크게 하락 > chatgpt 데이터 제거 및 벨리데이션셋 v6로 회귀
- (04/20) 추가 데이터 재생성 dbd3_prompts.txt, dbd4_prompts.txt
  - dbd3 - 81944 개 > preproc 7242 개 제거 > 74702 개
  - dbd4 - 70939 개 > preproc 7321 개 제거 > 63618 개
  - 전처리 로직 업데이트! (사람 이름 처리 및 기타 노이즈 정제 로직 추가)
    - 다음 부턴 생성 전에 전처리 부터 태우자
  - self-dedup - 8689 개 > 492900 개 > cross-dedup - 2479 개 (벨리드 기준으로 학습셋을 없앰) > 490421 개
  - 전처리 로직 업뎃 후 > 486415 개 됨 (4006개 제거)
