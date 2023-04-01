## dataset 만드는 방법

1. generate_sd2_images.py로 이미지 생성
2. make_captions.py로 캡션(메타데이터) 생성
3. split_dataset.py로 train/validation 스플릿(5%)

## chatgpt augmentation

- 4.0 수작업으로 3000 건 생성

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

## 03/31 일에 dedup 로직 개발

- 기존의 [diffusionDB 데이터](https://www.kaggle.com/datasets/jeinsong/sd2-images-211238)에 적용 및 dedup 스크립트 포함 시켜 놓음
  - [train/validation split](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v2)에도 적용 해 놓음(metadata_dedup.jsonl)
    - 학습 데이터 11010 건 제거 (155813 건)
      - train/metadata_dedup.jsonl (중복제거된 메타데이터)
    - 벨리데이션 데이터 33 건 제거 (8765 건)
      - validation/metadata_dedup.jsonl (중복제거된 메타데이터)
- 3/30일 까지 모았던 45126 chatgpt prompts에 dedup 적용
  - 25707 건 (./resources/prompts_chatgpt_0330_dedup.txt)
  - [chatgpt images ~ 3/30](https://www.kaggle.com/datasets/jeinsong/chatgpt-images-dedup-0330)
    - [train/validation split](https://www.kaggle.com/datasets/jeinsong/chatgpt-images-dedup-0330-split)
    - train/validation 각각을 DiffusionDB split으로 복사하고, 아래 명령어를 통해 메타데이터 통합
      - `cat chatgpt_0330_train/metadata.jsonl >> train/metadata_dedup.jsonl`
        - 23830 건
      - `cat chatgpt_0330_validation/metadata.jsonl >> validation/metadata_dedup.jsonl`
        - 1248 건
- gustavosta 데이터 통합
  - 데이터 [링크](https://www.kaggle.com/datasets/motono0223/gustavosta-stable-diffusion-prompts-sd2-v2)
    - 이미 스플릿되어 공개된 데이터셋
  - convert_gustavosta_dataset.py를 실행하여 데이터 형태 변환 및 Diffusion DB split로의 복사
  - dedup_prompts_metadata_format.py를 통해 각 gustavosta_train_images/gustavosta_eval_images split 각각이 갖고 있는 metadata dedup
    - `gustavosta_train_images/metadata_dedup.jsonl`
      - 14351 건 (59367 건 제거)
    - `gustavosta_eval_images/metadata_dedup.jsonl`
      - 4920 건 (3272 건 제거)
  - 메타데이터 통합
    - `cat gustavosta_train_images/metadata_dedup.jsonl >> train/metadata_dedup.jsonl`
    - `cat gustavosta_eval_images/metadata_dedup.jsonl >> validation/metadata_dedup.jsonl`
- 여기까지! 통합된 세 가지 데이터를 합쳐서 [v3 데이터](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v3)로 만듬
  - 학습 데이터 193994 건
  - 벨리데이션 데이터 14933 건
  - 참고
    - [v2 데이터](https://www.kaggle.com/datasets/jeinsong/image-to-prompt-train-valid-split-v2)
      - [DiffusionDB 데이터](https://www.kaggle.com/datasets/jeinsong/sd2-images-211238) 대상으로 03/31기준 preproc_split_dataset.py 실행
- 3/31일 open_prompts 10G(천만건) 데이터에 dedup적용(dedup_prompts.py)
  - 232187 건 (./resources/openprompts_dedup_075.txt)
  - filter_before_generate_image.py 적용하여 전처리를 미리 적용해 본 후 필터링
    - 192744 건 (./resources/openprompts_dedup_075_filtered.txt)
