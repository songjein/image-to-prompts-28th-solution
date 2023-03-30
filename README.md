## dataset 만드는 방법

1. generate_sd2_images.py로 이미지 생성
2. make_captions.py로 캡션(메타데이터) 생성
3. split_dataset.py로 train/validation 스플릿(5%)
4. convert_gustavosta_dataset.py로 gustavosta dataset 변환(train/eval)

- 파일 앞에 prefix를 붙이고
- metadata.jsonl을 생성함

5. gustavosta 데이터를 train/validation 각각에 복사, metadata.jsonl은 cat

- `cat gustavosta_train_images/metadata.jsonl >> train/metadata.jsonl`
- `cat gustavosta_eval_images/metadata.jsonl >> validation/metadata.jsonl`

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
  ```

- 4.0에서 랜덤 10개 가져오고, 학습 프롬프트에서 10개 가져와서 생성
  - prompt_inf.py
  - 결과: resources/\*.txt

## coca training

- make_cocadata.py로 train_coca.csv, valid_coca.csv 생성
- open_clip/src/에서 ./start 실행
