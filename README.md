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
