from collections import Counter

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

if __name__ == "__main__":
    texts = []
    with open("./resources/v7.txt") as f:
        for line in f:
            texts.append(" ".join(line.strip().split()[:3]))

    counter = Counter(texts)

    prompts = []
    for prompt, count in counter.most_common()[:500]:
        prompts.append(prompt)
        print(prompt, count)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("diffusion/gpt-outputs-gpt2")

    with open("./resources/gpt-prompts-generated.txt", "w", encoding="utf-8") as f:
        for prompt in tqdm(prompts):
            # 문장을 tokenize합니다.
            inputs = tokenizer(prompt, return_tensors="pt")

            # 모델을 이용하여 문장을 생성합니다.
            outputs = model.generate(
                inputs["input_ids"],
                min_length=32,
                max_length=77,
                do_sample=True,
                top_p=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                num_return_sequences=500,
                eos_token_id=50256,  # 학습시 세팅해줘야할 듯 <|endoftext|>
            )

            # 생성된 문장을 decode하여 출력합니다.
            for index, output in enumerate(outputs):
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                print(generated_text)
                f.write(f"{generated_text}\n")
