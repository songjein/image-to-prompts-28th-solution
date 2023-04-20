import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


if __name__ == "__main__":
    # tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    # model = GPT2LMHeadModel.from_pretrained("diffusion/gpt-outputs")

    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("diffusion/gpt-outputs-2")

    # 생성할 문장을 정의합니다.
    # prompt = "A photograph of"
    prompt = "portrait of"
    # prompt = "an painting of"
    # prompt = "photo of"
    # prompt = "scene of"

    # 문장을 tokenize합니다.
    inputs = tokenizer(prompt, return_tensors="pt")
    print(inputs)

    # 모델을 이용하여 문장을 생성합니다.
    outputs = model.generate(
        inputs["input_ids"],
        min_length=32,
        max_length=77,
        do_sample=True,
        top_p=0.7,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        num_return_sequences=100,
        eos_token_id=50256,  # 학습시 세팅해줘야할 듯 <|endoftext|>
    )

    # 생성된 문장을 decode하여 출력합니다.
    for index, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(generated_text)
