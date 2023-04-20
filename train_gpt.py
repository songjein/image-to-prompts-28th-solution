import random

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def fine_tune_gpt2(
    model_name,
    train_file,
    eval_file,
    output_dir,
    epochs=1,
    batch_size=4,
    max_seq_length=128,
):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=train_file, block_size=max_seq_length
    )
    eval_dataset = TextDataset(
        tokenizer=tokenizer, file_path=eval_file, block_size=max_seq_length
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=1000,
        save_total_limit=2,
        logging_dir="./diffusion/gpt-logs/",
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        weight_decay=0.1,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=1000,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    trainer.save_model(output_dir)


if __name__ == "__main__":
    random.seed(42)

    with open("./resources/v7.txt") as f:
        texts = []
        for line in f:
            texts.append(line)

    random.shuffle(texts)

    n_train = int(len(texts) * 0.95)
    train_data = texts[:n_train]
    valid_data = texts[n_train:]

    with open("./diffusion/gpt_train.txt", "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line.replace("\n", "<|endoftext|>\n"))

    with open("./diffusion/gpt_valid.txt", "w", encoding="utf-8") as f:
        for line in valid_data:
            f.write(line.replace("\n", "<|endoftext|>\n"))

    # model_name = "distilgpt2"  # GPT-2 모델 사용
    train_file = "./diffusion/gpt_train.txt"
    valid_file = "./diffusion/gpt_valid.txt"
    output_dir = "./diffusion/gpt-outputs-gpt2/"

    model_name = "gpt2"  # "distilgpt2"
    epochs = 5
    batch_size = 128
    max_seq_length = 77

    fine_tune_gpt2(
        model_name,
        train_file,
        valid_file,
        output_dir,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
