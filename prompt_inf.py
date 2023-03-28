import random
import time

import openai

if __name__ == "__main__":
    OPENAI_KEY = "-"
    openai.api_key = OPENAI_KEY
    completion = openai.ChatCompletion()

    num_aug = 10000

    gpt4_prompts = []
    with open("./chatgpt_prompts_filtered.txt") as f:
        for line in f:
            gpt4_prompts.append(line.strip())

    with open("./chatgpt_3.5_prompts.txt", "w", encoding="utf-8") as f:
        for i in range(num_aug):
            sample_prompts_from_gpt4 = random.choices(gpt4_prompts, k=5)
            prompt_str = "\n".join(
                [f"- {prompt}" for prompt in sample_prompts_from_gpt4]
            )
            prompt = f"""
Examples
{prompt_str}
- hyper realistic photo of very friendly and dystopian crater
- ramen carved out of fractal rose ebony, in the style of hudson river school
- ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus
- a thundering retro robot crane inks on parchment with a droopy french bulldog
- portrait painting of a shimmering greek hero, next to a loud frill-necked lizard
- an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms
- Kaggle employee Phil at a donut shop ordering all the best donuts, with a speech bubble that proclaims "Donuts. It's what's for dinner!"

These are the examples of prompts for generating images.

You are a prompt generator.

Make other prompts in the similar style as many as possible, with variety of random scenes, and range from fairly simple to complex with multiple objects and modifiers. Different words from examples are preferred.
Each prompts must be started with dash(-), not with number.
            """.strip()

            response = completion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Make some prompts as many as possible, with variety of random scenes, and range from fairly simple to complex with multiple objects and modifiers. Different words from examples are preferred.",
                    },
                ],
            )

            prompts = [
                line[1:].strip()
                for line in response["choices"][0]["message"]["content"].split("\n")
            ]

            print(f"[{i}]")
            for prompt in prompts:
                f.write(prompt + "\n")
                print(prompt)

            time.sleep(1)
