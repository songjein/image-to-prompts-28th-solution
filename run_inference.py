import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

    model.load_state_dict(torch.load("./output_git-model-42s-5ep/last_model_ep_4.pth"))

    for i in range(1000):
        path = f"diffusion/images/{i:08d}.jpg"
        image = Image.open(path)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            top_p=0.8,
            no_repeat_ngram_size=1,
        )
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        print(path, "=>", generated_caption)
