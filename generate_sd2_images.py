import os
from argparse import ArgumentParser

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

parser = ArgumentParser()
group = parser.add_argument_group(title="inference settings")
group.add_argument("--prefix", type=str, default="openprompts")
group.add_argument("--gpu-idx", type=int, default=0)
group.add_argument("--stride-idx", type=int, default=0)
group.add_argument("--num-gpus", type=int, default=4)
group.add_argument("--image-dir-path", type=str, default="./diffusion/images")
group.add_argument("--prompts-path", type=str, default="./diffusion/prompts-large.txt")

if __name__ == "__main__":
    # python generate_sd2_images.py --prefix openprompts --gpu-idx 0 --stride-idx 0 --num-gpus 3 --image-dir-path ./diffusion/openprompts_images/ --prompts-path ./resources/openprompts_dedup_075_filtered.txt
    args = parser.parse_args()

    class CFG:
        # specify GPU device
        device = torch.device(
            f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu"
        )
        seed = 42
        generator = torch.Generator(device).manual_seed(seed)
        image_gen_steps = 50
        image_gen_model_id = "stabilityai/stable-diffusion-2"
        image_gen_guidance_scale = 9

    print(CFG.device)

    image_gen_model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float16,
        revision="fp16",
        guidance_scale=9,
    )
    image_gen_model = image_gen_model.to(CFG.device)

    def generate_image(prompt, model):
        image = model(
            prompt,
            num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale,
        ).images[0]
        return image

    os.makedirs(args.image_dir_path, exist_ok=True)

    with open(args.prompts_path) as f:
        for idx, line in enumerate(tqdm(f)):
            prompt = line.strip()

            fn = os.path.join(args.image_dir_path, f"{args.prefix}_{idx:08d}.jpg")
            if idx % args.num_gpus != args.stride_idx:
                continue

            # When resuming after interrupting the process
            if os.path.exists(fn):
                continue

            img = generate_image(prompt, image_gen_model).resize((512, 512))

            img.save(fn)

            # please comment out s steps below
            # print(f"[debug] fn = {fn}")
            # if _ > 4:
            #     break
