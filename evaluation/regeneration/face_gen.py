import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

prompt = "a human standing"
import os
os.makedirs("outputs_zimage", exist_ok=True)
import tqdm
for i in tqdm.tqdm(range(100)):
    images = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9, 
        guidance_scale=0.0,
        num_images_per_prompt=10,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images
    for idx, img in enumerate(images):
        img.save(f"outputs_zimage/{i}_{idx}.png")