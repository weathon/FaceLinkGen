"""Generate five Arc2Face reconstructions for each saved student embedding."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "arc2face"))

from arc2face import CLIPTextModelWrapper, project_face_embs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    records = []
    with open(args.manifest) as f:
        for line in f:
            records.append(json.loads(line))
    records.sort(key=lambda record: record["index"])
    assert len(records) == 300

    checkpoint_root = os.path.join(ROOT, "checkpoints", "arc2face")
    encoder = CLIPTextModelWrapper.from_pretrained(
        checkpoint_root,
        subfolder="encoder",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint_root,
        subfolder="arc2face",
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to("cuda")

    for record in tqdm(records, desc="Arc2Face"):
        sample_dir = os.path.join(args.output, "%04d" % record["index"])
        os.makedirs(sample_dir, exist_ok=True)
        complete_path = os.path.join(sample_dir, "complete.json")
        if os.path.exists(complete_path):
            with open(complete_path) as f:
                complete = json.load(f)
            assert complete["source"] == record["source"]
            for image_path in complete["images"]:
                assert os.path.exists(image_path)
            continue

        embedding = torch.from_numpy(
            np.load(record["output"])
        )[None].to(torch.float16)
        embedding = embedding / torch.norm(
            embedding, dim=1, keepdim=True
        )
        prompt_embeddings = project_face_embs(
            pipeline, embedding.cuda()
        )
        images = pipeline(
            prompt_embeds=prompt_embeddings,
            num_inference_steps=25,
            guidance_scale=2.5,
            num_images_per_prompt=5,
        ).images

        image_paths = []
        for image_index, image in enumerate(images):
            image_path = os.path.join(
                sample_dir, "gen_%d.png" % image_index
            )
            image.save(image_path)
            image_paths.append(image_path)
        with open(complete_path, "w") as f:
            json.dump({
                "index": record["index"],
                "source": record["source"],
                "embedding": record["output"],
                "images": image_paths,
            }, f, indent=2)

    print("generated 5 images for %d identities in %s" % (
        len(records), args.output
    ), flush=True)


if __name__ == "__main__":
    main()
