"""Measure RGB PSNR for the saved U-Net reconstructions."""

import json
import math
import os
import statistics

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS = [
    ("partialface_fixed", "PartialFace fixed"),
    ("minusface_random", "MinusFace random"),
    ("fracface_fixed", "FracFace fixed"),
]

results = {
    "definition": {
        "images": 300,
        "color": "RGB",
        "size": [112, 112],
        "source_resize": "torchvision PIL bilinear",
        "data_range": 1.0,
        "aggregation": "mean of per-image PSNR",
        "reconstruction": "saved U-Net PNG",
    },
    "configs": {},
}

for config_name, display_name in CONFIGS:
    manifest_path = os.path.join(
        ROOT,
        "artifacts",
        "new_plan",
        "reconstructions",
        "unet",
        config_name,
        "manifest.jsonl",
    )
    with open(manifest_path) as manifest_file:
        records = [json.loads(line) for line in manifest_file]
    records.sort(key=lambda record: record["index"])
    assert len(records) == 300
    assert [record["index"] for record in records] == list(range(300))

    per_image = []
    for record in records:
        source = Image.open(record["source"]).convert("RGB")
        source = transform_functional.resize(
            source,
            [112, 112],
            interpolation=InterpolationMode.BILINEAR,
        )
        source_tensor = transform_functional.to_tensor(source)

        reconstruction = Image.open(record["output"]).convert("RGB")
        reconstruction_tensor = transform_functional.to_tensor(reconstruction)
        assert reconstruction_tensor.shape == torch.Size([3, 112, 112])

        mse = torch.mean(
            (source_tensor - reconstruction_tensor) ** 2
        ).item()
        if mse == 0:
            raise RuntimeError(
                "zero MSE for %s index %d" % (
                    config_name,
                    record["index"],
                )
            )
        per_image.append({
            "index": record["index"],
            "source": record["source"],
            "reconstruction": record["output"],
            "mse": mse,
            "psnr_db": 10.0 * math.log10(1.0 / mse),
        })

    values = [record["psnr_db"] for record in per_image]
    results["configs"][config_name] = {
        "display_name": display_name,
        "n": len(values),
        "mean_psnr_db": statistics.mean(values),
        "std_psnr_db": statistics.pstdev(values),
        "median_psnr_db": statistics.median(values),
        "min_psnr_db": min(values),
        "max_psnr_db": max(values),
        "per_image": per_image,
    }

output_path = os.path.join(
    ROOT,
    "artifacts",
    "new_plan",
    "metrics",
    "unet_psnr.json",
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as output_file:
    json.dump(results, output_file, indent=2)

for config_name, _ in CONFIGS:
    summary = results["configs"][config_name]
    print(
        "%s n=%d mean=%.6f std=%.6f median=%.6f min=%.6f max=%.6f"
        % (
            config_name,
            summary["n"],
            summary["mean_psnr_db"],
            summary["std_psnr_db"],
            summary["median_psnr_db"],
            summary["min_psnr_db"],
            summary["max_psnr_db"],
        ),
        flush=True,
    )
print(output_path, flush=True)
