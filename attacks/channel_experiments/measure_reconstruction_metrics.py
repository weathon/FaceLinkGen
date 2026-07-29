"""Measure PSNR and SSIM for every final U-Net and ours reconstruction."""

import csv
import json
import os
import statistics

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS = [
    ("fracface_fixed", "FracFace fixed"),
    ("partialface_fixed", "PartialFace fixed"),
    (
        "fracface_random_train_fixed_test",
        "FracFace random train / fixed test",
    ),
    (
        "partialface_random_train_fixed_test",
        "PartialFace random train / fixed test",
    ),
    ("minusface_random", "MinusFace random"),
]

results = {
    "definition": {
        "images_per_config": 300,
        "color": "RGB",
        "comparison_size": [112, 112],
        "resize": "PIL bilinear",
        "data_range": 1.0,
        "psnr": "skimage.metrics.peak_signal_noise_ratio",
        "ssim": (
            "skimage.metrics.structural_similarity, "
            "channel_axis=2, default window"
        ),
        "aggregation": "mean of per-image metrics",
        "ours_images_per_identity": 1,
    },
    "results": {},
}
summary_rows = []

for config_name, display_name in CONFIGS:
    results["results"][config_name] = {}
    for attack in ["unet", "ours"]:
        manifest_path = os.path.join(
            ROOT,
            "artifacts",
            "new_plan",
            "reconstructions",
            attack,
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
            source = source.resize((112, 112), Image.BILINEAR)
            source_array = np.asarray(source).astype(np.float32) / 255.0

            if attack == "unet":
                reconstruction_path = record["output"]
            else:
                complete_path = os.path.join(
                    ROOT,
                    "artifacts",
                    "new_plan",
                    "generated",
                    "ours",
                    config_name,
                    "%04d" % record["index"],
                    "complete.json",
                )
                with open(complete_path) as complete_file:
                    reconstruction_paths = json.load(complete_file)["images"]
                assert len(reconstruction_paths) == 1
                reconstruction_path = reconstruction_paths[0]

            reconstruction = Image.open(
                reconstruction_path
            ).convert("RGB")
            reconstruction = reconstruction.resize(
                (112, 112),
                Image.BILINEAR,
            )
            reconstruction_array = (
                np.asarray(reconstruction).astype(np.float32) / 255.0
            )

            per_image.append({
                "index": record["index"],
                "source": record["source"],
                "reconstruction": reconstruction_path,
                "psnr_db": float(peak_signal_noise_ratio(
                    source_array,
                    reconstruction_array,
                    data_range=1.0,
                )),
                "ssim": float(structural_similarity(
                    source_array,
                    reconstruction_array,
                    data_range=1.0,
                    channel_axis=2,
                )),
            })

        psnr_values = [record["psnr_db"] for record in per_image]
        ssim_values = [record["ssim"] for record in per_image]
        summary = {
            "display_name": display_name,
            "attack": attack,
            "n": len(per_image),
            "mean_psnr_db": statistics.mean(psnr_values),
            "std_psnr_db": statistics.pstdev(psnr_values),
            "median_psnr_db": statistics.median(psnr_values),
            "min_psnr_db": min(psnr_values),
            "max_psnr_db": max(psnr_values),
            "mean_ssim": statistics.mean(ssim_values),
            "std_ssim": statistics.pstdev(ssim_values),
            "median_ssim": statistics.median(ssim_values),
            "min_ssim": min(ssim_values),
            "max_ssim": max(ssim_values),
            "per_image": per_image,
        }
        results["results"][config_name][attack] = summary
        summary_rows.append({
            key: value for key, value in summary.items()
            if key != "per_image"
        })

output_root = os.path.join(
    ROOT,
    "artifacts",
    "new_plan",
    "metrics",
)
os.makedirs(output_root, exist_ok=True)
json_path = os.path.join(
    output_root,
    "reconstruction_psnr_ssim.json",
)
with open(json_path, "w") as output_file:
    json.dump(results, output_file, indent=2)

csv_path = os.path.join(
    output_root,
    "reconstruction_psnr_ssim_summary.csv",
)
with open(csv_path, "w", newline="") as output_file:
    writer = csv.DictWriter(
        output_file,
        fieldnames=list(summary_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(summary_rows)

for row in summary_rows:
    print(
        "%s | %s | n=%d | PSNR=%.6f | SSIM=%.6f"
        % (
            row["display_name"],
            row["attack"],
            row["n"],
            row["mean_psnr_db"],
            row["mean_ssim"],
        ),
        flush=True,
    )
print(json_path, flush=True)
print(csv_path, flush=True)
