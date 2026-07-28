"""Evaluate U-Net reconstructions of FracFace templates against the originals with Face++.

The U-Net already outputs a face image, so no Arc2Face generation step is needed --
the reconstruction is sent to the Face++ compare endpoint directly.

A sample counts as a success when Face++ detects a face in both images and the
confidence exceeds the 1e-5 threshold. Every raw API response is recorded.

Usage:
    python eval_facepp_frac.py --pkl frac_web.pkl --limit 1000 --out facepp_frac_results.pkl
"""

import argparse
import io
import json
import os
import pickle
import time

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"


def tensor_to_pil(t):
    arr = t.detach().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def compare_faces(img_gen, img_real, api_key, api_secret, max_retries=8):
    """Post one pair to Face++, retrying through the free tier's concurrency limit."""
    for attempt in range(max_retries):
        buf1, buf2 = io.BytesIO(), io.BytesIO()
        img_gen.convert("RGB").save(buf1, format="JPEG")
        img_real.convert("RGB").save(buf2, format="JPEG")
        buf1.seek(0)
        buf2.seek(0)
        try:
            r = requests.post(
                COMPARE_URL,
                data={"api_key": api_key, "api_secret": api_secret},
                files={
                    "image_file1": ("gen.jpg", buf1, "image/jpeg"),
                    "image_file2": ("real.jpg", buf2, "image/jpeg"),
                },
                timeout=60,
            )
            res = r.json()
        except Exception as e:
            res = {"error_message": "REQUEST_FAILED: %s" % e}

        err = res.get("error_message", "")
        if "CONCURRENCY_LIMIT_EXCEEDED" in err or "REQUEST_FAILED" in err:
            time.sleep(2 * (attempt + 1))
            continue
        return res
    return res


def grade(res):
    """Return the strictest FAR threshold the pair passes, or a failure reason."""
    if "faces1" not in res or "faces2" not in res:
        return "APIError"
    if not res["faces1"]:
        return "NoFaceGenerated"
    if not res["faces2"]:
        return "NoFaceReal"
    if "confidence" not in res or "thresholds" not in res:
        return "APIError"
    conf, th = res["confidence"], res["thresholds"]
    if conf > th["1e-5"]:
        return "1e-5"
    if conf > th["1e-4"]:
        return "1e-4"
    if conf > th["1e-3"]:
        return "1e-3"
    return "Failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="frac_web.pkl")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="facepp_frac_results.pkl")
    ap.add_argument("--save_images", default="eval_facepp_frac")
    args = ap.parse_args()

    api_key = os.environ["FACEPP_API_KEY"]
    api_secret = os.environ["FACEPP_API_SECRET"]

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    gen_images = torch.cat(data["gen_images"], dim=0)
    filenames = sum([list(b) for b in data["filenames"]], [])
    n = min(args.limit, len(filenames), gen_images.shape[0])
    print("Evaluating %d of %d pairs" % (n, len(filenames)))

    os.makedirs(args.save_images, exist_ok=True)

    records = []
    levels = []
    for idx in range(n):
        img_gen = tensor_to_pil(gen_images[idx])
        real_path = filenames[idx]
        img_real = Image.open(real_path).convert("RGB")

        res = compare_faces(img_gen, img_real, api_key, api_secret)
        level = grade(res)
        levels.append(level)
        records.append({
            "index": idx,
            "filename": real_path,
            "level": level,
            "confidence": res.get("confidence"),
            "thresholds": res.get("thresholds"),
            "response": res,
        })

        if idx < 50:
            img_gen.save(os.path.join(args.save_images, "%04d_gen.png" % idx))
            img_real.save(os.path.join(args.save_images, "%04d_real.png" % idx))

        if (idx + 1) % 25 == 0 or idx == n - 1:
            passed = sum(1 for l in levels if l == "1e-5")
            uniq, cnt = np.unique(levels, return_counts=True)
            print("[%d/%d] 1e-5 success rate: %.4f  %s" % (
                idx + 1, n, passed / len(levels), dict(zip(uniq, cnt.tolist()))))
            with open(args.out, "wb") as f:
                pickle.dump({"records": records, "levels": levels}, f)

    passed = sum(1 for l in levels if l == "1e-5")
    summary = {
        "n": len(levels),
        "success_1e-5": passed,
        "success_rate_1e-5": passed / len(levels) if levels else 0.0,
        "level_counts": {k: int(v) for k, v in zip(*np.unique(levels, return_counts=True))},
    }
    with open(args.out, "wb") as f:
        pickle.dump({"records": records, "levels": levels, "summary": summary}, f)
    with open(args.out.replace(".pkl", ".json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
