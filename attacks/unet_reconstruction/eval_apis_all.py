"""Score U-Net reconstructions against the originals with Face++ and Amazon.

The U-Net emits a face image, so reconstructions go straight to the compare endpoints --
no Arc2Face generation step.

Face++  success: a face detected in BOTH images AND confidence above the 1e-5 threshold.
Amazon  success: CompareFaces returns a match. Rekognition applies its SimilarityThreshold
        server-side and returns an empty match list when the faces differ, so non-empty
        means match.

Amazon reports two distinct kinds of non-match and they must not be conflated:

  - empty `items`            -> faces compared, judged different. Counts as a failure.
  - ProviderInvalidInputError -> no face found in the reconstruction, so CompareFaces
                                 rejected the request. This is an attack failure, NOT an
                                 API failure, and it counts in the denominator. Excluding
                                 it inflates the match rate of exactly the methods whose
                                 reconstructions are worst.

Only transport-level failures (no HTTP response) are excluded, and the count is reported.

EdenAI reports similarity in [0,1], not Rekognition's native 0-100. The `confidence > 80`
test in evaluation/regeneration/eval_arc2face_3methods.py can therefore never fire.

Every method is scored on the same subset so the numbers are paired across methods.

    python eval_apis_all.py --limit 300
    python eval_apis_all.py --limit 300 --apis facepp
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

FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"
EDENAI_URL = "https://api.edenai.run/v2/image/face_compare"
MIN_SIZE = 256

METHODS = [
    ("FracFace", "../../artifacts/reconstructions/frac_web.pkl"),
    ("MinusFace", "../../artifacts/reconstructions/minus_web.pkl"),
    ("PartialFace", "../../artifacts/reconstructions/par_web.pkl"),
    ("FracFace (variant)", "../../artifacts/reconstructions/frac_web_corrected_freqmajor.pkl"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--apis", default="facepp,amazon")
ap.add_argument("--subset", default="../../data_splits/val_minus_lfw_filenames.txt")
ap.add_argument("--data_root", default="/path/to/casia-webface")
ap.add_argument("--out", default="../../artifacts/api_results/all_methods")
args = ap.parse_args()

apis = args.apis.split(",")
facepp_key = os.environ["FACEPP_API_KEY"]
facepp_secret = os.environ["FACEPP_API_SECRET"]
edenai_token = os.environ["EDENAI_API_TOKEN"]

tables = {}
order = None
for name, path in METHODS:
    with open(path, "rb") as f:
        d = pickle.load(f)
    gen = torch.cat(d["gen_images"], dim=0)
    fns = sum([list(b) for b in d["filenames"]], [])
    tables[name] = {fn: gen[i] for i, fn in enumerate(fns)}
    if order is None:
        order = fns

with open(args.subset) as f:
    subset = [ln.strip().replace("/path/to/casia-webface", args.data_root)
              for ln in f if ln.strip()]
common = [fn for fn in subset if all(fn in tables[n] for n, _ in METHODS)][:args.limit]
print("methods: %s" % ", ".join(n for n, _ in METHODS))
print("paired subset: %d of %d images\n" % (len(common), len(subset)), flush=True)

records = {name: [] for name, _ in METHODS}
os.makedirs(args.out, exist_ok=True)

for i, fn in enumerate(common):
    real = Image.open(fn).convert("RGB")
    real_jpg = io.BytesIO()
    real.resize((MIN_SIZE, MIN_SIZE), Image.LANCZOS).save(real_jpg, format="JPEG")

    for name, _ in METHODS:
        arr = tables[name][fn].detach().float().clamp(0, 1).permute(1, 2, 0).numpy()
        gen = Image.fromarray((arr * 255).astype(np.uint8)).resize((MIN_SIZE, MIN_SIZE),
                                                                   Image.LANCZOS)
        gen_jpg = io.BytesIO()
        gen.save(gen_jpg, format="JPEG")
        rec = {"index": i, "filename": fn}

        if "facepp" in apis:
            # Face++ free tier serialises requests; a 403 here is a queueing signal.
            while True:
                gen_jpg.seek(0)
                real_jpg.seek(0)
                res = requests.post(
                    FACEPP_URL,
                    data={"api_key": facepp_key, "api_secret": facepp_secret},
                    files={"image_file1": ("gen.jpg", gen_jpg, "image/jpeg"),
                           "image_file2": ("real.jpg", real_jpg, "image/jpeg")},
                    timeout=60).json()
                if "error_message" not in res or "CONCURRENCY_LIMIT_EXCEEDED" not in res["error_message"]:
                    break
                time.sleep(2)

            if not res["faces1"]:
                level = "NoFaceGenerated"
            elif not res["faces2"]:
                level = "NoFaceReal"
            elif res["confidence"] > res["thresholds"]["1e-5"]:
                level = "1e-5"
            elif res["confidence"] > res["thresholds"]["1e-4"]:
                level = "1e-4"
            elif res["confidence"] > res["thresholds"]["1e-3"]:
                level = "1e-3"
            else:
                level = "Failed"
            rec["facepp_level"] = level
            rec["facepp_response"] = res

        if "amazon" in apis:
            gen_jpg.seek(0)
            real_jpg.seek(0)
            res = requests.post(
                EDENAI_URL,
                data={"providers": "amazon"},
                files={"file1": ("gen.jpg", gen_jpg, "image/jpeg"),
                       "file2": ("real.jpg", real_jpg, "image/jpeg")},
                headers={"Authorization": "Bearer %s" % edenai_token},
                timeout=90).json()
            node = res["amazon"]
            if node["status"] == "success":
                items = node["items"]
                rec["amazon_matched"] = len(items) > 0
                rec["amazon_confidence"] = max(i["confidence"] for i in items) * 100 if items else 0.0
            elif node["error"]["type"] == "ProviderInvalidInputError":
                # No face detected in the reconstruction: an attack failure, not an API one.
                rec["amazon_matched"] = False
                rec["amazon_confidence"] = 0.0
                rec["amazon_no_face"] = True
            else:
                raise RuntimeError("unexpected Amazon error on %s: %s" % (fn, node["error"]))
            rec["amazon_response"] = res

        records[name].append(rec)

    if (i + 1) % 20 == 0 or i == len(common) - 1:
        print("[%d/%d]" % (i + 1, len(common)), flush=True)
        for name, _ in METHODS:
            rs = records[name]
            line = "    %-20s" % name
            if "facepp" in apis:
                line += "  Face++ 1e-5 %5.1f%%" % (
                    100 * sum(r["facepp_level"] == "1e-5" for r in rs) / len(rs))
            if "amazon" in apis:
                line += "  Amazon %5.1f%% (no-face %d)" % (
                    100 * sum(r["amazon_matched"] for r in rs) / len(rs),
                    sum("amazon_no_face" in r for r in rs))
            print(line, flush=True)
        with open(os.path.join(args.out, "records.pkl"), "wb") as f:
            pickle.dump(records, f)

summary = {}
for name, _ in METHODS:
    rs = records[name]
    s = {"n": len(rs)}
    if "facepp" in apis:
        levels = [r["facepp_level"] for r in rs]
        s["facepp_1e-5"] = sum(l == "1e-5" for l in levels)
        s["facepp_1e-5_rate"] = s["facepp_1e-5"] / len(rs)
        s["facepp_levels"] = {a: int(b) for a, b in zip(*np.unique(levels, return_counts=True))}
    if "amazon" in apis:
        s["amazon_match"] = sum(r["amazon_matched"] for r in rs)
        s["amazon_match_rate"] = s["amazon_match"] / len(rs)
        s["amazon_no_face"] = sum("amazon_no_face" in r for r in rs)
        matched = [r["amazon_confidence"] for r in rs if r["amazon_matched"]]
        s["amazon_mean_conf_when_matched"] = float(np.mean(matched)) if matched else None
    summary[name] = s

with open(os.path.join(args.out, "records.pkl"), "wb") as f:
    pickle.dump({"records": records, "summary": summary}, f)
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\n" + json.dumps(summary, indent=2))
