"""Arc2Face generation eval with all 3 comparators: Face++, Amazon(edenai), local InsightFace.

For each identity: generate N_IMG faces from the student embedding via Arc2Face, compare each
generated face to the real source image with:
  - Face++ compare API  -> pass at FAR levels 1e-5 / 1e-4 / 1e-3 (its returned thresholds)
  - Amazon (edenai)     -> pass if confidence > 80
  - local buffalo_l     -> cosine similarity (reported, pass at 0.3 for reference)
Reports per-image pass rates at each Face++ FAR level, at-least-one-of-N passed, Amazon pass
rate, and mean local cosine.

Env: PKL, N (identities), N_IMG (gen per id), GPU, TAG.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")

import sys, pickle, time
from io import BytesIO
import numpy as np
import torch
import requests
from PIL import Image

sys.path.append("../../third_party/arc2face"); sys.path.append("../../third_party/arc2face")

PKL = os.environ.get("PKL", "../../attacks/transfer/val_minus_lfw.pkl")
N = int(os.environ.get("N", "100"))
N_IMG = int(os.environ.get("N_IMG", "5"))
TAG = os.environ.get("TAG", "minus")
OUTDIR = os.environ.get("OUTDIR", f"eval_arc2face_3m_{TAG}")
os.makedirs(OUTDIR, exist_ok=True)

FPP_KEY = "YOUR_FACEPP_API_KEY"
FPP_SECRET = "YOUR_FACEPP_API_SECRET"
EDENAI_TOKEN = "YOUR_EDENAI_TOKEN"


def fpp_compare(gen_img, real_img):
    """Face++ compare. Returns dict with 'confidence' and 'thresholds', or None on failure."""
    b1, b2 = BytesIO(), BytesIO()
    gen_img.save(b1, format="JPEG"); b1.seek(0)
    real_img.save(b2, format="JPEG"); b2.seek(0)
    files = {"image_file1": ("a.jpg", b1, "image/jpeg"), "image_file2": ("b.jpg", b2, "image/jpeg")}
    data = {"api_key": FPP_KEY, "api_secret": FPP_SECRET}
    for attempt in range(4):
        try:
            r = requests.post("https://api-us.faceplusplus.com/facepp/v3/compare",
                              data=data, files=files, timeout=30)
            j = r.json()
            if "confidence" in j or "faces1" in j:
                return j
            # rate limited -> backoff
            time.sleep(2 + attempt * 2)
            b1.seek(0); b2.seek(0)
        except Exception as e:
            time.sleep(2 + attempt * 2)
            b1.seek(0); b2.seek(0)
    return None


def amazon_compare(gen_img, real_img):
    """edenai amazon face_compare. Returns max confidence (0-100) or None."""
    b1, b2 = BytesIO(), BytesIO()
    gen_img.save(b1, format="PNG"); b1.seek(0)
    real_img.save(b2, format="PNG"); b2.seek(0)
    files = {"file1": ("a.png", b1, "image/png"), "file2": ("b.png", b2, "image/png")}
    try:
        r = requests.post("https://api.edenai.run/v2/image/face_compare",
                          data={"providers": "amazon"}, files=files,
                          headers={"Authorization": f"Bearer {EDENAI_TOKEN}"}, timeout=60)
        j = r.json()
        items = j.get("amazon", {}).get("items", [])
        if not items:
            return 0.0
        return max(i["confidence"] for i in items)
    except Exception:
        return None


# ----- Arc2Face pipeline -----
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from arc2face import CLIPTextModelWrapper, project_face_embs
base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
encoder = CLIPTextModelWrapper.from_pretrained("../../checkpoints/arc2face", subfolder="encoder", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("../../checkpoints/arc2face", subfolder="arc2face", torch_dtype=torch.float16)
pipeline = StableDiffusionPipeline.from_pretrained(
    base_model, text_encoder=encoder, unet=unet, torch_dtype=torch.float16, safety_checker=None)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")

# ----- local recognizer -----
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(160, 160))


def insight_sim(gen_img, real_img):
    try:
        f1 = app.get(np.array(gen_img.convert("RGB")))
        f2 = app.get(np.array(real_img.convert("RGB")))
        if not f1 or not f2:
            return None
        e1 = torch.from_numpy(f1[0].normed_embedding)
        e2 = torch.from_numpy(f2[0].normed_embedding)
        return torch.nn.functional.cosine_similarity(e1[None], e2[None]).item()
    except Exception:
        return None


# ----- data -----
d = pickle.load(open(PKL, "rb"))
se = d["student_embeddings"]; se = torch.cat(se, dim=0) if isinstance(se, list) else se
fn = d["filenames"]; fn = sum(fn, []) if isinstance(fn[0], (list, tuple)) else list(fn)
n = min(N, se.shape[0], len(fn))
print(f"EVAL3 pkl={PKL} n={n} n_img={N_IMG}")

fpp_far = {"1e-5": [], "1e-4": [], "1e-3": []}      # per-image pass flags
fpp_atleast = {"1e-5": [], "1e-4": [], "1e-3": []}  # per-identity (>=1 of N_IMG)
amazon_pass = []
insight_sims = []

for idx in range(n):
    id_emb = se[idx][None, :].to(torch.float16)
    id_emb = (id_emb / torch.norm(id_emb, dim=1, keepdim=True)).cuda()
    id_emb = project_face_embs(pipeline, id_emb)
    images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=2.5,
                      num_images_per_prompt=N_IMG).images
    real = Image.open(fn[idx]).convert("RGB")
    per_id = {"1e-5": [], "1e-4": [], "1e-3": []}
    for i, gimg in enumerate(images):
        if idx < 10 and i == 0:
            gimg.save(f"{OUTDIR}/{TAG}_{idx:04}_gen.png")
        j = fpp_compare(gimg, real)
        if j is not None and "confidence" in j and "thresholds" in j:
            conf, th = j["confidence"], j["thresholds"]
            for lvl in fpp_far:
                p = 1.0 if conf >= th[lvl] else 0.0
                fpp_far[lvl].append(p); per_id[lvl].append(p)
        a = amazon_compare(gimg, real)
        if a is not None:
            amazon_pass.append(1.0 if a > 80 else 0.0)
        s = insight_sim(gimg, real)
        if s is not None:
            insight_sims.append(s)
    for lvl in fpp_far:
        if per_id[lvl]:
            fpp_atleast[lvl].append(1.0 if any(per_id[lvl]) else 0.0)
    if (idx + 1) % 5 == 0:
        m = lambda x: float(np.mean(x)) if x else 0.0
        print(f"[{idx+1}/{n}] FPP per-img 1e-3={m(fpp_far['1e-3']):.3f} 1e-4={m(fpp_far['1e-4']):.3f} "
              f"1e-5={m(fpp_far['1e-5']):.3f} | amazon={m(amazon_pass):.3f} | insight_sim={m(insight_sims):.3f}")

m = lambda x: float(np.mean(x)) if x else 0.0
print("RESULT_3M tag=%s n=%d | FPP_perimg 1e-5=%.4f 1e-4=%.4f 1e-3=%.4f | "
      "FPP_atleast1 1e-5=%.4f 1e-4=%.4f 1e-3=%.4f | amazon_pass=%.4f | insight_mean_sim=%.4f" % (
      TAG, n,
      m(fpp_far["1e-5"]), m(fpp_far["1e-4"]), m(fpp_far["1e-3"]),
      m(fpp_atleast["1e-5"]), m(fpp_atleast["1e-4"]), m(fpp_atleast["1e-3"]),
      m(amazon_pass), m(insight_sims)))
with open(f"{OUTDIR}/result.pkl", "wb") as f:
    pickle.dump({"fpp_far": fpp_far, "fpp_atleast": fpp_atleast,
                 "amazon_pass": amazon_pass, "insight_sims": insight_sims}, f)
