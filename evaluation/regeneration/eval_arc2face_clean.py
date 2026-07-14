"""Self-contained Arc2Face generation eval (offline, local buffalo_l recognizer).

Loads student embeddings from a pkl (produced by insight_test_*.py), generates a face
per identity with Arc2Face, then measures identity preservation by comparing the
generated face to the real source image with the local InsightFace buffalo_l model.

Reports: mean cosine similarity (gen vs real) and pass rate at a cosine threshold.
Env vars: PKL (embeddings pkl), N (num identities), GPU (cuda idx), TAG (output label),
          THRESH (cosine pass threshold, default 0.3).
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")

import sys
sys.path.append("../../third_party/arc2face")              # repo provides arc2face package
sys.path.append("../../third_party/arc2face")

import pickle
import numpy as np
import torch
from PIL import Image

PKL = os.environ.get("PKL", "../../attacks/transfer/val_minus_lfw.pkl")
N = int(os.environ.get("N", "100"))
TAG = os.environ.get("TAG", "minus")
THRESH = float(os.environ.get("THRESH", "0.3"))
OUTDIR = os.environ.get("OUTDIR", f"eval_arc2face_clean_{TAG}")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- Arc2Face pipeline
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from arc2face import CLIPTextModelWrapper, project_face_embs

base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
encoder = CLIPTextModelWrapper.from_pretrained("../../checkpoints/arc2face", subfolder="encoder", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("../../checkpoints/arc2face", subfolder="arc2face", torch_dtype=torch.float16)
pipeline = StableDiffusionPipeline.from_pretrained(
    base_model, text_encoder=encoder, unet=unet, torch_dtype=torch.float16, safety_checker=None
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to("cuda")

# ---------------------------------------------------------------- local recognizer
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(160, 160))


def embed(pil_img):
    """Return normed buffalo_l embedding of the largest detected face, or None."""
    arr = np.array(pil_img.convert("RGB"))
    faces = app.get(arr)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


# ---------------------------------------------------------------- load embeddings
d = pickle.load(open(PKL, "rb"))
se = d["student_embeddings"]
se = torch.cat(se, dim=0) if isinstance(se, list) else se
fn = d["filenames"]
fn = sum(fn, []) if isinstance(fn[0], (list, tuple)) else list(fn)
n = min(N, se.shape[0], len(fn))
print(f"EVAL pkl={PKL} n={n} thresh={THRESH}")

sims = []
gen_fail = 0
det_fail = 0
for idx in range(n):
    id_emb = se[idx][None, :].to(torch.float16)
    id_emb = (id_emb / torch.norm(id_emb, dim=1, keepdim=True)).cuda()
    id_emb = project_face_embs(pipeline, id_emb)
    img = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=2.5,
                   num_images_per_prompt=1).images[0]
    g = embed(img)
    if g is None:
        gen_fail += 1
        sims.append(-1.0)            # no face generated -> count as fail
        continue
    real = embed(Image.open(fn[idx]))
    if real is None:
        det_fail += 1
        continue
    s = torch.nn.functional.cosine_similarity(g, real).item()
    sims.append(s)
    if idx < 20:
        img.save(f"{OUTDIR}/{TAG}_{idx:04}_gen.png")
    if (idx + 1) % 20 == 0:
        valid = [x for x in sims if x > -1]
        pr = np.mean([1.0 if x >= THRESH else 0.0 for x in sims])
        print(f"[{idx+1}/{n}] mean_sim={np.mean(valid):.4f} pass@{THRESH}={pr:.4f} genfail={gen_fail}")

valid = [x for x in sims if x > -1]
pass_rate = float(np.mean([1.0 if x >= THRESH else 0.0 for x in sims])) if sims else 0.0
mean_sim = float(np.mean(valid)) if valid else 0.0
print(f"RESULT_ARC2FACE tag={TAG} n={n} mean_sim={mean_sim:.4f} "
      f"pass_rate@{THRESH}={pass_rate:.4f} gen_fail={gen_fail} det_fail={det_fail}")
with open(f"{OUTDIR}/result.pkl", "wb") as f:
    pickle.dump({"sims": sims, "mean_sim": mean_sim, "pass_rate": pass_rate,
                 "thresh": THRESH, "gen_fail": gen_fail, "det_fail": det_fail}, f)
