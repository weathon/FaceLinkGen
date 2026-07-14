"""
User: "train gaussian, val fracface+gaussian USED TO WORK" but now val_frac~0.96 (~random).
Hunt the regression in the FRAC val path. Train a gaussian-highpass student, then val on several
fracface reductions/variants. The current dataset does out(81,112,112).mean(0) -> if identity is in
the channel variation, mean kills it. Test alternatives.
"""
import os, sys, random
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
sys.path = [p for p in sys.path if p != "../.."]
sys.path.insert(0, "../../methods/fracface")
import data2npy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from onnx2torch import convert
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(1)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).train()
emb_root = '/path/to/casia-webface/insight_embeddings'


def gray_minmax(t):
    if t.dim() == 3: t = t.unsqueeze(0)
    t = t.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True); mx = t.amax(dim=(1, 2, 3), keepdim=True)
    return ((t - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, k=5):
    hp = img - TF.gaussian_blur(img, (k, k))
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


root = '/path/to/casia-webface'
train_p, val_p = [], []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        (train_p if sp == "train" else val_p).append(fn)
random.shuffle(train_p); random.seed(42); val_p = random.sample(val_p, 150); random.seed(1)


def teach_of(paths):
    return F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in paths]).float().to(device), dim=1)


val_teach = teach_of(val_p)
# frac 81-channel outputs for val
print("computing fracface for 150 val faces...", flush=True)
frac81 = []
for p in val_p:
    img = Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))
    frac81.append(data2npy.preprocess_and_return(img, 1)[0])  # (81,112,112)
frac81 = torch.stack(frac81).to(device)  # (150,81,112,112)
print("frac81:", tuple(frac81.shape), "range", float(frac81.min()), float(frac81.max()), flush=True)

# variants of reducing 81 -> 3ch, then val pipeline (+/- highpass)
def red_mean(x): return x.mean(1, keepdim=True).repeat(1, 3, 1, 1)
def red_std(x):  return x.std(1, keepdim=True).repeat(1, 3, 1, 1)
def red_ch0(x):  return x[:, :1].repeat(1, 3, 1, 1)
def red_sumabs(x): return x.abs().sum(1, keepdim=True).repeat(1, 3, 1, 1)
def red_first3(x): return x[:, :3]

VARIANTS = {
    "mean+hp (CURRENT)": lambda: norm(highpass(gray_minmax(red_mean(frac81)))),
    "mean NO-hp":        lambda: norm(gray_minmax(red_mean(frac81))),
    "std+hp":            lambda: norm(highpass(gray_minmax(red_std(frac81)))),
    "ch0+hp":            lambda: norm(highpass(gray_minmax(red_ch0(frac81)))),
    "sumabs+hp":         lambda: norm(highpass(gray_minmax(red_sumabs(frac81)))),
    "first3+hp":         lambda: norm(highpass(gray_minmax(red_first3(frac81)))),
}
V = {k: f() for k, f in VARIANTS.items()}
for k, v in V.items():
    print(f"  {k:20s} shape={tuple(v.shape)} std={v.std():.3f}")
save_image(torch.cat([v[:8] * 0.5 + 0.5 for v in V.values()], 0).clamp(0, 1), "frac_variants_grid.png", nrow=8)
print("saved frac_variants_grid.png rows:", list(V.keys()), flush=True)


def val_cos(v):
    student.eval()
    with torch.no_grad():
        e = F.normalize(student(v), dim=1)
        d = (1 - (e * val_teach).sum(1)).mean().item()
    student.train()
    return d


# train gaussian-highpass student
opt = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-3)
tp = train_p[:20000]; bs = 128
print("\n=== train on highpass(CLEAN); val on each frac variant ===", flush=True)
hdr = "  ".join(f"{k[:14]:>14s}" for k in V)
print(f"{'step':>5s}  {hdr}")
for step in range(0, 401):
    if step > 0:
        ps = random.sample(tp, bs)
        imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in ps]).to(device)
        teach = teach_of(ps)
        inp = norm(highpass(gray_minmax(imgs)))
        en = F.normalize(student(inp), dim=1)
        loss = (1 - (en * teach).sum(1)).mean() + F.l1_loss(en, teach)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0:
        vals = "  ".join(f"{val_cos(v):14.4f}" for v in V.values())
        print(f"{step:5d}  {vals}", flush=True)
