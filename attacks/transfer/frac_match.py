"""
Proxy-difficulty matching to FRACFACE (per user 2026-05-30).
Hypothesis: my proxy is too EASY (too face-like) vs fracface, so the student doesn't learn hard
enough features. Diagnostic: INIT loss = mean(1 - cos(student_init(input), teacher_clean)) with the
UNTRAINED student. If a proxy's init loss << fracface's, it's too easy.

Reports init-loss + simple stats for: frac target, minus target (ref), and a sweep of highpass /
block-DCT proxy configs. Saves a visual grid (frac vs proxies). Goal: find proxy params whose init
loss + look match fracface.
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
from torchjpeg import dct
from onnx2torch import convert
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(5)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])

mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).eval()  # UNTRAINED (init)
emb_root = '/path/to/casia-webface/insight_embeddings'


def highpass(img, strength=1.0, k=5):
    hp = (img - TF.gaussian_blur(img, (k, k))) * strength
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def cvt(o):
    o = o.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = o.amin(dim=(1, 2, 3), keepdim=True); mx = o.amax(dim=(1, 2, 3), keepdim=True)
    return ((o - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def block_dct(img, block=8):
    B, C, H, W = img.shape; nh, nw = H // block, W // block
    x = img.view(B, C, nh, block, nw, block).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, nh * nw, block, block)
    return dct.block_dct(x), (B, C, nh, nw)


def iblock(xd, meta, block=8):
    B, C, nh, nw = meta
    return dct.block_idct(xd).view(B, C, nh, nw, block, block).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, nh * block, nw * block)


def dct_hp(img, nz=2):
    xd, m = block_dct(img); xd[:, :, :, :nz, :nz] = 0
    xb = iblock(xd, m); mn = xb.amin(dim=(1, 2, 3), keepdim=True); mx = xb.amax(dim=(1, 2, 3), keepdim=True)
    return (xb - mn) / (mx - mn + 1e-8) / 2 + 0.5


def dctrand(img, drop, gain, nz=2):
    xd, m = block_dct(img); xd[:, :, :, :nz, :nz] = 0
    if gain > 0: xd = xd * torch.exp(gain * torch.randn_like(xd))
    if drop > 0: xd = xd * (torch.rand_like(xd) > drop).float()
    return iblock(xd, m)


# calibration faces from train split (with teacher embeddings)
root = '/path/to/casia-webface'
paths = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        if sp == "train": paths.append(fn)
random.shuffle(paths)
paths = paths[:64]
imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in paths]).to(device)
teacher = torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in paths]).float().to(device)
teacher = F.normalize(teacher, dim=1)

with torch.no_grad():
    mo = mface(imgs)[5]
    frac_list = [data2npy.preprocess_and_return(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112)), 1)[0].mean(0, keepdim=True).repeat(3, 1, 1) for p in paths]
    frac_raw = torch.stack(frac_list).to(device)

face = cvt(imgs)


def init_loss(inp):
    with torch.no_grad():
        e = F.normalize(student(inp), dim=1)
    return (1 - (e * teacher).sum(1)).mean().item()


# targets (val pipeline): convert -> highpass(k5) -> norm
t_frac = norm(highpass(cvt(frac_raw), 1, 5))
t_minus = norm(highpass(cvt(mo), 1, 5))

cands = {
    "FRAC (target)": t_frac,
    "MINUS (target)": t_minus,
    "gauss_hp k5 (current val-hp)": norm(highpass(face, 1, 5)),
    "gauss_hp k3": norm(highpass(face, 1, 3)),
    "gauss_hp k9": norm(highpass(face, 1, 9)),
    "gauss_hp k15": norm(highpass(face, 1, 15)),
    "dct_hp nz2 (deliverable base)": norm(highpass(dct_hp(face, 2), 1, 5)),
    "dct_hp nz4": norm(highpass(dct_hp(face, 4), 1, 5)),
    "dct_hp nz6": norm(highpass(dct_hp(face, 6), 1, 5)),
    "dctrand d.3 g.5 (deliverable)": norm(dctrand(highpass(face, 1, 5), .3, .5)),
    "dctrand d.6 g1.0": norm(dctrand(highpass(face, 1, 5), .6, 1.0)),
    "dctrand d.8 g1.5": norm(dctrand(highpass(face, 1, 5), .8, 1.5)),
    "hp_then_hp (k5 twice)": norm(highpass(highpass(face, 1, 5), 1, 5)),
}

print("=== INIT loss (1-cos to teacher, UNTRAINED student). Higher=harder ===")
print(f"{'candidate':34s} {'init_loss':>10s} {'std':>7s} {'kurt':>7s}")
rows = {}
for name, t in cands.items():
    il = init_loss(t)
    x = t.flatten().double(); z = (x - x.mean()) / (x.std() + 1e-9); kurt = (z**4).mean().item() - 3
    rows[name] = il
    print(f"{name:34s} {il:10.4f} {t.std().item():7.3f} {kurt:7.2f}")

print(f"\n>>> FRAC target init loss = {rows['FRAC (target)']:.4f}  (match the proxy to THIS)")
print(f">>> MINUS target init loss = {rows['MINUS (target)']:.4f}")

def disp(t):
    a = t[:8]; mn = a.amin(dim=(1,2,3),keepdim=True); mx = a.amax(dim=(1,2,3),keepdim=True)
    return (a - mn) / (mx - mn + 1e-8)
grid = torch.cat([disp(cands[k]) for k in ["FRAC (target)", "gauss_hp k5 (current val-hp)",
        "dctrand d.3 g.5 (deliverable)", "dctrand d.8 g1.5", "gauss_hp k3", "dct_hp nz6"]], 0)
save_image(grid, "frac_match_grid.png", nrow=8)
print("\nsaved frac_match_grid.png (rows: FRAC, gauss_k5, dctrand-deliverable, dctrand-hard, gauss_k3, dct_nz6)")
