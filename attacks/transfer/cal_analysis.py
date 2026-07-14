"""
Calibration analysis (BLACK-BOX outputs only; we never read minus/frac internals).

Goal: measure the EXACT post-norm tensors the student sees for the two targets
(minusface, fracface) vs. our proxy candidates, on a CALIBRATION set drawn from the
TRAIN split (never the 30 val faces -- keeps val honest).

Measured per domain:
  - 2D average amplitude spectrum  |FFT|  (captures the 8x8 block grid as periodic peaks)
  - radial amplitude envelope (for plotting + spectral matching)
  - block-boundary energy profile (mean |pixel diff| as a function of position mod 8)
  - pixel histogram stats (kurtosis, fraction clamped at +/-5)

Saves cal_targets.pt with the averaged 2D amplitude envelopes -> used later to build a
spectral-matching proxy. Also dumps a comparison figure + sample grids.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
sys.path = [p for p in sys.path if p != "../.."]
sys.path.insert(0, "../../methods/fracface")
import data2npy  # black box, val/analysis only

import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)

mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)

tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


# ---- exact pipeline pieces copied from insight_train_minus.py ----
def convert_batch(conv_raw, convert=True):
    conv_raw = conv_raw.to(device)
    with torch.no_grad():
        out = mface(conv_raw)[5] if convert else conv_raw
    imgs = out.float()
    imgs = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    imgs = (imgs - 0.5) / 0.5
    return imgs


def highpass(img, strength=1.0, kernel_size=5):
    blurred = TF.gaussian_blur(img, (kernel_size, kernel_size))
    hp = (img - blurred) * strength
    hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5
    return hp


def norm(imgs, eps=1e-6, clamp_val=5.0):
    mean = imgs.mean(dim=(1, 2, 3), keepdim=True)
    std = imgs.std(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mean) / (std + eps)).clamp(-clamp_val, clamp_val)


# ---- load calibration faces from TRAIN split only ----
root = '/path/to/casia-webface'
train_paths = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, split = line.strip().split()
        if split == "train":
            train_paths.append(os.path.join(root, fn))
random.shuffle(train_paths)
N = 96
cal_paths = train_paths[:N]
print(f"calibration faces: {len(cal_paths)} (from train split)")

imgs = torch.stack([tf_conv(Image.open(p).convert("RGB").resize((112, 112))) for p in cal_paths]).to(device)
print("faces:", imgs.shape, float(imgs.min()), float(imgs.max()))

# ---- TARGET 1: minusface -> highpass(k5) -> norm ----
t_minus = norm(highpass(convert_batch(imgs, convert=True), 1.0, 5))

# ---- TARGET 2: fracface -> highpass(k5) -> norm ----
frac_list = []
for p in cal_paths:
    img = Image.open(p).convert("RGB").resize((112, 112))
    o = data2npy.preprocess_and_return(img, 1)[0]            # (81,112,112)
    frac_list.append(o.mean(dim=0, keepdim=True).repeat(3, 1, 1))
frac_raw = torch.stack(frac_list).to(device)
t_frac = norm(highpass(convert_batch(frac_raw, convert=False), 1.0, 5))

# ---- PROXY candidates (operate on raw face, convert=False path) ----
face_conv = convert_batch(imgs, convert=False)  # grayscale minmax [-1,1]


def fft_highpass(img, cutoff=8):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    Fimg = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=img.device).view(-1, 1) - cy
    xx = torch.arange(W, device=img.device).view(1, -1) - cx
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2))
    mask = (r > cutoff).float()[None, None]
    out = torch.fft.ifft2(torch.fft.ifftshift(Fimg * mask, dim=(-2, -1))).real
    mn = out.amin(dim=(1, 2, 3), keepdim=True)
    mx = out.amax(dim=(1, 2, 3), keepdim=True)
    return (out - mn) / (mx - mn + 1e-8) / 2 + 0.5


proxies = {
    "gauss_hp_k5":  norm(highpass(face_conv, 1.0, 5)),
    "gauss_hp_k21": norm(highpass(face_conv, 1.0, 21)),
    "fft_hp_c8":    norm(fft_highpass(face_conv, 8)),
}

# ---- metrics ----
def avg_amp2d(t):
    """mean over batch of |fftshift(fft2(gray))|, returns (H,W) on cpu."""
    g = t.mean(dim=1)
    A = torch.fft.fftshift(torch.fft.fft2(g.double()), dim=(-2, -1)).abs()
    return A.mean(0).cpu()


def radial_from_2d(A):
    H, W = A.shape
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).round().long()
    rmax = int(r.max())
    s = torch.bincount(r.flatten(), weights=A.flatten(), minlength=rmax + 1)
    c = torch.bincount(r.flatten(), minlength=rmax + 1).clamp(min=1)
    return (s / c).numpy()


def block_profile(t, block=8):
    """mean |horizontal diff| as a function of (column index mod block).
    A spike at index 0 == strong block-boundary (DCT/JPEG-like) artifacts."""
    g = t.mean(dim=1)                       # (B,H,W)
    d = (g[:, :, 1:] - g[:, :, :-1]).abs()  # (B,H,W-1) diff at boundary between col j and j+1
    cols = torch.arange(d.shape[-1])
    prof = np.zeros(block)
    for m in range(block):
        sel = (cols % block) == m
        prof[m] = d[:, :, sel].mean().item()
    return prof


def hist_stats(t):
    x = t.flatten().double()
    mu, sd = x.mean(), x.std()
    z = (x - mu) / (sd + 1e-9)
    kurt = (z ** 4).mean().item() - 3.0
    clamped = ((t.abs() >= 4.999).float().mean()).item()
    return kurt, clamped


domains = {"MINUS(target)": t_minus, "FRAC(target)": t_frac, **proxies}

print("\n=== block-boundary profile (mean|dx| by col%8; spike@0 => blocking) ===")
amp2d = {}
radial = {}
for name, t in domains.items():
    amp2d[name] = avg_amp2d(t)
    radial[name] = radial_from_2d(amp2d[name])
    prof = block_profile(t)
    kurt, clamped = hist_stats(t)
    spike = prof[0] / (prof[1:].mean() + 1e-9)
    print(f"{name:16s} std={t.std():.3f} kurt={kurt:+.2f} clamp%={clamped*100:5.2f} "
          f"blockspike={spike:.3f}  prof={np.round(prof,3)}")

# radial PSD-shape distance to each target
print("\n=== radial log-amp L2 distance to targets (shape, peak-normalized) ===")
def shape_dist(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    la = np.log10(a / (a.max() + 1e-12) + 1e-9)
    lb = np.log10(b / (b.max() + 1e-12) + 1e-9)
    return float(np.linalg.norm(la - lb))
for tgt in ["MINUS(target)", "FRAC(target)"]:
    row = {nm: round(shape_dist(radial[tgt], radial[nm]), 3) for nm in proxies}
    print(f"  vs {tgt}: {row}")

# ---- save calibration constants + figures ----
torch.save({
    "amp2d_minus": amp2d["MINUS(target)"],
    "amp2d_frac": amp2d["FRAC(target)"],
    "radial_minus": radial["MINUS(target)"],
    "radial_frac": radial["FRAC(target)"],
}, "cal_targets.pt")
print("\nsaved cal_targets.pt")

# radial plot
plt.figure(figsize=(10, 6))
for name in domains:
    plt.plot(np.log10(radial[name] + 1e-9), label=name)
plt.xlabel("radial freq"); plt.ylabel("log10 mean|F|"); plt.legend(fontsize=8)
plt.title("Radial amplitude envelope (post-norm student inputs)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("cal_radial.png", dpi=100)

# 2D log-amplitude maps for the two targets (show block-grid peaks)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for i, name in enumerate(["MINUS(target)", "FRAC(target)", "gauss_hp_k5", "fft_hp_c8"]):
    ax[i].imshow(np.log10(amp2d[name].numpy() + 1e-9), cmap='viridis')
    ax[i].set_title(name); ax[i].axis('off')
plt.tight_layout(); plt.savefig("cal_amp2d.png", dpi=100)

# sample grids
def disp(t):
    return ((t[:8] - t[:8].amin(dim=(1,2,3),keepdim=True)) /
            (t[:8].amax(dim=(1,2,3),keepdim=True) - t[:8].amin(dim=(1,2,3),keepdim=True) + 1e-8))
save_image(disp(t_minus), "cal_sample_minus.png", nrow=4)
save_image(disp(t_frac), "cal_sample_frac.png", nrow=4)
print("saved cal_radial.png cal_amp2d.png cal_sample_minus.png cal_sample_frac.png")
