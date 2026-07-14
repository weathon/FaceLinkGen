"""
Precompute the target radial amplitude envelopes (post-norm student inputs) for minus and
frac, from a CALIBRATION set of TRAIN-split faces (never the 30 val faces). Saves a small
constant file proxy_envelopes.pt that the training script loads -- so training NEVER calls
minus/frac on its data; it only uses these ~80-number frequency-analysis summaries.
This is the spectral-calibration approach sanctioned by HANDOFF TODO #1.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
sys.path = [p for p in sys.path if p != "../.."]
sys.path.insert(0, "../../methods/fracface")
import data2npy
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(123)  # distinct seed from val (42); calibration faces != val faces
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])

mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)


def highpass(img, k=5):
    hp = img - TF.gaussian_blur(img, (k, k))
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def cvt(o):
    o = o.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = o.amin(dim=(1, 2, 3), keepdim=True); mx = o.amax(dim=(1, 2, 3), keepdim=True)
    return ((o - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def radial_amp(t):
    g = t.mean(dim=1)
    A = torch.fft.fftshift(torch.fft.fft2(g.double()), dim=(-2, -1)).abs().mean(0).cpu()
    H, W = A.shape; cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).round().long()
    rmax = int(r.max())
    s = torch.bincount(r.flatten(), weights=A.flatten(), minlength=rmax + 1)
    c = torch.bincount(r.flatten(), minlength=rmax + 1).clamp(min=1)
    return (s / c).float()


root = '/path/to/casia-webface'
tp = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        if sp == "train":
            tp.append(os.path.join(root, fn))
random.shuffle(tp)
N = 256
cal = tp[:N]
print(f"building envelopes from {N} train-split calibration faces")

imgs = torch.stack([tf_conv(Image.open(p).convert("RGB").resize((112, 112))) for p in cal]).to(device)
with torch.no_grad():
    mo = mface(imgs)[5]
t_minus = norm(highpass(cvt(mo)))

frac_list = []
for p in cal:
    img = Image.open(p).convert("RGB").resize((112, 112))
    o = data2npy.preprocess_and_return(img, 1)[0]
    frac_list.append(o.mean(dim=0, keepdim=True).repeat(3, 1, 1))
t_frac = norm(highpass(cvt(torch.stack(frac_list).to(device))))

env_minus = radial_amp(t_minus)
env_frac = radial_amp(t_frac)
torch.save({"env_minus": env_minus, "env_frac": env_frac}, "proxy_envelopes.pt")
print("env_minus[:10] =", np.round(env_minus[:10].numpy(), 1))
print("env_frac[:10]  =", np.round(env_frac[:10].numpy(), 1))
print("len:", len(env_minus), "saved proxy_envelopes.pt")
