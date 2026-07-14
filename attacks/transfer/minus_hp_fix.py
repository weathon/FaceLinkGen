"""
minus+hp5 still leaves ~12.5% low-freq (vs frac+hp5 0.5%). gaussian img-blur with bigger k
makes it WORSE. Test STRONGER true-highpass options to actually kill minus low-freq:
  - hp5 applied twice / thrice
  - FFT hard highpass (cutoff radius) -- truly zeros low band
  - DoG / unsharp variants
Report low-freq fraction after each, target ~frac's 0.5-1%. The winning one is what val_minus
(and the deliverable) should use for the minus post-process.
"""
import os, sys, random
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(3)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)


def to_gray_minmax(out):
    imgs = out.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def hp(img, k=5):
    h = img - TF.gaussian_blur(img, (k, k))
    return (h - h.min()) / (h.max() - h.min() + 1e-8) / 2 + 0.5


def fft_hp(img, cutoff=4):
    H, W = img.shape[-2:]
    Fi = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
    yy = torch.arange(H, device=img.device).view(-1, 1) - H // 2
    xx = torch.arange(W, device=img.device).view(1, -1) - W // 2
    r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2)
    mask = (r > cutoff).float()[None, None]
    out = torch.fft.ifft2(torch.fft.ifftshift(Fi * mask, dim=(-2, -1))).real
    mn = out.amin(dim=(1, 2, 3), keepdim=True); mx = out.amax(dim=(1, 2, 3), keepdim=True)
    return (out - mn) / (mx - mn + 1e-8) / 2 + 0.5


def radial(t):
    g = t.mean(dim=1)
    A = torch.fft.fftshift(torch.fft.fft2(g.double()), dim=(-2, -1)).abs().mean(0).cpu()
    H, W = A.shape
    yy, xx = torch.meshgrid(torch.arange(H) - H // 2, torch.arange(W) - W // 2, indexing='ij')
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).round().long()
    s = torch.bincount(r.flatten(), weights=A.flatten(), minlength=int(r.max()) + 1)
    c = torch.bincount(r.flatten(), minlength=int(r.max()) + 1).clamp(min=1)
    return (s / c).numpy()


def lowfrac(t):
    r = radial(t); return float(r[1:7].sum() / (r[1:].sum() + 1e-9))


root = '/path/to/casia-webface'
paths = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        if sp != "train":
            paths.append(fn)
random.seed(42); paths = random.sample(paths, 32)
imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in paths]).to(device)
with torch.no_grad():
    mg = to_gray_minmax(mface(imgs)[5])

cands = {
    "hp5 [current]": hp(mg, 5),
    "hp5 x2": hp(hp(mg, 5), 5),
    "hp5 x3": hp(hp(hp(mg, 5), 5), 5),
    "hp3 x2": hp(hp(mg, 3), 3),
    "fft_hp c2": fft_hp(mg, 2),
    "fft_hp c4": fft_hp(mg, 4),
    "fft_hp c6": fft_hp(mg, 6),
    "fft_hp c8": fft_hp(mg, 8),
    "hp5 then fft c4": fft_hp(hp(mg, 5), 4),
}
print("=== low-freq fraction after each (target ~frac's 0.5%) ===")
for n, t in cands.items():
    print(f"  {n:18s} lowfreq={lowfrac(t)*100:5.2f}%  std={t.std().item():.3f}")


def disp(t):
    a = t[:8]; mn = a.amin(dim=(1,2,3),keepdim=True); mx = a.amax(dim=(1,2,3),keepdim=True)
    return (a - mn) / (mx - mn + 1e-8)
save_image(torch.cat([disp(cands[k]) for k in ["hp5 [current]", "hp5 x2", "fft_hp c4", "fft_hp c8", "hp5 then fft c4"]], 0),
           "minus_hp_fix.png", nrow=8)
print("saved minus_hp_fix.png rows: hp5, hp5x2, fft_c4, fft_c8, hp5+fft_c4")
