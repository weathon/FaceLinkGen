"""
Check: is minus's low frequency actually removed by highpass(k5)? Visualize minus at each
stage + radial spectrum, compared to frac. If minus still has a low-freq bump after hp(k5),
the highpass is too weak (k5 only removes a narrow low band).
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
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(3)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)


def to_gray_minmax(out):
    imgs = out.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, k=5):
    hp = img - TF.gaussian_blur(img, (k, k))
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def radial(t):
    g = t.mean(dim=1)
    A = torch.fft.fftshift(torch.fft.fft2(g.double()), dim=(-2, -1)).abs().mean(0).cpu()
    H, W = A.shape; cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).round().long()
    s = torch.bincount(r.flatten(), weights=A.flatten(), minlength=int(r.max()) + 1)
    c = torch.bincount(r.flatten(), minlength=int(r.max()) + 1).clamp(min=1)
    return (s / c).numpy()


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
    minus_raw = mface(imgs)[5]
minus_g = to_gray_minmax(minus_raw)                    # minus, grayscale [-1,1], BEFORE highpass
minus_hp5 = highpass(minus_g, 5)                        # current
minus_hp9 = highpass(minus_g, 9)
minus_hp15 = highpass(minus_g, 15)
minus_hp25 = highpass(minus_g, 25)

frac_raw = torch.stack([data2npy.preprocess_and_return(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112)), 1)[0].mean(0, keepdim=True).repeat(3, 1, 1) for p in paths]).to(device)
frac_g = to_gray_minmax(frac_raw)
frac_hp5 = highpass(frac_g, 5)

# visualize stages (display-normalized)
def disp(t):
    a = t[:8]; mn = a.amin(dim=(1,2,3),keepdim=True); mx = a.amax(dim=(1,2,3),keepdim=True)
    return (a - mn) / (mx - mn + 1e-8)
save_image(torch.cat([disp(minus_g), disp(minus_hp5), disp(minus_hp9), disp(minus_hp15), disp(frac_g), disp(frac_hp5)], 0),
           "minus_lowfreq_stages.png", nrow=8)
print("saved minus_lowfreq_stages.png")
print("rows: 1=minus(no hp) 2=minus+hp5[CURRENT] 3=minus+hp9 4=minus+hp15 5=frac(no hp) 6=frac+hp5")

# radial spectra (log), low freq = left
plt.figure(figsize=(10, 6))
for name, t in [("minus (no hp)", minus_g), ("minus+hp5 [current]", minus_hp5),
                ("minus+hp9", minus_hp9), ("minus+hp15", minus_hp15), ("minus+hp25", minus_hp25),
                ("frac (no hp)", frac_g), ("frac+hp5", frac_hp5)]:
    plt.plot(np.log10(radial(t) + 1e-9), label=name)
plt.axvspan(0, 6, alpha=0.1, color='red')  # low-freq band
plt.xlabel("radial freq (left=low). red band = low freq"); plt.ylabel("log10 |F|"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.title("Radial spectrum: is minus low-freq removed by hp(k5)?")
plt.tight_layout(); plt.savefig("minus_lowfreq_radial.png", dpi=100)
print("saved minus_lowfreq_radial.png")

# quantify low-freq energy fraction (freq bins 0-6) after each hp
def lowfrac(t):
    r = radial(t); return float(r[1:7].sum() / (r[1:].sum() + 1e-9))
print("\n=== low-freq energy fraction (bins 1-6 / all AC) ===")
for name, t in [("minus no-hp", minus_g), ("minus+hp5", minus_hp5), ("minus+hp9", minus_hp9),
                ("minus+hp15", minus_hp15), ("minus+hp25", minus_hp25), ("frac no-hp", frac_g), ("frac+hp5", frac_hp5)]:
    print(f"  {name:14s} lowfreq={lowfrac(t)*100:5.2f}%  std={t.std().item():.3f}")
