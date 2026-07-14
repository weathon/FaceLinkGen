"""
Test minimal DCT-based highpass: block DCT -> zero low-freq coeffs -> iDCT.
This is a STANDARD signal processing technique (not from minusface/fracface code).
Will visualize and compute PSD to compare against minus_hp(k=5) and frac_norm.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from torchjpeg import dct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
tf_conv = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)


def dct_highpass(img, block=8, n_zero=3):
    """Block DCT (8x8) -> zero top-left n_zero x n_zero low-frequency coefficients -> iDCT.
    A standard DCT-based highpass filter. img: (B, 3, 112, 112) in any range.
    Returns same shape, same range as input (approximately, modulo numerical error).
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    assert H % block == 0 and W % block == 0
    # Reshape into blocks
    x = img.view(B, C, H // block, block, W // block, block)
    # (B, C, nh, block, nw, block) -> (B, C, nh, nw, block, block)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    nh, nw = H // block, W // block
    # Flatten block to (..., block*block) and apply torchjpeg block_dct on (..., block, block)
    # torchjpeg.dct.block_dct expects shape (..., 8, 8).
    x_dct = dct.block_dct(x)  # (B, C, nh, nw, block, block)
    # Zero out top-left n_zero x n_zero coefficients per block
    x_dct[..., :n_zero, :n_zero] = 0
    # Inverse
    x_back = dct.block_idct(x_dct)  # (B, C, nh, nw, block, block)
    x_back = x_back.permute(0, 1, 2, 4, 3, 5).contiguous()  # (B, C, nh, block, nw, block)
    x_back = x_back.view(B, C, H, W)
    return x_back


def to_grayscale_minmax(t):
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True)
    mx = t.amax(dim=(1, 2, 3), keepdim=True)
    t = (t - mn) / (mx - mn + 1e-6)
    return (t - 0.5) / 0.5


def gauss_hp(img, k=5):
    return img - TF.gaussian_blur(img, (k, k))


def radial_psd(t):
    gray = t.mean(dim=1)
    F_img = torch.fft.fftshift(torch.fft.fft2(gray.double()), dim=(-2, -1))
    psd = (F_img.abs() ** 2)
    H, W = gray.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).int()
    rmax = r.max().item()
    res = []
    for p in psd:
        w = p.flatten().cpu().double().numpy()
        i = r.flatten().cpu().numpy()
        s = np.bincount(i, weights=w, minlength=rmax + 1)
        c = np.bincount(i, minlength=rmax + 1)
        res.append(s / np.maximum(c, 1))
    return np.stack(res).mean(axis=0)


root = '/path/to/casia-webface'
paths = []
for d in sorted(os.listdir(root))[:16]:
    if os.path.isdir(os.path.join(root, d)):
        for f in sorted(os.listdir(os.path.join(root, d)))[:1]:
            paths.append(os.path.join(root, d, f))

imgs = torch.stack([tf_conv(Image.open(p).convert("RGB").resize((112, 112))) for p in paths])
imgs = imgs.to(device) * 2 - 1
imgs_norm = to_grayscale_minmax(imgs)

# Reference: minusface + hp(k=5)
with torch.no_grad():
    minus_out = mface(imgs)[5]
minus_norm = to_grayscale_minmax(minus_out)
minus_hp = gauss_hp(minus_norm, k=5)

# Candidates
cands = {
    "minus_hp_k5_ref": minus_hp,
    "dct_hp_n1": dct_highpass(imgs_norm, block=8, n_zero=1),
    "dct_hp_n2": dct_highpass(imgs_norm, block=8, n_zero=2),
    "dct_hp_n3": dct_highpass(imgs_norm, block=8, n_zero=3),
    "dct_hp_n4": dct_highpass(imgs_norm, block=8, n_zero=4),
    "gauss_hp_k5": gauss_hp(imgs_norm, k=5),
}


def to_disp(t):
    t = t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True)
    mx = t.amax(dim=(1, 2, 3), keepdim=True)
    return (t - mn) / (mx - mn + 1e-6)


for name, t in cands.items():
    save_image(to_disp(t[:8]), f"dct_{name}.png", nrow=4)

plt.figure(figsize=(10, 6))
for name, t in cands.items():
    psd = radial_psd(t)
    plt.plot(np.log10(psd + 1e-12), label=name)
plt.xlabel("radial freq")
plt.ylabel("log10 power")
plt.title("DCT highpass vs minus_hp_k5 reference")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dct_hp_psd.png", dpi=100)

ref_psd = radial_psd(cands["minus_hp_k5_ref"])
print("=== Distance from minus_hp_k5 reference ===")
for name, t in cands.items():
    if name == "minus_hp_k5_ref":
        continue
    p = radial_psd(t)
    rp = ref_psd / (ref_psd.max() + 1e-12)
    cp = p / (p.max() + 1e-12)
    dist = np.linalg.norm(np.log10(rp + 1e-12) - np.log10(cp + 1e-12))
    print(f"{name:20s}  l2_log_psd={dist:.3f}  std={t.std().item():.4f}  range=[{t.min():.3f}, {t.max():.3f}]")
print("Saved dct_*.png and dct_hp_psd.png")
