"""
Frequency analysis: compare radial power spectrum of
  - minusface output (post conversion model, post mean over channels)
  - fracface output (data2npy.preprocess_and_return, post mean over 81 ch)
  - simple gaussian highpass of raw face
  - our train proxy (current insight_train_minus.py settings)
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
sys.path = [p for p in sys.path if p != "../.."]
sys.path.insert(0, "../../methods/fracface")
import data2npy
import random, torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load minusface
mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)

tf_conv = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])


def _auto_kernel_from_sigma(sigma):
    return int(2 * round(3 * sigma) + 1)


def highpass(img, strength=1.0, kernel_size=5, sigma=None):
    if sigma is None:
        blurred = TF.gaussian_blur(img, (kernel_size, kernel_size))
    else:
        blurred = TF.gaussian_blur(img, (kernel_size, kernel_size), sigma=[sigma, sigma])
    hp = img - blurred
    hp = hp * strength
    hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5
    return hp


def to_grayscale_minmax(t):
    """3-ch tensor -> mean over channels, minmax to [-1, 1]. Mirrors convert_batch."""
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True)
    mx = t.amax(dim=(1, 2, 3), keepdim=True)
    t = (t - mn) / (mx - mn + 1e-6)
    return (t - 0.5) / 0.5  # [-1, 1]


def radial_power_spectrum(img):
    """img: 3-ch tensor [-1, 1] in (3, H, W) or (B, 3, H, W). Returns radial PSD."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    gray = img.mean(dim=1)  # (B, H, W)
    B, H, W = gray.shape
    F_img = torch.fft.fftshift(torch.fft.fft2(gray.double()), dim=(-2, -1))
    psd = (F_img.abs() ** 2)  # (B, H, W)
    # radial average
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    r = torch.sqrt(yy ** 2 + xx ** 2).int()
    rmax = r.max().item()
    out = []
    for batch_psd in psd:
        # bincount weighted by psd
        weights = batch_psd.flatten().cpu().double().numpy()
        idx = r.flatten().cpu().numpy()
        sums = np.bincount(idx, weights=weights, minlength=rmax + 1)
        counts = np.bincount(idx, minlength=rmax + 1)
        radial = sums / np.maximum(counts, 1)
        out.append(radial)
    return np.stack(out)  # (B, rmax+1)


def collect_samples(N=64):
    root = '/path/to/casia-webface'
    paths = []
    for d in sorted(os.listdir(root))[:N + 5]:
        if os.path.isdir(os.path.join(root, d)):
            for f in sorted(os.listdir(os.path.join(root, d)))[:1]:
                paths.append(os.path.join(root, d, f))
                if len(paths) >= N:
                    break
        if len(paths) >= N:
            break
    return paths


paths = collect_samples(32)
print(f"Loaded {len(paths)} face paths")

# Convert to batched tensors
imgs = []
for p in paths:
    img = Image.open(p).convert("RGB").resize((112, 112))
    imgs.append(tf_conv(img))
faces = torch.stack(imgs).to(device)  # (B, 3, 112, 112) in [0, 1]
print(f"faces shape: {faces.shape}, range: [{faces.min():.3f}, {faces.max():.3f}]")

# 1. Minusface output -> mean -> minmax
with torch.no_grad():
    minus_out = mface(faces)[5]  # (B, 3, 112, 112)
minus_norm = to_grayscale_minmax(minus_out)
print(f"minus_norm range: [{minus_norm.min():.3f}, {minus_norm.max():.3f}]")

# 2. Minusface output + highpass(k=5, s=1)
minus_hp5 = highpass(minus_norm, strength=1, kernel_size=5)
# 3. Minusface output + highpass(k=21)
minus_hp21 = highpass(minus_norm, strength=1, kernel_size=21)

# 4. Fracface output -> mean -> minmax
frac_outs = []
for p in paths:
    img = Image.open(p).convert("RGB").resize((112, 112))
    o = data2npy.preprocess_and_return(img, 1)[0]  # (81, 112, 112)
    frac_outs.append(o.mean(dim=0, keepdim=True).repeat(3, 1, 1))
frac_tensor = torch.stack(frac_outs).to(device)
frac_norm = to_grayscale_minmax(frac_tensor)
print(f"frac_norm range: [{frac_norm.min():.3f}, {frac_norm.max():.3f}]")

# 5. Simple highpass on raw face (k=5)
face_norm = to_grayscale_minmax(faces)
simple_hp5 = highpass(face_norm, strength=1, kernel_size=5)

# 6. Simple highpass on raw face (k=21)
simple_hp21 = highpass(face_norm, strength=1, kernel_size=21)

# 7. Highpass twice
hp5_twice = highpass(simple_hp5, strength=1, kernel_size=5)

# Compute radial PSD for each
candidates = {
    "minus_norm": minus_norm,
    "minus+hp(k=5)": minus_hp5,
    "minus+hp(k=21)": minus_hp21,
    "frac_norm": frac_norm,
    "simple_hp(k=5)": simple_hp5,
    "simple_hp(k=21)": simple_hp21,
    "simple_hp(k=5)_twice": hp5_twice,
}

# Save sanity images
for name, t in candidates.items():
    safe = name.replace("(", "_").replace(")", "").replace("+", "_").replace("=", "")
    img_save = (t[:8] * 0.5 + 0.5).clamp(0, 1)
    save_image(img_save, f"freq_sample_{safe}.png", nrow=4)

# Compute and plot
plt.figure(figsize=(10, 6))
for name, t in candidates.items():
    psd = radial_power_spectrum(t.float())
    mean_psd = psd.mean(axis=0)  # avg over batch
    # log scale, normalize so all start at 0 (relative to DC)
    log_psd = np.log10(mean_psd + 1e-12)
    # plot vs radial index
    plt.plot(np.arange(len(log_psd)), log_psd, label=name)
plt.xlabel("radial frequency (px^-1, fftshift)")
plt.ylabel("log10 power")
plt.title("Radial power spectrum")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("freq_psd_comparison.png", dpi=100)

# Print summary stats
print("\n=== Summary ===")
for name, t in candidates.items():
    psd = radial_power_spectrum(t.float()).mean(axis=0)
    log_psd = np.log10(psd + 1e-12)
    print(f"{name:30s}  total_power={psd.sum():.3e}  peak_freq={log_psd.argmax()}  std={t.std().item():.3f}")
print("\nSaved freq_psd_comparison.png and freq_sample_*.png")
