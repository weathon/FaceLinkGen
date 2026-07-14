import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FRACFACE = os.path.join(_HERE, "..", "..", "methods", "fracface")
sys.path.insert(0, _FRACFACE)
import data2npy
sys.path.insert(0, os.path.join(_HERE, "..", "..", "methods", "minusface"))
sys.path.append(os.path.join(_HERE, "..", "partialface"))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchjpeg.dct as tjd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/path/to/casia-webface"
device = "cuda" if torch.cuda.is_available() else "cpu"

from minusface import MinusBackbone
conv = MinusBackbone(mode='stage1')
conv.load_state_dict(torch.load(
    os.path.join(_HERE, "..", "..", "checkpoints", "minusface_stage1.pth"),
    map_location='cpu'))
conv = conv.eval().to(device)


def minus_out(img):
    """img: PIL 112. minusface stage1 output as (112,112) in [0,1]."""
    t = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)[None]
    t = t.mean(1, keepdim=True).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        o = conv(t)[5].float()
    o = o.mean(1, keepdim=True)
    o = (o - o.amin()) / (o.amax() - o.amin() + 1e-6)
    return o[0, 0].cpu().numpy()


def jpeg_dct_highpass(gray, block=8, k=1):
    """Upsample 112 -> 112*block, JPEG block DCT (torchjpeg), zero top-left k x k low-freq
    channels, mean over surviving coefficient channels -> (112,112) edge map. No IDCT."""
    H, W = gray.shape
    im = torch.from_numpy(gray.astype(np.float32))[None, None]
    up = F.interpolate(im, size=(H * block, W * block), mode="bilinear", align_corners=False)
    coeff = tjd.block_dct(tjd.blockify(up, block))
    L = coeff.shape[2]
    c = coeff.reshape(L, block * block).numpy()
    mask = np.ones((block, block), np.float32)
    mask[:k, :k] = 0.0
    c = c[:, mask.ravel() > 0]
    return c.mean(axis=1).reshape(H, W)


def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


val_rel = [l.split()[0] for l in open(os.path.join(_HERE, "..", "..", "data_splits", "index.txt")) if l.split()[1] != "train"]
seen, paths = set(), []
for rel in val_rel:
    ident = rel.split("/")[0]
    if ident not in seen:
        seen.add(ident)
        paths.append(os.path.join(ROOT, rel))
    if len(paths) == 4:
        break

# Compare PROXY (clean image -> our highpass) against the two eval targets:
#   FRAC (raw fracface output) and MINUS->highpass (minus output -> our highpass).
# Show a couple of (block,k) so we can pick what matches both.
B, K = 8, 1   # the proxy config to visually verify
cols = ["orig", "PROXY clean+hp", "FRAC", "MINUS+hp"]
fig, axes = plt.subplots(len(paths), len(cols), figsize=(2.6 * len(cols), 2.6 * len(paths)))

for r, p in enumerate(paths):
    img = Image.open(p).convert("RGB").resize((112, 112))
    gray = (np.asarray(img, np.float32) / 255.0).mean(2)
    mn = minus_out(img)
    panels = [
        np.asarray(img),
        norm01(jpeg_dct_highpass(gray, B, K)),                                   # proxy: clean + hp
        norm01(np.asarray(data2npy.preprocess_and_return(img, 1)[0], np.float32).mean(0)),  # FRAC
        norm01(jpeg_dct_highpass(mn, B, K)),                                     # minus + hp
    ]
    for c, im in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(im) if c == 0 else ax.imshow(im, cmap="gray")
        if r == 0:
            ax.set_title(cols[c], fontsize=10)
        ax.axis("off")

plt.tight_layout()
png = os.path.join(_HERE, "proxy_vs_targets.png")
plt.savefig(png, dpi=90, bbox_inches="tight")
print("saved", png)
