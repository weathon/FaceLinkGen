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
    t = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)[None]
    t = t.mean(1, keepdim=True).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        o = conv(t)[5].float()
    o = o.mean(1, keepdim=True)
    o = (o - o.amin()) / (o.amax() - o.amin() + 1e-6)
    return o[0, 0].cpu().numpy()


def jpeg_dct_highpass(gray, block, k):
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
    if len(paths) == 3:
        break

# candidate (block, k) centers + the two targets
cands = [(8, 1), (8, 2), (8, 3), (16, 2), (16, 4)]
cols = ["orig", "FRAC", "MINUS+hp(8,2)"] + [f"proxy b{b}k{k}" for b, k in cands]
fig, axes = plt.subplots(len(paths), len(cols), figsize=(2.3 * len(cols), 2.3 * len(paths)))

for r, p in enumerate(paths):
    img = Image.open(p).convert("RGB").resize((112, 112))
    gray = (np.asarray(img, np.float32) / 255.0).mean(2)
    mn = minus_out(img)
    panels = [np.asarray(img),
              norm01(np.asarray(data2npy.preprocess_and_return(img, 1)[0], np.float32).mean(0)),
              norm01(jpeg_dct_highpass(mn, 8, 2))]
    panels += [norm01(jpeg_dct_highpass(gray, b, k)) for b, k in cands]
    for c, im in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(im) if c == 0 else ax.imshow(im, cmap="gray")
        if r == 0:
            ax.set_title(cols[c], fontsize=8)
        ax.axis("off")

plt.tight_layout()
png = os.path.join(_HERE, "pick_center.png")
plt.savefig(png, dpi=90, bbox_inches="tight")
print("saved", png)
