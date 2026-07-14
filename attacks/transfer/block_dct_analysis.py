"""
Per-block 8x8 DCT coefficient profile of the minus target vs our dct_highpass proxy.
We've matched the RADIAL spectrum (failed). This matches the finer 8x8 DCT energy map:
which of the 64 block-frequencies carry energy in highpass(minus(face)) vs dct_highpass(face).
If they differ, we can shape the proxy's per-block coefficient gains to match -> a more
minus-faithful block proxy. Fair: analysis of black-box OUTPUT statistics only.
Saves block_dct_profile.pt (8x8 target gain map) for use in training.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
sys.path = [p for p in sys.path if p != "../.."]
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchjpeg import dct
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(7)
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


def block_dct(img, block=8):
    B, C, H, W = img.shape
    nh, nw = H // block, W // block
    x = img.view(B, C, nh, block, nw, block).permute(0, 1, 2, 4, 3, 5).contiguous()
    return dct.block_dct(x.view(B, C, nh * nw, block, block))  # (B,C,nblk,8,8)


def coeff_profile(t, block=8):
    """mean |DCT coeff| per of the 64 block-frequencies -> (8,8). t is post-norm (B,3,H,W)."""
    g = t.mean(dim=1, keepdim=True)  # luminance
    xd = block_dct(g, block)         # (B,1,nblk,8,8)
    return xd.abs().mean(dim=(0, 1, 2)).cpu()  # (8,8)


def dct_highpass(img, n_zero=2, block=8):
    B, C, H, W = img.shape
    nh, nw = H // block, W // block
    x = img.view(B, C, nh, block, nw, block).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, nh * nw, block, block)
    xd = dct.block_dct(x)
    xd[..., :n_zero, :n_zero] = 0
    xb = dct.block_idct(xd).view(B, C, nh, nw, block, block).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    mn = xb.amin(dim=(1, 2, 3), keepdim=True); mx = xb.amax(dim=(1, 2, 3), keepdim=True)
    return (xb - mn) / (mx - mn + 1e-8) / 2 + 0.5


root = '/path/to/casia-webface'
tp = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        if sp == "train":
            tp.append(os.path.join(root, fn))
random.shuffle(tp)
cal = tp[:128]
imgs = torch.stack([tf_conv(Image.open(p).convert("RGB").resize((112, 112))) for p in cal]).to(device)
with torch.no_grad():
    mo = mface(imgs)[5]

t_minus = norm(highpass(cvt(mo)))                 # target: highpass(minus) -> norm
face_g = cvt(imgs)                                 # grayscale clean face [-1,1]
p_dcthp = norm(dct_highpass(face_g, n_zero=2))     # proxy: dct_highpass(clean)
p_plainhp = norm(highpass(face_g))                 # proxy: gaussian highpass(clean) (hponly)

prof_minus = coeff_profile(t_minus)
prof_dcthp = coeff_profile(p_dcthp)
prof_plain = coeff_profile(p_plainhp)

np.set_printoptions(precision=2, suppress=True)
print("=== mean |8x8 DCT coeff| profile (row=vert freq, col=horiz freq; [0,0]=DC) ===")
print("\nTARGET highpass(minus):\n", prof_minus.numpy())
print("\nPROXY dct_highpass(clean):\n", prof_dcthp.numpy())
print("\nPROXY gaussian-hp(clean):\n", prof_plain.numpy())

# normalized shapes (sum=1) to compare distribution over frequencies
def shp(p):
    p = p.clone(); p[0, 0] = 0  # ignore DC (removed by norm)
    return p / (p.sum() + 1e-9)
sm, sd, sp = shp(prof_minus), shp(prof_dcthp), shp(prof_plain)
print("\nL1 distance of freq-distribution to target:")
print(f"  dct_highpass : {(sm - sd).abs().sum().item():.4f}")
print(f"  gaussian-hp  : {(sm - sp).abs().sum().item():.4f}")

# target per-frequency GAIN to apply to a dct_highpass proxy so its profile matches minus
gain = (shp(prof_minus) / (shp(prof_dcthp) + 1e-6)).clamp(0, 5)
gain[0, 0] = 0
print("\nper-frequency gain (minus/dct_highpass), to shape proxy block-DCT:\n", gain.numpy())
torch.save({"prof_minus": prof_minus, "prof_dcthp": prof_dcthp, "gain8x8": gain}, "block_dct_profile.pt")
print("\nsaved block_dct_profile.pt")
