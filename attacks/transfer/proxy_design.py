"""
Proxy design: build candidate proxy transforms (NO minus/frac code) and score how well
they match the two targets' post-norm statistics:
  - radial amplitude envelope shape  (flatness)
  - 8x8 block-boundary spike
  - kurtosis (sparsity)

Candidates use only standard signal ops: gaussian highpass, FFT amplitude transplant
(spectral whitening), and block-DCT quantization/whitening (torchjpeg dct as math).

Caches target tensors to cal_cache.pt so reruns are fast.
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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from torchjpeg import dct
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


def highpass(img, strength=1.0, kernel_size=5):
    blurred = TF.gaussian_blur(img, (kernel_size, kernel_size))
    hp = (img - blurred) * strength
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(imgs, eps=1e-6, clamp_val=5.0):
    mean = imgs.mean(dim=(1, 2, 3), keepdim=True)
    std = imgs.std(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mean) / (std + eps)).clamp(-clamp_val, clamp_val)


def convert_false(conv_raw):
    imgs = conv_raw.to(device).float()
    imgs = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    return (imgs - 0.5) / 0.5


# ---------- block DCT helpers (torchjpeg dct as math primitive) ----------
def block_dct(img, block=8):
    B, C, H, W = img.shape
    nh, nw = H // block, W // block
    x = img.view(B, C, nh, block, nw, block).permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(B, C, nh * nw, block, block)
    return dct.block_dct(x), (B, C, nh, nw)


def block_idct(x_dct, meta, block=8):
    B, C, nh, nw = meta
    x = dct.block_idct(x_dct).view(B, C, nh, nw, block, block)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, nh * block, nw * block)
    return x


def jpeg_quant(img, q_frac=1.5, block=8):
    """Adaptive JPEG-like quant: block DCT -> quantize each AC frequency to steps of
    (q_frac * that-frequency's std across blocks) -> iDCT. Scale-invariant; q_frac larger
    => coarser => stronger 8x8 block boundaries + sparsity. DC left intact."""
    xd, meta = block_dct(img, block)
    s = xd.std(dim=2, keepdim=True).clamp(min=1e-8)   # (B,C,1,8,8) per-freq AC std
    step = q_frac * s
    xq = torch.round(xd / step) * step
    xq[:, :, :, 0, 0] = xd[:, :, :, 0, 0]             # keep DC
    return block_idct(xq, meta, block)


def dct_whiten(img, block=8, eps=1e-3, keep_dc=True):
    """Per-image DCT whitening: flatten each of the 64 frequency bands to unit variance
    across the blocks of that image -> flat (white) spectrum + 8x8 block structure."""
    xd, meta = block_dct(img, block)          # (B,C,nblk,8,8)
    s = xd.std(dim=2, keepdim=True)           # std across blocks per freq
    xw = xd / (s + eps)
    if keep_dc:
        xw[:, :, :, 0, 0] = xd[:, :, :, 0, 0]  # leave DC alone (it's removed by norm anyway)
    return block_idct(xw, meta, block)


def amp_transplant(img, radial_env, jitter=0.0):
    """Keep FFT phase of img, replace radial amplitude with radial_env (a 1D envelope).
    Spectral whitening / coloring to a target envelope."""
    B, C, H, W = img.shape
    Fimg = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=img.device).view(-1, 1) - cy
    xx = torch.arange(W, device=img.device).view(1, -1) - cx
    r = torch.sqrt(yy.float() ** 2 + xx.float() ** 2).round().long().clamp(max=len(radial_env) - 1)
    env = radial_env.to(img.device)[r]                       # (H,W)
    if jitter > 0:
        env = env * (1 + jitter * torch.randn_like(env))
    amp = env[None, None]
    out = torch.fft.ifft2(torch.fft.ifftshift(torch.exp(1j * Fimg.angle()) * amp, dim=(-2, -1))).real
    return out


# ---------- metrics ----------
def avg_amp2d(t):
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


def block_spike(t, block=8):
    g = t.mean(dim=1)
    d = (g[:, :, 1:] - g[:, :, :-1]).abs()
    cols = torch.arange(d.shape[-1])
    prof = np.array([d[:, :, (cols % block) == m].mean().item() for m in range(block)])
    return prof[0] / (prof[1:].mean() + 1e-9)


def kurt(t):
    x = t.flatten().double()
    z = (x - x.mean()) / (x.std() + 1e-9)
    return (z ** 4).mean().item() - 3.0


def shape_dist(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    la = np.log10(a / (a.max() + 1e-12) + 1e-9)
    lb = np.log10(b / (b.max() + 1e-12) + 1e-9)
    return float(np.linalg.norm(la - lb))


# ---------- load / cache targets ----------
if os.path.exists("cal_cache.pt"):
    cache = torch.load("cal_cache.pt")
    t_minus, t_frac, face_conv = cache["t_minus"].to(device), cache["t_frac"].to(device), cache["face_conv"].to(device)
    print("loaded cal_cache.pt", t_minus.shape)
else:
    mface = MinusBackbone(mode='stage1')
    mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
    mface = mface.eval().to(device)
    root = '/path/to/casia-webface'
    tp = []
    with open("../../data_splits/index.txt") as f:
        for line in f:
            fn, sp = line.strip().split()
            if sp == "train":
                tp.append(os.path.join(root, fn))
    random.shuffle(tp)
    cal = tp[:96]
    imgs = torch.stack([tf_conv(Image.open(p).convert("RGB").resize((112, 112))) for p in cal]).to(device)
    with torch.no_grad():
        mo = mface(imgs)[5]

    def cvt(o):
        o = o.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        mn = o.amin(dim=(1, 2, 3), keepdim=True); mx = o.amax(dim=(1, 2, 3), keepdim=True)
        return ((o - mn) / (mx - mn + 1e-6) - 0.5) / 0.5
    t_minus = norm(highpass(cvt(mo), 1.0, 5))
    frac_list = []
    for p in cal:
        img = Image.open(p).convert("RGB").resize((112, 112))
        o = data2npy.preprocess_and_return(img, 1)[0]
        frac_list.append(o.mean(dim=0, keepdim=True).repeat(3, 1, 1))
    frac_raw = torch.stack(frac_list).to(device)
    t_frac = norm(highpass(cvt(frac_raw), 1.0, 5))
    face_conv = convert_false(imgs)
    torch.save({"t_minus": t_minus.cpu(), "t_frac": t_frac.cpu(), "face_conv": face_conv.cpu()}, "cal_cache.pt")
    print("computed + cached targets")

rad_minus = radial_from_2d(avg_amp2d(t_minus))
rad_frac = radial_from_2d(avg_amp2d(t_frac))
env_minus = torch.tensor(rad_minus, dtype=torch.float32)
env_frac = torch.tensor(rad_frac, dtype=torch.float32)

# Parametric "fair" envelope: 1/f^a tilt * mid-band plateau, fit only by eye to the flat
# broadband shape (NOT the exact target numbers). 3 knobs: low rolloff, plateau, high rolloff.
freqs = torch.arange(len(env_minus), dtype=torch.float32)
def param_env(plateau_lo=6, plateau_hi=52, low_p=1.0, hi_p=1.2):
    e = torch.ones_like(freqs)
    lo = freqs < plateau_lo
    e[lo] = (freqs[lo] / plateau_lo).clamp(min=0.05) ** low_p     # rise into plateau
    hi = freqs > plateau_hi
    e[hi] = ((len(freqs) - freqs[hi]) / (len(freqs) - plateau_hi)).clamp(min=0.02) ** hi_p
    return e
env_param = param_env()

hp5 = highpass(face_conv, 1.0, 5)

# ---------- candidates ----------
cands = {
    "gauss_hp_k5":         norm(hp5),
    "amp->minus":          norm(amp_transplant(hp5, env_minus)),
    "amp->frac":           norm(amp_transplant(hp5, env_frac)),
    "amp->param":          norm(amp_transplant(hp5, env_param)),
    "jpeg_qf1.5":          norm(jpeg_quant(hp5, q_frac=1.5)),
    "jpeg_qf3":            norm(jpeg_quant(hp5, q_frac=3.0)),
    "dct_whiten":          norm(dct_whiten(hp5)),
    "amp->minus+jpeg1.5":  norm(jpeg_quant(amp_transplant(hp5, env_minus), q_frac=1.5)),
    "amp->minus+jpeg3":    norm(jpeg_quant(amp_transplant(hp5, env_minus), q_frac=3.0)),
    "amp->param+jpeg2":    norm(jpeg_quant(amp_transplant(hp5, env_param), q_frac=2.0)),
    "amp->frac+jpeg1.5":   norm(jpeg_quant(amp_transplant(hp5, env_frac), q_frac=1.5)),
}

print(f"\n{'TARGET minus':18s} blockspike=1.47 kurt=+15.3")
print(f"{'TARGET frac':18s} blockspike=1.03 kurt=+9.0")
print("\n%-20s %8s %8s %10s %10s" % ("cand", "bspike", "kurt", "d_minus", "d_frac"))
for name, t in cands.items():
    r = radial_from_2d(avg_amp2d(t))
    print("%-20s %8.3f %8.2f %10.3f %10.3f" % (
        name, block_spike(t), kurt(t), shape_dist(rad_minus, r), shape_dist(rad_frac, r)))

# save sample grids for the most promising
def disp(t):
    a = t[:8]
    mn = a.amin(dim=(1,2,3),keepdim=True); mx = a.amax(dim=(1,2,3),keepdim=True)
    return (a - mn) / (mx - mn + 1e-8)
for name in ["amp->minus", "amp->param", "jpeg_qf3", "amp->minus+jpeg3", "amp->param+jpeg2"]:
    save_image(disp(cands[name]), f"pd_{name.replace('>','').replace('-','').replace('+','_')}.png", nrow=4)
print("\nsaved pd_*.png sample grids")
