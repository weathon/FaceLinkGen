"""
Smooth frac-matched proxy search. fracface is SMOOTH + faint + broadband, but dctrand is
spiky/sparse. Test SMOOTH-and-HARD candidates (keep frac's character AND difficulty):
  - amplitude transplant to frac's flat envelope (whitening, keeps phase -> smooth)
  - amplitude transplant to FLAT
  - multi-scale DoG highpass
  - strong gaussian highpass
Report init-loss (1-cos to teacher, untrained student) + std/kurt + visual vs frac.
Target: frac init-loss ~1.03, std ~0.905, kurt ~9.5, SMOOTH look.
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
from onnx2torch import convert
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(5)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).eval()
emb_root = '/path/to/casia-webface/insight_embeddings'
ENV = torch.load("proxy_envelopes.pt", map_location='cpu')
ENV_FRAC = ENV["env_frac"].to(device).float()
_yy = (torch.arange(112, device=device).view(-1, 1) - 56); _xx = (torch.arange(112, device=device).view(1, -1) - 56)
_RIDX = torch.sqrt(_yy.float()**2 + _xx.float()**2).round().long().clamp(max=ENV_FRAC.numel() - 1)


def highpass(img, strength=1.0, k=5):
    hp = (img - TF.gaussian_blur(img, (k, k))) * strength
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def cvt(o):
    o = o.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = o.amin(dim=(1, 2, 3), keepdim=True); mx = o.amax(dim=(1, 2, 3), keepdim=True)
    return ((o - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def amp_transplant(img, env, jitter=0.0):
    B, C, H, W = img.shape
    Fi = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
    em = env[_RIDX][None, None]
    if jitter > 0: em = em * (1 + jitter * torch.randn(B, 1, H, W, device=img.device)).clamp(min=0.1)
    out = torch.fft.ifft2(torch.fft.ifftshift(torch.exp(1j * Fi.angle()) * em, dim=(-2, -1))).real
    return out


def dog(img, k1, k2):  # difference of gaussians (smooth bandpass)
    return TF.gaussian_blur(img, (k1, k1)) - TF.gaussian_blur(img, (k2, k2))


root = '/path/to/casia-webface'
paths = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        if sp == "train": paths.append(fn)
random.shuffle(paths); paths = paths[:64]
imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in paths]).to(device)
teacher = F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in paths]).float().to(device), dim=1)
with torch.no_grad():
    frac_raw = torch.stack([data2npy.preprocess_and_return(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112)), 1)[0].mean(0, keepdim=True).repeat(3, 1, 1) for p in paths]).to(device)
face = cvt(imgs)
flat_env = torch.ones_like(ENV_FRAC) * float(ENV_FRAC.median())


def il(inp):
    with torch.no_grad():
        return (1 - (F.normalize(student(inp), dim=1) * teacher).sum(1)).mean().item()


hp5 = highpass(face, 1, 5)
cands = {
    "FRAC (target)": norm(highpass(cvt(frac_raw), 1, 5)),
    "amp->frac": norm(amp_transplant(hp5, ENV_FRAC)),
    "amp->frac jit.3": norm(amp_transplant(hp5, ENV_FRAC, 0.3)),
    "amp->flat": norm(amp_transplant(hp5, flat_env)),
    "amp->flat jit.5": norm(amp_transplant(hp5, flat_env, 0.5)),
    "amp->frac on raw-phase": norm(amp_transplant(face, ENV_FRAC)),
    "dog k3-k9": norm(dog(face, 3, 9)),
    "dog k3-k21": norm(dog(face, 3, 21)),
    "gauss_hp k5 str3": norm(highpass(face, 3, 5)),
    "hp5 then amp->frac": norm(amp_transplant(hp5, ENV_FRAC, 0.0)),
}
print("=== SMOOTH frac-match: init-loss (target frac~1.03) + std(~0.905) + kurt(~9.5) ===")
print(f"{'candidate':28s} {'init':>8s} {'std':>7s} {'kurt':>7s}")
for n, t in cands.items():
    x = t.flatten().double(); z = (x - x.mean()) / (x.std() + 1e-9)
    print(f"{n:28s} {il(t):8.4f} {t.std().item():7.3f} {(z**4).mean().item()-3:7.2f}")


def disp(t):
    a = t[:8]; mn = a.amin(dim=(1,2,3),keepdim=True); mx = a.amax(dim=(1,2,3),keepdim=True)
    return (a - mn) / (mx - mn + 1e-8)
keys = ["FRAC (target)", "amp->frac", "amp->flat jit.5", "dog k3-k21", "amp->frac on raw-phase"]
save_image(torch.cat([disp(cands[k]) for k in keys], 0), "frac_smooth_grid.png", nrow=8)
print("\nsaved frac_smooth_grid.png rows:", keys)
