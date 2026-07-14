"""
Eval-only: does removing minus's residual low-freq (hp5 -> hp5x2/x3/fft) help val_minus?
Load the trained best checkpoint, recompute val_minus cosine under different minus post-process.
Fast: no retraining. If hp5x2 lowers val_minus, the deliverable should use it for minus.
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
from onnx2torch import convert
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "student_best_full_epoch.pth"
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
emb_root = '/path/to/casia-webface/insight_embeddings'
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).eval()
student.load_state_dict(torch.load(CKPT, map_location='cpu')); print("loaded", CKPT)


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
    out = torch.fft.ifft2(torch.fft.ifftshift(Fi * (r > cutoff).float()[None, None], dim=(-2, -1))).real
    mn = out.amin(dim=(1, 2, 3), keepdim=True); mx = out.amax(dim=(1, 2, 3), keepdim=True)
    return (out - mn) / (mx - mn + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


root = '/path/to/casia-webface'
vp = []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        if sp != "train":
            vp.append(fn)
random.seed(42); vp = random.sample(vp, 150)
imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in vp]).to(device)
teach = F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in vp]).float().to(device), dim=1)
with torch.no_grad():
    mg = to_gray_minmax(mface(imgs)[5])


def valcos(post):
    with torch.no_grad():
        e = F.normalize(student(norm(post(mg))), dim=1)
    return (1 - (e * teach).sum(1)).mean().item()


print("=== val_minus cosine under different minus post-process (current trained ckpt) ===")
for name, post in [("hp5 [current]", lambda x: hp(x, 5)),
                   ("hp5 x2", lambda x: hp(hp(x, 5), 5)),
                   ("hp5 x3", lambda x: hp(hp(hp(x, 5), 5), 5)),
                   ("hp3 x2", lambda x: hp(hp(x, 3), 3)),
                   ("fft_hp c4", lambda x: fft_hp(x, 4)),
                   ("fft_hp c6", lambda x: fft_hp(x, 6)),
                   ("hp5 then fft c4", lambda x: fft_hp(hp(x, 5), 4))]:
    print(f"  {name:18s} val_minus={valcos(post):.4f}")
