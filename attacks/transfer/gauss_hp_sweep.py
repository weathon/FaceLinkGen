import os, sys, random, copy
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
from onnx2torch import convert
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(1)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device)
INIT_SD = copy.deepcopy(student.state_dict())
emb_root = '/path/to/casia-webface/insight_embeddings'

KERNELS = [int(x) for x in os.environ.get("KS", "3,5,7,9,15,21").split(",")]
STEPS = int(os.environ.get("STEPS", "400"))


def gray_minmax(t):
    t = t.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True); mx = t.amax(dim=(1, 2, 3), keepdim=True)
    return ((t - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, k):
    hp = img - TF.gaussian_blur(img, (k, k))
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


root = '/path/to/casia-webface'
train_p, val_p = [], []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        (train_p if sp == "train" else val_p).append(fn)
random.shuffle(train_p); random.seed(42); val_p = random.sample(val_p, 150); random.seed(1)


def teach_of(paths):
    return F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in paths]).float().to(device), dim=1)


def load_imgs(paths):
    return torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in paths]).to(device)


val_teach = teach_of(val_p)
val_imgs = load_imgs(val_p)
print("precomputing val minus + frac converted outputs (once)...", flush=True)
with torch.no_grad():
    V_MINUS = gray_minmax(mface(val_imgs)[5])
    frac81 = torch.stack([data2npy.preprocess_and_return(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112)), 1)[0].mean(0, keepdim=True).repeat(3, 1, 1) for p in val_p]).to(device)
    V_FRAC = gray_minmax(frac81)
    FACE = gray_minmax(val_imgs)
tp = train_p[:30000]; bs = 128


def vcos(inp):
    student.eval()
    with torch.no_grad():
        d = (1 - (F.normalize(student(inp), dim=1) * val_teach).sum(1)).mean().item()
    student.train()
    return d


print(f"\nsweep kernels={KERNELS}  steps/each={STEPS}")
print(f"{'k':>3s} | {'best val_minus':>14s} {'best val_frac':>13s} {'val_clean(sanity)':>17s}")
results = {}
for k in KERNELS:
    student.load_state_dict(INIT_SD); student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-3)
    bm, bf = 9, 9
    for step in range(1, STEPS + 1):
        ps = random.sample(tp, bs)
        imgs = load_imgs(ps); teach = teach_of(ps)
        inp = norm(highpass(gray_minmax(imgs), k))
        en = F.normalize(student(inp), dim=1)
        loss = (1 - (en * teach).sum(1)).mean() + F.l1_loss(en, teach)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0:
            vm = vcos(norm(highpass(V_MINUS, k))); vf = vcos(norm(highpass(V_FRAC, k)))
            bm, bf = min(bm, vm), min(bf, vf)
    vc = vcos(norm(highpass(FACE, k)))
    results[k] = (bm, bf, vc)
    print(f"{k:3d} | {bm:14.4f} {bf:13.4f} {vc:17.4f}", flush=True)

print("\n=== summary (best over training; lower=better) ===")
for k, (bm, bf, vc) in results.items():
    print(f"k={k:2d}  val_minus={bm:.4f}  val_frac={bf:.4f}  (clean sanity={vc:.4f})")
