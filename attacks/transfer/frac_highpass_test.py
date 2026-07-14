"""
Reproduce "train: gaussian highpass, val: fracface (worked before)" and find the regression.
HANDOFF note: val_frac ORIGINALLY fed fracface output directly (NO highpass); highpass(k5) was
ADDED later "for symmetry". FracFace output is itself a high-freq residual, so the added highpass
may DOUBLE-highpass it -> mismatch with the single-highpass train proxy.

Train student on gaussian highpass(clean) -> teacher. Val on fracface:
  (A) frac -> highpass(k5) -> norm   [CURRENT]
  (B) frac -> norm                   [OLD, no extra highpass]
  (C) frac -> highpass(k5,strength)  sweep
Also init-loss + visual. If (B) << (A), the added highpass is the regression.
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
random.seed(1)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).train()
emb_root = '/path/to/casia-webface/insight_embeddings'


def convert_false(x):
    x = x.to(device).float()
    imgs = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, strength=1.0, k=5):
    hp = (img - TF.gaussian_blur(img, (k, k))) * strength
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def teach_of(paths):
    return F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in paths]).float().to(device), dim=1)


root = '/path/to/casia-webface'
train_p, val_p = [], []
with open("../../data_splits/index.txt") as f:
    for line in f:
        fn, sp = line.strip().split()
        (train_p if sp == "train" else val_p).append(fn)
random.shuffle(train_p)
random.seed(42); val_p = random.sample(val_p, 150); random.seed(1)

# val fracface raw outputs (mean of 81 channels), as the dataset builds them
val_imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in val_p]).to(device)
val_teach = teach_of(val_p)
frac_raw = []
for p in val_p:
    o = data2npy.preprocess_and_return(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112)), 1)[0]
    frac_raw.append(o.mean(0, keepdim=True).repeat(3, 1, 1))
frac_raw = torch.stack(frac_raw).to(device)

V_frac_cf = convert_false(frac_raw)         # frac -> grayscale/minmax/[-1,1]
V_A = norm(highpass(V_frac_cf))             # CURRENT: + highpass(k5)
V_B = norm(V_frac_cf)                        # OLD: no highpass
V_C9 = norm(highpass(V_frac_cf, 1, 9))


def val_cos(V):
    student.eval()
    with torch.no_grad():
        d = (1 - (F.normalize(student(V), dim=1) * val_teach).sum(1)).mean().item()
    student.train()
    return d


# init-loss (untrained) for context
print("=== init-loss (untrained student) ===")
print(f"frac + highpass(k5) [CURRENT] : {val_cos(V_A):.4f}")
print(f"frac, NO highpass   [OLD]     : {val_cos(V_B):.4f}")
print(f"frac + highpass(k9)           : {val_cos(V_C9):.4f}")
save_image(torch.cat([(V_A[:8]-V_A[:8].amin(dim=(1,2,3),keepdim=True))/(V_A[:8].amax(dim=(1,2,3),keepdim=True)-V_A[:8].amin(dim=(1,2,3),keepdim=True)+1e-8),
                      (V_B[:8]-V_B[:8].amin(dim=(1,2,3),keepdim=True))/(V_B[:8].amax(dim=(1,2,3),keepdim=True)-V_B[:8].amin(dim=(1,2,3),keepdim=True)+1e-8)], 0),
           "frac_hp_vs_nohp.png", nrow=8)

# train on gaussian highpass(clean) -> teacher
opt = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-3)
tp = train_p[:20000]
print("\n=== train: gaussian highpass(clean) -> teacher; val on frac (A) hp vs (B) no-hp ===")
print(f"step 0    val_A(hp)={val_cos(V_A):.4f}  val_B(nohp)={val_cos(V_B):.4f}")
for step in range(1, 601):
    ps = random.sample(tp, 128)
    imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in ps]).to(device)
    teach = teach_of(ps)
    inp = norm(highpass(convert_false(imgs)))    # train proxy = plain gaussian highpass
    e = F.normalize(student(inp), dim=1)
    loss = (1 - (e * teach).sum(1)).mean() + F.l1_loss(e, teach)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  train_loss={loss.item():.4f}  val_A(frac+hp)={val_cos(V_A):.4f}  val_B(frac,NOhp)={val_cos(V_B):.4f}")
print("\nsaved frac_hp_vs_nohp.png (top: frac+highpass [CURRENT], bottom: frac no-highpass [OLD])")
