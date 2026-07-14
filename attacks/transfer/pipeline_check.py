"""
Pipeline sanity / bug hunt (per user: check for non-essential bugs — number range, processing
mismatch, dimension mixup — recalling that plain gaussian highpass on BOTH train and val worked).

Decisive test: train student on highpass(CLEAN face) -> teacher; val on (a) highpass(CLEAN val
face) and (b) highpass(MINUS val face), same everything else.
  - If clean-val -> LOW and minus-val -> HIGH: pipeline is bug-free; the minus domain is the real gap.
  - If clean-val is ALSO high: there is a non-essential bug.
Also prints range/shape of train vs val tensors and asserts train-path == val-path on identical input.
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
random.seed(1)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).train()
emb_root = '/path/to/casia-webface/insight_embeddings'


def convert_batch(x, conv):
    x = x.to(device)
    with torch.no_grad():
        out = mface(x)[5] if conv else x
    imgs = out.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, k=5):
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


def load(paths):
    imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in paths]).to(device)
    teach = F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in paths]).float().to(device), dim=1)
    return imgs, teach


val_imgs, val_teach = load(val_p)
# precompute val inputs
with torch.no_grad():
    V_CLEAN = convert_batch(val_imgs, False)   # clean grayscale [-1,1]
    V_MINUS = convert_batch(val_imgs, True)    # minus output

# ---- BUG CHECKS ----
print("=== shapes / ranges ===")
tr0 = norm(highpass(convert_batch(val_imgs[:8], False)))
print(f"train-path input  shape={tuple(tr0.shape)} range=[{tr0.min():.2f},{tr0.max():.2f}] mean={tr0.mean():.3f} std={tr0.std():.3f}")
vm = norm(highpass(V_MINUS[:8]))
print(f"val-minus input   shape={tuple(vm.shape)} range=[{vm.min():.2f},{vm.max():.2f}] mean={vm.mean():.3f} std={vm.std():.3f}")
# processing equivalence: same clean input through train-path vs val-clean-path
a = norm(highpass(convert_batch(val_imgs[:8], False)))
b = norm(highpass(V_CLEAN[:8]))
print(f"train-path == val-clean-path on identical input?  max|diff|={ (a-b).abs().max().item():.2e}  (should be ~0)")

# ---- decisive train: highpass(CLEAN) -> teacher ; val on clean & minus ----
opt = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-3)
bs = 128
train_paths = train_p[:20000]
train_imgs_cache = None


def get_train_batch():
    ps = random.sample(train_paths, bs)
    imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in ps]).to(device)
    teach = F.normalize(torch.stack([torch.from_numpy(np.load(emb_root + "/" + p.replace("/", "_").replace(".jpg", ".npy"))) for p in ps]).float().to(device), dim=1)
    return imgs, teach


def val_cos(V):
    student.eval()
    with torch.no_grad():
        e = F.normalize(student(norm(highpass(V))), dim=1)
        d = (1 - (e * val_teach).sum(1)).mean().item()
    student.train()
    return d


print("\n=== train on highpass(CLEAN faces) -> teacher; val on clean-hp AND minus-hp ===")
print(f"step 0    val_clean={val_cos(V_CLEAN):.4f}  val_minus={val_cos(V_MINUS):.4f}")
for step in range(1, 501):
    imgs, teach = get_train_batch()
    inp = norm(highpass(convert_batch(imgs, False)))   # plain gaussian highpass of clean
    e = student(inp)
    en = F.normalize(e, dim=1)
    loss = (1 - (en * teach).sum(1)).mean() + F.l1_loss(en, teach)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  train_loss={loss.item():.4f}  val_clean={val_cos(V_CLEAN):.4f}  val_minus={val_cos(V_MINUS):.4f}")
