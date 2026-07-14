"""
Test the REAL old proxy dct_transform (verbatim from insight_train.py.bak:31) as the train proxy.
It's a GENERIC multi-scale block-DCT highpass (block 4/8/16, random low-freq cutoff, mean over kept
channels) -- no minus/frac code. The previous session removed it; likely the regression.
Train student on dct_transform(clean) -> norm -> teacher; val on frac (no-hp & hp) and minus.
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
from torchjpeg import dct
from onnx2torch import convert
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(1)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
student = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).train()
emb_root = '/path/to/casia-webface/insight_embeddings'


def dct_transform(x):   # VERBATIM from insight_train.py.bak (generic block-DCT highpass)
    assert x.shape[1] == 3
    size = random.choice([4, 8, 16]); stride = size; pad = 0; dilation = 1; ratio = size
    x = x * 0.5 + 0.5
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)
    x = x * 255; x = dct.to_ycbcr(x); x = x - 128
    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2).view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)
    chs_remove = list(range(random.randint(1, x_freq.shape[2] - 1)))
    channels = list(set(range(x_freq.shape[2])) - set(chs_remove))
    x_freq = x_freq[:, :, channels, :, :]
    x_freq = x_freq.reshape(b, -1, n_block, n_block).mean(dim=1)
    return x_freq  # (b, 112, 112)


def convert_batch(x, conv):
    x = x.to(device)
    with torch.no_grad():
        out = mface(x)[5] if conv else x
    imgs = out.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def proxy(imgs):  # dct_transform on the [-1,1] grayscale-ish clean face, -> (b,3,112,112) norm
    cf = convert_batch(imgs, False)            # (b,3,112,112) [-1,1]
    d = dct_transform(cf)                       # (b,112,112)
    d = d.unsqueeze(1).repeat(1, 3, 1, 1)
    return norm(d)


def highpass(img, k=5):
    hp = img - TF.gaussian_blur(img, (k, k))
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
val_imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in val_p]).to(device)
val_teach = teach_of(val_p)
frac_raw = torch.stack([data2npy.preprocess_and_return(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112)), 1)[0].mean(0, keepdim=True).repeat(3, 1, 1) for p in val_p]).to(device)
with torch.no_grad():
    Vf = convert_batch(frac_raw, False)
    V_FRAC_nohp = norm(Vf); V_FRAC_hp = norm(highpass(Vf))
    V_MINUS = norm(highpass(convert_batch(val_imgs, True)))


def vc(V):
    student.eval()
    with torch.no_grad():
        d = (1 - (F.normalize(student(V), dim=1) * val_teach).sum(1)).mean().item()
    student.train()
    return d


opt = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-3)
tp = train_p[:20000]
print("=== train: REAL dct_transform(clean) -> teacher; val frac(no-hp/hp) + minus ===")
print(f"step 0    frac_nohp={vc(V_FRAC_nohp):.4f}  frac_hp={vc(V_FRAC_hp):.4f}  minus={vc(V_MINUS):.4f}")
for step in range(1, 801):
    ps = random.sample(tp, 24)
    imgs = torch.stack([tf_conv(Image.open(os.path.join(root, p)).convert("RGB").resize((112, 112))) for p in ps]).to(device)
    teach = teach_of(ps)
    e = F.normalize(student(proxy(imgs)), dim=1)
    loss = (1 - (e * teach).sum(1)).mean() + F.l1_loss(e, teach)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  train_loss={loss.item():.4f}  frac_nohp={vc(V_FRAC_nohp):.4f}  frac_hp={vc(V_FRAC_hp):.4f}  minus={vc(V_MINUS):.4f}")
