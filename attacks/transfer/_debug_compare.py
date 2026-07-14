import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.append("../../methods/fracface")
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

device = "cuda"

conversion_model = MinusBackbone(mode='stage1')
conversion_model.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
conversion_model = conversion_model.eval().to(device)

tf_conv = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])

def convert_batch(conv_raw, convert=True):
    conv_raw = conv_raw.to(device)
    with torch.no_grad():
        if convert:
            out = conversion_model(conv_raw)[5]
        else:
            out = conv_raw
    imgs = out.float()
    imgs = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    imgs = (imgs - 0.5) / 0.5
    return imgs

def norm(imgs):
    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    return imgs

def highpass(img, strength=1.0, kernel_size=5, sigma=None):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    blurred = TF.gaussian_blur(img, (kernel_size, kernel_size))
    highpass_img = img - blurred
    highpass_img = highpass_img * strength
    highpass_img = (highpass_img - highpass_img.min()) / (highpass_img.max() - highpass_img.min() + 1e-8) / 2 + 0.5
    return highpass_img

# load some paths
root = '/path/to/casia-webface'
paths = []
with open("../../data_splits/index.txt","r") as f:
    for line in f.readlines():
        filename, split = line.strip().split()
        paths.append(os.path.join(root, filename))
random.seed(0)
sample = random.sample(paths, 16)

raws = []
for p in sample:
    img_s = Image.open(p).convert("RGB").resize((112,112))
    raws.append(tf_conv(img_s))
raws = torch.stack(raws)

# ---- TRAIN pipeline ----
train_imgs = []
for c_img in raws:
    hp_strength = random.uniform(0.3, 10)
    c = highpass(c_img, strength=hp_strength, kernel_size=random.randint(1,3)*2+1, sigma=None)
    c = c.mean(dim=0, keepdim=True).repeat(3,1,1)
    train_imgs.append(c)
train_imgs = torch.stack(train_imgs).to(device)
train_s = convert_batch(train_imgs, convert=False)
train_s = norm(train_s)

# ---- VAL pipeline ----
val_c = convert_batch(raws, convert=True)
val_c = highpass(val_c, strength=1, kernel_size=21)
val_s = norm(val_c)

def stats(name, x):
    print(f"{name:12s} shape={tuple(x.shape)} min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} std={x.std():.3f}")

stats("train_s", train_s)
stats("val_s", val_s)

# per-sample mean/std spread
print("train per-sample mean:", train_s.mean(dim=(1,2,3)).cpu().numpy().round(3))
print("val   per-sample mean:", val_s.mean(dim=(1,2,3)).cpu().numpy().round(3))
print("train per-sample std :", train_s.std(dim=(1,2,3)).cpu().numpy().round(3))
print("val   per-sample std :", val_s.std(dim=(1,2,3)).cpu().numpy().round(3))

# save side-by-side images
from torchvision.utils import save_image
save_image(train_s, "_debug_train.png", nrow=4)
save_image(val_s, "_debug_val.png", nrow=4)
print("saved _debug_train.png and _debug_val.png")
