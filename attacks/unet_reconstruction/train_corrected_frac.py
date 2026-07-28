"""Train the U-Net reconstruction attack against the CORRECTED FracFace implementation.

Mirrors main_frac.ipynb exactly (same architecture, optimizer, schedule, loss, splits) so
the only variable is released-code vs corrected-code protection.
"""

import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, "../fracface/FracFace")
import corrected_fracface as cf

MODE = os.environ.get("PROTECT_MODE", "corrected")
TAG = "" if MODE == "corrected" else "_" + MODE
device = "cuda"
ROOT = "/path/to/casia-webface"
INDEX = "../fracface/index.txt"

tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


class FaceDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = cv2.imread(p)[..., ::-1]
        img = Image.fromarray(img.astype("uint8"))
        raw = tf_conv(img)                     # [3,112,112] in [0,1]
        if MODE == "freqmajor":
            prot = cf.protect_freqmajor(raw.unsqueeze(0))[0]
        else:
            prot = cf.protect(raw.unsqueeze(0))[0]  # fresh secret M0/L0 per image
        return p, prot, raw


def load_split(split):
    out = []
    with open(INDEX) as f:
        for line in f:
            fn, s = line.strip().split()
            if (s == "train") == (split == "train"):
                out.append(os.path.join(ROOT, fn))
    return out


def main():
    model = torch.hub.load("mateuszbuda/brain-segmentation-pytorch", "unet",
                           in_channels=81, out_channels=3, init_features=3,
                           pretrained=False).to(device)

    train_ds, val_ds = FaceDataset(load_split("train")), FaceDataset(load_split("val"))
    print("train %d  val %d" % (len(train_ds), len(val_ds)), flush=True)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

    epochs = 10
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    sched = CosineAnnealingLR(opt, T_max=len(loader) * epochs)

    for e in range(epochs):
        model.train()
        tot = n = 0
        for _, prot, raw in tqdm.tqdm(loader, desc="epoch %d" % e):
            raw, prot = raw.to(device), prot.to(device)
            loss = torch.nn.functional.l1_loss(model(prot), raw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        torch.save(model.state_dict(), "model_frac_corrected%s.pth" % TAG)

        model.eval()
        vt = vn = 0
        with torch.no_grad():
            for _, prot, raw in tqdm.tqdm(val_loader, desc="val %d" % e):
                raw, prot = raw.to(device), prot.to(device)
                vt += torch.nn.functional.l1_loss(model(prot), raw).item()
                vn += 1
        print("epoch %d  train_l1 %.4f  val_l1 %.4f" % (e, tot / n, vt / vn), flush=True)

    # Dump val reconstructions for metric comparison.
    model.eval()
    gens, fns = [], []
    with torch.no_grad():
        for fn, prot, _ in tqdm.tqdm(val_loader, desc="dump"):
            gens.append(model(prot.to(device)).detach().cpu())
            fns.append(fn)
    import pickle
    with open("frac_web_corrected%s.pkl" % TAG, "wb") as f:
        pickle.dump({"filenames": fns, "gen_images": gens}, f)
    print("wrote frac_web_corrected%s.pkl" % TAG, flush=True)


if __name__ == "__main__":
    main()
