"""
3-method LFW eval (minus, frac, partial) under the deployment pipeline, with gaussian
highpass(k5) applied to ALL THREE for consistency (per user). Black-box callables only:
  minus  : MinusBackbone(...)[5]
  frac   : data2npy.preprocess_and_return(img, 1)
  partial: processing_utils.form_training_batch(imgs, block_size, remove_count)   [partialface]
Reports per method: LFW verification 10-fold acc + AUC, and ALIGNMENT = mean cos(student(target),
InsightFace(clean)) -- the generation-relevant metric (emb feeds Arc2Face).

Usage: python eval_three.py [checkpoint.pth]   (default student_best_full_epoch.pth)
"""
import os, sys, random
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
sys.path = [p for p in sys.path if p != "../.."]
sys.path.insert(0, "../../methods/fracface")
import data2npy
sys.path.insert(0, "../partialface")
from processing_utils import form_training_batch   # black-box partial transform
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from onnx2torch import convert
from PIL import Image
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
LFW_ROOT = "/path/to/lfw"
ANN = os.path.join(LFW_ROOT, "lfw_ann.txt")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "student_best_full_epoch.pth"

mface = MinusBackbone(mode='stage1'); mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu')); mface = mface.eval().to(device)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
clean_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)


def to_gray_minmax(out):
    imgs = out.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, k=5):
    hp = img - TF.gaussian_blur(img, (k, k))
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def load_student(ckpt):
    s = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).eval()
    if ckpt and os.path.exists(ckpt):
        s.load_state_dict(torch.load(ckpt, map_location='cpu')); print("loaded", ckpt)
    else:
        print("WARNING: no ckpt, using clean init")
    return s


labels, p1, p2 = [], [], []
with open(ANN) as f:
    for line in f:
        a = line.split(); labels.append(int(a[0])); p1.append(a[1]); p2.append(a[2])
labels = np.array(labels)
allpaths = sorted(set(p1) | set(p2)); idx = {p: i for i, p in enumerate(allpaths)}
print(f"{len(labels)} pairs, {len(allpaths)} unique images; ckpt={CKPT}")

student = load_student(CKPT)


def pil(p):
    return Image.open(os.path.join(LFW_ROOT, p)).convert("RGB").resize((112, 112))


def embed(method, bs=128):
    """method in {clean, minus, frac, partial}. highpass(k5) applied to all degraded methods."""
    embs = []
    for i in tqdm.tqdm(range(0, len(allpaths), bs), desc=method):
        chunk = allpaths[i:i + bs]
        x = torch.stack([tf_conv(pil(p)) for p in chunk]).to(device)
        with torch.no_grad():
            if method == "clean":
                xi = clean_norm(x)
            elif method == "minus":
                xi = norm(highpass(to_gray_minmax(mface(x)[5])))
            elif method == "frac":
                fr = torch.stack([data2npy.preprocess_and_return(pil(p), 1)[0].mean(0, keepdim=True).repeat(3, 1, 1) for p in chunk]).to(device)
                xi = norm(highpass(to_gray_minmax(fr)))
            elif method == "partial":
                # partial transform as black box, EXACTLY as insight_test_partial.py:204-205:
                # inputs, _ = form_training_batch(conv_raw, [1]*B); reshape->mean over freq chans.
                gray3 = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # test feeds grayscale face
                inp, _ = form_training_batch(gray3, [1] * gray3.shape[0])
                inp = inp.reshape(inp.shape[0], -1, 112, 112).mean(1, keepdim=True).repeat(1, 3, 1, 1).float()
                xi = norm(highpass(to_gray_minmax(inp)))
            e = F.normalize(student(xi), dim=1)
        embs.append(e.cpu())
    return torch.cat(embs).numpy()


def verify(embs):
    e1 = embs[[idx[p] for p in p1]]; e2 = embs[[idx[p] for p in p2]]
    sims = (e1 * e2).sum(1)
    order = np.argsort(-sims); ls = labels[order]
    tps = np.cumsum(ls); fps = np.cumsum(1 - ls); P = labels.sum(); N = len(labels) - P
    auc = float(np.trapz(tps / P, fps / N))
    n = len(sims); fold = n // 10; ths = np.linspace(-1, 1, 400); accs = []
    for k in range(10):
        te = np.zeros(n, bool); te[k * fold:(k + 1) * fold] = True; tr = ~te
        best = max(ths, key=lambda t: ((sims[tr] > t) == labels[tr]).mean())
        accs.append(((sims[te] > best) == labels[te]).mean())
    return float(np.mean(accs)), float(np.std(accs)), auc


print("\nembedding clean (alignment reference + sanity)...")
e_clean = embed("clean")
results = {}
for m in ["minus", "frac", "partial"]:
    try:
        E = embed(m)
        acc, sd, auc = verify(E)
        align = float((E * e_clean).sum(1).mean())
        results[m] = (acc, auc, align)
    except Exception as ex:
        print(f"{m}: FAILED {type(ex).__name__}: {ex}")
        results[m] = None

acc, sd, auc = verify(e_clean)
print("\n================ LFW 3-method eval (highpass on all) ================")
print(f"{'clean (sanity)':10s} verif={acc*100:5.2f}%  AUC={auc:.4f}")
for m, r in results.items():
    if r:
        acc, auc, al = r
        print(f"{m:10s} verif={acc*100:5.2f}%  AUC={auc:.4f}  align(cos to clean)={al:+.3f}")
