"""
LFW eval under the deployment pipeline (minus -> highpass(k5) -> zscore -> student).
Reports TWO metrics per checkpoint:
  - VERIFICATION: standard LFW 10-fold accuracy + AUC (separability of same/diff pairs).
  - ALIGNMENT  : mean cosine( student(minus(x)) , InsightFace(clean x) ). This is the
    GENERATION-relevant metric: Arc2Face is conditioned on real InsightFace embeddings, so the
    student emb must ALIGN to clean InsightFace's space. (LFW analog of val_cosine; align=1-dist.)

Usage: python lfw_eval.py [ckpt1.pth ckpt2.pth ...]   (default: student_deliverable.pth)
New standalone script; uses MinusBackbone as a black-box callable; does not edit other files.
"""
import os, sys
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
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
LFW_ROOT = "/path/to/lfw"
ANN = os.path.join(LFW_ROOT, "lfw_ann.txt")
CKPTS = sys.argv[1:] or ["student_deliverable.pth"]

mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)
tf_conv = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
clean_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)


def convert_batch(x, convert=True):
    x = x.to(device)
    with torch.no_grad():
        out = mface(x)[5] if convert else x
    imgs = out.float().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = imgs.amin(dim=(1, 2, 3), keepdim=True); mx = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return ((imgs - mn) / (mx - mn + 1e-6) - 0.5) / 0.5


def highpass(img, k=5):
    hp = img - TF.gaussian_blur(img, (k, k))
    return (hp - hp.min()) / (hp.max() - hp.min() + 1e-8) / 2 + 0.5


def norm(x, eps=1e-6, c=5.0):
    return ((x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + eps)).clamp(-c, c)


def load_student(ckpt=None):
    s = torch.nn.Sequential(convert("../../checkpoints/model.onnx")).to(device).eval()
    if ckpt:
        s.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return s


labels, p1, p2 = [], [], []
with open(ANN) as f:
    for line in f:
        a = line.split(); labels.append(int(a[0])); p1.append(a[1]); p2.append(a[2])
labels = np.array(labels)
allpaths = sorted(set(p1) | set(p2)); idx = {p: i for i, p in enumerate(allpaths)}
print(f"{len(labels)} pairs, {len(allpaths)} unique images; checkpoints: {CKPTS}")


def load_batch(paths):
    return torch.stack([tf_conv(Image.open(os.path.join(LFW_ROOT, p)).convert("RGB").resize((112, 112))) for p in paths])


student_clean = load_student(None)
ckpt_students = {os.path.basename(c).replace("student_best_", "").replace(".pth", ""): load_student(c) for c in CKPTS}

# (1) clean teacher embeddings + clean-LFW sanity
e_clean, e_noadapt = [], []
ckpt_embs = {k: [] for k in ckpt_students}
bs = 200
for i in tqdm.tqdm(range(0, len(allpaths), bs), desc="embed"):
    x = load_batch(allpaths[i:i + bs]).to(device)
    with torch.no_grad():
        e_clean.append(F.normalize(student_clean(clean_norm(x)), dim=1).cpu())
        xi = norm(highpass(convert_batch(x, convert=True)))      # minus deployment input (shared)
        e_noadapt.append(F.normalize(student_clean(xi), dim=1).cpu())
        for k, st in ckpt_students.items():
            ckpt_embs[k].append(F.normalize(st(xi), dim=1).cpu())
e_clean = torch.cat(e_clean).numpy()
e_noadapt = torch.cat(e_noadapt).numpy()
ckpt_embs = {k: torch.cat(v).numpy() for k, v in ckpt_embs.items()}


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


def align(embs):
    """mean cosine(student(minus x), InsightFace(clean x)) — generation-relevant."""
    return float((embs * e_clean).sum(1).mean())


print("\n================ LFW (6000 pairs) ================")
acc, sd, auc = verify(e_clean)
print(f"{'clean / CLEAN LFW [sanity]':36s} verif={acc*100:5.2f}%  AUC={auc:.4f}")
acc, sd, auc = verify(e_noadapt)
print(f"{'clean-init / LFW-minus [no adapt]':36s} verif={acc*100:5.2f}%  AUC={auc:.4f}  align(cos)={align(e_noadapt):+.3f}")
for k, E in ckpt_embs.items():
    acc, sd, auc = verify(E)
    al = align(E)
    print(f"{k[:36]:36s} verif={acc*100:5.2f}%  AUC={auc:.4f}  align(cos)={al:+.3f}  (val-style dist={1-al:.3f})")
