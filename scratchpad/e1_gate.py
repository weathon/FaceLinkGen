"""Protection gate, run before any attack. Same shape as methods/perceptface/check_protection.py.

Reports cos(orig, protected) under BOTH families on the held-out FFHQ gate-val split:
  - ArcFace IR-SE50, the protection side (what CanFG/TIP-IM optimise against)
  - AdaFace IR-101 WebFace12M, the attack side
each with an impostor baseline (roll by one) so the number has a scale, plus pixel L1 and
an 8-pair visual panel.

Exits non-zero if either mean genuine cosine is above 0.6. For CanFG / CanFG-Ano this
doubles as the crop-match check: the released checkpoint was trained on CelebA aligned by
CanFG's own MTCNN, so a high cosine here says the alignment did not carry over.

Usage: python e1_gate.py {canfg|canfg_ano|perceptface|tipim}
"""
import os
import sys
import json
import random
import numpy as np
import cv2
import torch

WORK = '/raid/wg25r/redteam_work'
PRE = WORK + '/canfg_premodels/extracted'
sys.path.insert(0, PRE)
sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/scratchpad')
from premodels.irse import Backbone
from adaface_wrap import load_adaface

# The crop the method actually consumed, so pixel L1 compares like with like.
CROPSIZE = {'canfg': 128, 'canfg_ano': 128, 'perceptface': 224, 'tipim': 112}
METHOD = sys.argv[1]
SRC = '%s/crops/ffhq/%d' % (WORK, CROPSIZE[METHOD])
PROT = '%s/protected/%s/ffhq' % (WORK, METHOD)
device = 'cuda'

names = sorted(open(WORK + '/splits/ffhq_gate_val.txt').read().split())
print('gate set: %d images, method=%s, orig crops=%s' % (len(names), METHOD, SRC), flush=True)

arcface = Backbone(50, 0.6, 'ir_se').to(device).eval()
arcface.load_state_dict(torch.load(PRE + '/premodels/model_ir_se50.pth', map_location='cpu'))
arcface.requires_grad_(False)
adaface = load_adaface(device)
adaface.requires_grad_(False)


def embed(root):
    """(arcface, adaface) embeddings for `names` under `root`, both L2-normalised.
    ArcFace IR-SE50 wants RGB 112 in [-1,1]; AdaFace wants BGR 112 in [-1,1]."""
    arc, ada = [], []
    for i in range(0, len(names), 128):
        batch = names[i:i + 128]
        imgs = []
        for n in batch:
            img = cv2.imread(os.path.join(root, n))
            if img is None:
                raise RuntimeError('unreadable: ' + os.path.join(root, n))
            imgs.append(cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR))
        bgr = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(device)
        bgr = (bgr / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            arc.append(torch.nn.functional.normalize(arcface(bgr.flip(1)), dim=1).cpu().numpy())
            ada.append(adaface(bgr)[0].cpu().numpy())
    return np.concatenate(arc), np.concatenate(ada)


arc_o, ada_o = embed(SRC)
arc_p, ada_p = embed(PROT)

res = {'method': METHOD, 'n': len(names)}
for tag, O, P in [('arcface', arc_o, arc_p), ('adaface', ada_o, ada_p)]:
    cos = (O * P).sum(1)
    imp = (np.roll(O, 1, axis=0) * P).sum(1)
    res[tag] = {'mean': float(cos.mean()), 'median': float(np.median(cos)),
                'std': float(cos.std()), 'min': float(cos.min()), 'max': float(cos.max()),
                'pct': [float(x) for x in np.percentile(cos, [1, 5, 25, 75, 95, 99])],
                'impostor_mean': float(imp.mean()), 'impostor_std': float(imp.std()),
                'impostor_p99': float(np.percentile(imp, 99)),
                'frac_above_impostor_p99': float((cos > np.percentile(imp, 99)).mean())}
    print('%-8s cos(orig,protected)  mean %.4f  median %.4f  std %.4f  min %.4f  max %.4f'
          % (tag, cos.mean(), np.median(cos), cos.std(), cos.min(), cos.max()), flush=True)
    print('         percentiles 1%% %.4f  5%% %.4f  25%% %.4f  75%% %.4f  95%% %.4f  99%% %.4f'
          % tuple(np.percentile(cos, [1, 5, 25, 75, 95, 99])))
    print('         impostor mean %.4f  std %.4f  p99 %.4f | genuine above imp-p99: %.4f'
          % (imp.mean(), imp.std(), np.percentile(imp, 99),
             (cos > np.percentile(imp, 99)).mean()))

random.seed(0)
panel = random.sample(names, 8)
rows = []
for i in range(0, 8, 2):
    tiles = []
    for n in panel[i:i + 2]:
        a = cv2.imread(os.path.join(SRC, n))
        b = cv2.imread(os.path.join(PROT, n))
        tiles.append(a)
        tiles.append(cv2.resize(b, a.shape[1::-1], interpolation=cv2.INTER_NEAREST))
    rows.append(cv2.hconcat(tiles))
panelpath = '%s/results/panel_protection_%s.png' % (WORK, METHOD)
os.makedirs(WORK + '/results', exist_ok=True)
cv2.imwrite(panelpath, cv2.vconcat(rows))

pix = []
for n in panel:
    a = cv2.imread(os.path.join(SRC, n)).astype(np.float32) / 255.0
    b = cv2.imread(os.path.join(PROT, n)).astype(np.float32) / 255.0
    pix.append(np.abs(a - cv2.resize(b, a.shape[1::-1])).mean())
res['pixel_l1_panel'] = float(np.mean(pix))
res['panel'] = panel
print('L1 pixel distance on the 8 panel pairs: %.4f' % np.mean(pix))
print('panel -> ' + panelpath)

json.dump(res, open('%s/results/gate_%s.json' % (WORK, METHOD), 'w'), indent=2)
worst = max(res['arcface']['mean'], res['adaface']['mean'])
if worst > 0.6:
    raise SystemExit('GATE FAILED (%s): max mean cosine %.4f > 0.6' % (METHOD, worst))
print('GATE PASSED (%s)' % METHOD)
