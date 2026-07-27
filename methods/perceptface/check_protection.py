"""Sanity check on the generated pairs, run before the attack.

Reports the Antelopev2 cosine between each original crop and its protected version, plus
an impostor baseline (protected_i vs original_j, j shifted by one) so the number has a
scale, and dumps an 8-pair visual panel.

Exits non-zero if the mean genuine cosine is above 0.6, which would mean the
preprocessing is wrong and the pairs are not worth attacking.

Reads the pickles written by ../../attacks/perceptface/extract_embeddings.py.
"""
import os
import pickle
import random
import numpy as np
import cv2

CROPS = '../../data/perceptface/crops224'
PROT = '../../data/perceptface/protected224'
LOG = '../../attacks/perceptface/log'

teacher = pickle.load(open(LOG + '/teacher_embeddings_insight.pkl', 'rb'))
protected = pickle.load(open(LOG + '/protected_embeddings_insight.pkl', 'rb'))
names = sorted(teacher)

O = np.stack([teacher[n] for n in names])
P = np.stack([protected[n] for n in names])
O /= np.linalg.norm(O, axis=1, keepdims=True)
P /= np.linalg.norm(P, axis=1, keepdims=True)

cos = (O * P).sum(1)
imp = (np.roll(O, 1, axis=0) * P).sum(1)

print('n = %d' % len(names))
print('cos(orig, protected)   mean %.4f  median %.4f  std %.4f  min %.4f  max %.4f'
      % (cos.mean(), np.median(cos), cos.std(), cos.min(), cos.max()))
print('  percentiles  1%% %.4f  5%% %.4f  25%% %.4f  75%% %.4f  95%% %.4f  99%% %.4f'
      % tuple(np.percentile(cos, [1, 5, 25, 75, 95, 99])))
print('impostor cos(orig_j, protected_i)  mean %.4f  std %.4f  99%% %.4f'
      % (imp.mean(), imp.std(), np.percentile(imp, 99)))
print('fraction of genuine pairs above impostor 99th pct: %.4f'
      % (cos > np.percentile(imp, 99)).mean())

random.seed(0)
panel_names = random.sample(names, 8)
rows = []
for i in range(0, 8, 2):
    tiles = []
    for n in panel_names[i:i + 2]:
        tiles.append(cv2.imread(os.path.join(CROPS, n)))
        tiles.append(cv2.imread(os.path.join(PROT, n)))
    rows.append(cv2.hconcat(tiles))
cv2.imwrite('panel_protection.png', cv2.vconcat(rows))
print('panel -> panel_protection.png')
print('panel names: ' + ' '.join(panel_names))

pix = []
for n in panel_names:
    a = cv2.imread(os.path.join(CROPS, n)).astype(np.float32) / 255.0
    b = cv2.imread(os.path.join(PROT, n)).astype(np.float32) / 255.0
    pix.append(np.abs(a - b).mean())
print('L1 pixel distance on the 8 panel pairs: %.4f' % float(np.mean(pix)))

if cos.mean() > 0.6:
    raise SystemExit('GATE FAILED: mean cosine %.4f > 0.6' % cos.mean())
print('GATE PASSED')
