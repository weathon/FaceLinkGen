"""Measure what actually landed on disk before building any split.

Reports FFHQ512 entry count / extensions / sizes / modes, and LFW identity count,
images-per-identity distribution, sizes / modes. Nothing here writes a split; the
split script is written only after these numbers are known.
"""
import os
import collections
import random
from PIL import Image

FFHQ = '/raid/wg25r/ffhq512_hf/images/FFHQ512/FFHQ512'
LFW = '/raid/wg25r/lfw/lfw-deepfunneled'

print('=' * 70)
print('FFHQ512', FFHQ)
entries = sorted(os.listdir(FFHQ))
print('  top-level entries:', len(entries))
print('  first 5          :', entries[:5])
ext = collections.Counter(os.path.splitext(e)[1] for e in entries)
print('  extensions       :', ext.most_common())

imgs = [e for e in entries if os.path.splitext(e)[1].lower() in ('.png', '.jpg', '.jpeg')]
print('  image files      :', len(imgs))
random.seed(0)
c = collections.Counter()
for f in random.sample(imgs, 300):
    im = Image.open(os.path.join(FFHQ, f))
    c[(im.size, im.mode)] += 1
print('  sampled 300 (size, mode):', c.most_common())

print('=' * 70)
print('LFW', LFW)
raw = sorted(os.listdir(LFW))
people = [p for p in raw if not p.startswith('.')]   # zip ships .DS_Store / __MACOSX
print('  raw entries      :', len(raw), '| dropped:', [p for p in raw if p.startswith('.')])
print('  identity dirs    :', len(people))
counts = {p: len(os.listdir(os.path.join(LFW, p))) for p in people}
total = sum(counts.values())
dist = collections.Counter(counts.values())
multi = [p for p, n in counts.items() if n >= 2]
print('  total images     :', total)
print('  identities >=2   :', len(multi))
print('  images of those  :', sum(counts[p] for p in multi))
print('  per-identity count -> #identities (top 12):', sorted(dist.items())[:12])
print('  max images/identity:', max(counts.values()),
      '->', max(counts, key=counts.get))

c = collections.Counter()
flat = [(p, f) for p in people for f in os.listdir(os.path.join(LFW, p))]
for p, f in random.sample(flat, 300):
    im = Image.open(os.path.join(LFW, p, f))
    c[(im.size, im.mode)] += 1
print('  sampled 300 (size, mode):', c.most_common())

gallery = total - len(multi)
K = -(-gallery * 5 // 1000)
print('=' * 70)
print('planned  query   =', len(multi))
print('planned  gallery =', gallery)
print('planned  K (0.5%) =', K)
