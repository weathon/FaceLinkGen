"""LFW gallery / query split.

Every identity with >=2 images contributes exactly one query image (clean -> teacher);
that image is removed from the gallery so rank-1 cannot trivially match itself.
Everything else in LFW goes into the gallery (protected -> student).

Measured on this disk: 5749 identities, 13233 images, 1680 identities with >=2 images
-> query 1680, gallery 11553.
"""
import os
import random

LFW = '/raid/wg25r/lfw/lfw-deepfunneled'
OUT = '/raid/wg25r/redteam_work/splits'
SEED = 20260727

os.makedirs(OUT, exist_ok=True)
people = sorted(p for p in os.listdir(LFW) if not p.startswith('.'))
imgs = {p: sorted(os.listdir(os.path.join(LFW, p))) for p in people}

rng = random.Random(SEED)
query = []
for p in people:
    if len(imgs[p]) >= 2:
        query.append(p + '/' + rng.choice(imgs[p]))
qset = set(query)
gallery = [p + '/' + f for p in people for f in imgs[p] if p + '/' + f not in qset]

assert not (set(gallery) & qset)
qids = {q.split('/')[0] for q in query}
gids = {g.split('/')[0] for g in gallery}
assert qids <= gids

open(OUT + '/lfw_query.txt', 'w').write('\n'.join(sorted(query)) + '\n')
open(OUT + '/lfw_gallery.txt', 'w').write('\n'.join(sorted(gallery)) + '\n')

print('identities        :', len(people))
print('total images      :', sum(len(v) for v in imgs.values()))
print('query             :', len(query), '(identities:', len(qids), ')')
print('gallery           :', len(gallery), '(identities:', len(gids), ')')
print('K = ceil(0.5% gal):', -(-len(gallery) * 5 // 1000))
print('overlap           :', len(set(gallery) & qset))
