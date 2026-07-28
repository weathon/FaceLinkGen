"""Rebuild every split over the 224-and-128 intersection.

PerceptFace/TIP-IM consume the 224/112 crops, CanFG/CanFG-Ano consume the 128 crops, and
SCRFD and MTCNN fail on different images. Restricting all four methods to the intersection
is what makes the four rows of the result table comparable: same gallery, same identities,
same attack pairs, same K.

LFW : pool = intersection; every identity with >=2 images in the pool gives one query
      (clean -> teacher), the rest of the pool is the gallery (protected -> student).
FFHQ: pool = intersection; gate-val 2000 / attack-train 2000 / protect-train rest, disjoint.

Replaces a4_build_lfw_splits.py and a5_build_ffhq_splits.py.
"""
import os
import random

W = '/raid/wg25r/redteam_work'
OUT = W + '/splits'
LFW = '/raid/wg25r/lfw/lfw-deepfunneled'
SEED = 20260727
N_GATE = 2000
N_ATTACK = 2000

os.makedirs(OUT, exist_ok=True)


def crops(ds, sz):
    return set(os.listdir('%s/crops/%s/%d' % (W, ds, sz)))


# ---------------- LFW ----------------
pool_keys = crops('lfw', 224) & crops('lfw', 128)
# The shuffle and the per-identity choice run over the whole pool, so a different crop
# count silently yields entirely different splits, and this script overwrites splits/.
assert len(pool_keys) == 12273, len(pool_keys)
people = sorted(p for p in os.listdir(LFW) if not p.startswith('.'))
by_id = {}
for p in people:
    kept = sorted(f for f in os.listdir(os.path.join(LFW, p))
                  if os.path.splitext(p + '__' + f)[0] + '.png' in pool_keys)
    if kept:
        by_id[p] = kept

rng = random.Random(SEED)
query, gallery = [], []
for p in sorted(by_id):
    fs = by_id[p]
    if len(fs) >= 2:
        q = rng.choice(fs)
        query.append(p + '/' + q)
        gallery += [p + '/' + f for f in fs if f != q]
    else:
        gallery += [p + '/' + f for f in fs]

assert not (set(gallery) & set(query))
assert {q.split('/')[0] for q in query} <= {g.split('/')[0] for g in gallery}
open(OUT + '/lfw_query.txt', 'w').write('\n'.join(sorted(query)) + '\n')
open(OUT + '/lfw_gallery.txt', 'w').write('\n'.join(sorted(gallery)) + '\n')

print('LFW  pool (224 and 128) :', len(pool_keys))
print('     identities in pool :', len(by_id))
print('     query              :', len(query))
print('     gallery            :', len(gallery),
      '(%d identities)' % len({g.split('/')[0] for g in gallery}))
print('     K = ceil(0.5%%)     : %d' % -(-len(gallery) * 5 // 1000))

# ---------------- FFHQ ----------------
names = sorted(crops('ffhq', 224) & crops('ffhq', 128))
assert len(names) == 68281, len(names)
rng = random.Random(SEED)
shuffled = names[:]
rng.shuffle(shuffled)
gate = sorted(shuffled[:N_GATE])
attack = shuffled[N_GATE:N_GATE + N_ATTACK]
protect = sorted(shuffled[N_GATE + N_ATTACK:])

assert not (set(gate) & set(attack)) and not (set(gate) & set(protect))
assert not (set(attack) & set(protect))
assert len(gate) + len(attack) + len(protect) == len(names)

open(OUT + '/ffhq_gate_val.txt', 'w').write('\n'.join(gate) + '\n')
open(OUT + '/ffhq_protect_train.txt', 'w').write('\n'.join(protect) + '\n')
for n in (100, 200, 500, 2000):
    open(OUT + '/ffhq_attack_%d.txt' % n, 'w').write('\n'.join(attack[:n]) + '\n')
open(OUT + '/tipim_targets.txt', 'w').write('\n'.join(protect[:10]) + '\n')

print('FFHQ pool (224 and 128) :', len(names))
print('     gate-val           :', len(gate))
print('     attack-train       :', len(attack), '-> prefixes 100/200/500/2000')
print('     protect-train      :', len(protect))
