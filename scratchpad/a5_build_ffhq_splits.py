"""FFHQ512 splits. The three uses are mutually exclusive.

gate-val     2000  held out from protection training, used for the cos(orig, protected) gate
attack-train 2000  held out, the distillation pairs; 500/200/100 are prefixes of this list
protect-train rest PerceptFace Stage1 + Stage2

Built over the 224 crops that actually exist (SCRFD found no face in 2659 of the 73098
FFHQ images, so those are not available downstream).
"""
import os
import random

CROPS = '/raid/wg25r/redteam_work/crops/ffhq/224'
OUT = '/raid/wg25r/redteam_work/splits'
SEED = 20260727
N_GATE = 2000
N_ATTACK = 2000

os.makedirs(OUT, exist_ok=True)
names = sorted(os.listdir(CROPS))
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

print('224 crops available :', len(names))
print('gate-val            :', len(gate))
print('attack-train        :', len(attack), '-> prefixes 100/200/500/2000')
print('protect-train       :', len(protect))
print('tipim targets       : 10 (from protect-train)')
