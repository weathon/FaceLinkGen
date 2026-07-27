"""Identity leakage on the validation split, before vs after the distillation attack.

before : cos(Antelopev2(protected), Antelopev2(original))
after  : cos(student(protected),    Antelopev2(original))
upper  : cos(Antelopev2(original),  Antelopev2(original)) = 1 by construction

Verification thresholds come from the impostor distribution inside the same validation
split. FFHQ has one image per identity, so every cross pair is an impostor and there is
no genuine same-identity different-photo pair available. Top-1 is closed-set rank-1
identification: gallery = all validation originals, one correct entry per query, no
distractors and no rejection.

Usage:  python insight_test.py <dir with insight_student_embeddings_val_epoch*.pkl>
        python insight_test.py log             # full-data attack
        python insight_test.py log_lowdata50   # 50-pair attack
The teacher/protected embeddings and the split always come from log/.
"""
import os
import sys
import pickle
import numpy as np

EMB = sys.argv[1]

epochs = sorted(int(f.split('epoch')[1].split('.')[0])
                for f in os.listdir(EMB) if f.startswith('insight_student_embeddings_val_epoch'))
E = epochs[-1]
print('using student embeddings from epoch %d (available: %s)' % (E, epochs))

teacher = pickle.load(open('log/teacher_embeddings_insight.pkl', 'rb'))
protected = pickle.load(open('log/protected_embeddings_insight.pkl', 'rb'))
student = pickle.load(open(EMB + '/insight_student_embeddings_val_epoch%d.pkl' % E, 'rb'))
val_names = pickle.load(open('log/val_paths.pkl', 'rb'))

O = np.stack([teacher[n] for n in val_names])
P = np.stack([protected[n] for n in val_names])
S = np.stack([student[n] for n in val_names])
O /= np.linalg.norm(O, axis=1, keepdims=True)
P /= np.linalg.norm(P, axis=1, keepdims=True)
S /= np.linalg.norm(S, axis=1, keepdims=True)
n = len(val_names)
off = ~np.eye(n, dtype=bool)

imp = (O @ O.T)[off]
thr3 = np.quantile(imp, 1 - 1e-3)
thr4 = np.quantile(imp, 1 - 1e-4)
print('val n = %d, impostor pairs = %d' % (n, off.sum()))
print('impostor cos: mean %.4f  std %.4f   FAR1e-3 thr %.4f   FAR1e-4 thr %.4f'
      % (imp.mean(), imp.std(), thr3, thr4))
print()

hdr = '%-22s %8s %8s %10s %10s %10s' % (
    'setting', 'mean', 'median', 'TAR@1e-3', 'TAR@1e-4', 'Top-1')
print(hdr)
print('-' * len(hdr))

for label, Q in [('before attack', P), ('after attack', S), ('upper bound (orig)', O)]:
    g = (Q * O).sum(1)
    rank1 = ((Q @ O.T).argmax(1) == np.arange(n)).mean()
    print('%-22s %8.4f %8.4f %10.4f %10.4f %10.4f'
          % (label, g.mean(), np.median(g), (g > thr3).mean(), (g > thr4).mean(), rank1))

np.save(EMB + '/val_cos_before.npy', (P * O).sum(1))
np.save(EMB + '/val_cos_after.npy', (S * O).sum(1))
print('\nper-pair cosines saved to %s/val_cos_{before,after}.npy' % EMB)
