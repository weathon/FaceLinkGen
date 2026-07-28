"""Same distillation attack as f1, but trained to convergence instead of a fixed 5000 steps.

Identical recipe otherwise -- batch 32, constant lr 1e-5, AdamW weight_decay 2e-2,
p=0.5 horizontal flip, loss = cos + 10*mae + 10*triplet(0.3), same teacher targets --
so the only difference from the f1 table is the step count.

Early stopping on a held-out set: FFHQ gate-val (2000 images, disjoint from attack-train
and from protect-train). Every 500 steps it measures
    val_cos = mean cos( student(protected_val), teacher(original_val) )
and stops after 10 consecutive evaluations with no new best (i.e. 5000 steps of no
improvement), or at 50000 steps. The checkpoint kept is the best-val one, not the last.

Usage: python f2_distill_converge.py {perceptface|canfg|canfg_ano|tipim} {100|200|500|2000}
"""
import os
import sys
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/scratchpad')
from adaface_wrap import load_adaface, read_112

WORK = '/raid/wg25r/redteam_work'
METHOD, NPAIRS = sys.argv[1], int(sys.argv[2])
BATCH, LR, WD, MARGIN = 32, 1e-5, 2e-2, 0.3
EVAL_EVERY, PATIENCE, MAX_STEPS = 500, 10, 50000
OUT = '%s/ckpt/converge_%s_%d' % (WORK, METHOD, NPAIRS)
os.makedirs(OUT, exist_ok=True)
if os.path.exists(OUT + '/summary.json'):
    # Resuming a finished config would re-enter the loop with bad >= PATIENCE already
    # true, train another EVAL_EVERY steps and rewrite summary.json.
    print('already finished: ' + open(OUT + '/summary.json').read(), flush=True)
    raise SystemExit(0)
device = 'cuda'
torch.manual_seed(0)
random.seed(0)

names = sorted(open('%s/splits/ffhq_attack_%d.txt' % (WORK, NPAIRS)).read().split())
val_names = sorted(open(WORK + '/splits/ffhq_gate_val.txt').read().split())
ORIG = WORK + '/crops/ffhq/112'
PROT = '%s/protected/%s/ffhq' % (WORK, METHOD)
print('%s n=%d: %d train pairs, %d val' % (METHOD, NPAIRS, len(names), len(val_names)),
      flush=True)

teacher = load_adaface(device)
teacher.requires_grad_(False)


def teacher_emb(root, ns):
    out = []
    for i in range(0, len(ns), 128):
        x = torch.stack([read_112(os.path.join(root, n)) for n in ns[i:i + 128]])
        with torch.no_grad():
            out.append(teacher(x.to(device))[0])
    return torch.cat(out)


targets = teacher_emb(ORIG, names)
val_target = F.normalize(teacher_emb(ORIG, val_names), dim=1)
val_prot = torch.stack([read_112(os.path.join(PROT, n)) for n in val_names])


class Pairs(Dataset):
    def __init__(self):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        x = read_112(os.path.join(PROT, self.names[i]))
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, i


loader = DataLoader(Pairs(), batch_size=BATCH, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=len(names) > BATCH, persistent_workers=True)

student = load_adaface(device).train()
opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)

step, best, best_step, bad = 0, -2.0, 0, 0
hist = []
if os.path.exists(OUT + '/last.pt'):          # resume state; absent on the first run
    ck = torch.load(OUT + '/last.pt', map_location='cpu', weights_only=False)
    student.load_state_dict(ck['student'])
    opt.load_state_dict(ck['opt'])
    step, best, best_step, bad = ck['step'], ck['best'], ck['best_step'], ck['bad']
    hist = json.load(open(OUT + '/history.json'))
    print('resumed at step %d (best %.4f @ %d, bad %d)' % (step, best, best_step, bad),
          flush=True)

stop = False
while not stop:
    for x, idx in loader:
        s = student(x.to(device, non_blocking=True))[0]
        t = targets[idx.to(device)]
        s_n = F.normalize(s, dim=1)
        t_n = F.normalize(t, dim=1)
        neg = t_n[torch.roll(torch.arange(s_n.size(0), device=device), shifts=1)]
        cos = (1 - (s_n * t_n).sum(1)).mean()
        mae = F.l1_loss(s_n, t_n) * 10
        trip = torch.clamp((1 - (s_n * t_n).sum(1)) - (1 - (s_n * neg).sum(1)) + MARGIN,
                           min=0).mean() * 10
        loss = cos + mae + trip

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1

        if step % EVAL_EVERY == 0:
            student.eval()
            vc = []
            with torch.no_grad():
                for i in range(0, len(val_names), 256):
                    v = student(val_prot[i:i + 256].to(device))[0]
                    vc.append((F.normalize(v, dim=1) * val_target[i:i + 256]).sum(1))
            val_cos = float(torch.cat(vc).mean())
            student.train()

            if val_cos > best:
                best, best_step, bad = val_cos, step, 0
                torch.save({'student': student.state_dict(), 'step': step,
                            'val_cos': val_cos}, OUT + '/best.pt')
            else:
                bad += 1
            hist.append({'step': step, 'loss': loss.item(), 'cos': cos.item(),
                         'val_cos': val_cos, 'best': best, 'bad': bad})
            print('step %d  loss %.4f  val_cos %.4f  best %.4f @ %d  bad %d/%d'
                  % (step, loss.item(), val_cos, best, best_step, bad, PATIENCE), flush=True)
            torch.save({'student': student.state_dict(), 'opt': opt.state_dict(),
                        'step': step, 'best': best, 'best_step': best_step, 'bad': bad},
                       OUT + '/last.pt')
            json.dump(hist, open(OUT + '/history.json', 'w'))

            if bad >= PATIENCE or step >= MAX_STEPS:
                stop = True
                break

why = 'patience' if bad >= PATIENCE else 'max_steps'
json.dump({'method': METHOD, 'n_pairs': NPAIRS, 'stopped_at': step, 'reason': why,
           'best_val_cos': best, 'best_step': best_step},
          open(OUT + '/summary.json', 'w'), indent=2)
print('DONE %s n=%d stopped at %d (%s), best val_cos %.4f @ step %d'
      % (METHOD, NPAIRS, step, why, best, best_step), flush=True)
