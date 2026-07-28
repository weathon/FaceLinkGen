"""Paired distillation attack: protected image -> the original's AdaFace embedding.

Fixed-compute sweep recipe (from /home/wg25r/face_deid/PerceptFace/sweep_data_size_reg.py):
  5000 steps, batch 32, constant lr 1e-5, AdamW weight_decay 2e-2, p=0.5 horizontal flip,
  loss = cos + 10*mae + 10*triplet(margin 0.3) with the in-batch torch.roll(idx, 1)
  negative, cos/mae computed after L2 normalisation.
Every pair count gets the same compute, so the rows are comparable.

Dropout(0.4) before the final Linear is NOT spliced in: AdaFace's Backbone.output_layer
already has it (BatchNorm2d -> Dropout(0.4) -> Flatten -> Linear -> BatchNorm1d), and the
student runs in .train() mode so it is live.

student = AdaFace IR-101 initialised from the pretrained weights, all parameters trainable.
teacher = the same frozen network in eval mode, reading the ORIGINAL 112 crop, so the target
is the same for all four methods regardless of what resolution the method's output is.

Usage: python f1_distill.py {perceptface|canfg|canfg_ano|tipim} {100|200|500|2000}
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
STEPS, BATCH, LR, WD, MARGIN = 5000, 32, 1e-5, 2e-2, 0.3
OUT = '%s/ckpt/distill_%s_%d' % (WORK, METHOD, NPAIRS)
os.makedirs(OUT, exist_ok=True)
device = 'cuda'
torch.manual_seed(0)
random.seed(0)

names = sorted(open('%s/splits/ffhq_attack_%d.txt' % (WORK, NPAIRS)).read().split())
ORIG = WORK + '/crops/ffhq/112'
PROT = '%s/protected/%s/ffhq' % (WORK, METHOD)
print('%s n=%d: %d pairs' % (METHOD, NPAIRS, len(names)), flush=True)

teacher = load_adaface(device)
teacher.requires_grad_(False)

targets = []
for i in range(0, len(names), 128):
    batch = torch.stack([read_112(os.path.join(ORIG, n)) for n in names[i:i + 128]])
    with torch.no_grad():
        targets.append(teacher(batch.to(device))[0])
targets = torch.cat(targets)


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

step = 0
hist = []
if os.path.exists(OUT + '/ckpt.pt'):          # resume state; absent on the first run
    ck = torch.load(OUT + '/ckpt.pt', map_location='cpu', weights_only=False)
    student.load_state_dict(ck['student'])
    opt.load_state_dict(ck['opt'])
    step = ck['step']
    hist = json.load(open(OUT + '/history.json'))   # else the pre-resume curve is overwritten
    print('resumed at step %d' % step, flush=True)
while step < STEPS:
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

        if step % 250 == 0:
            rec = {'step': step, 'loss': loss.item(), 'cos': cos.item(),
                   'mae': mae.item(), 'trip': trip.item()}
            hist.append(rec)
            print('step %d  loss %.4f  cos %.4f  mae %.4f  trip %.4f'
                  % (step, loss.item(), cos.item(), mae.item(), trip.item()), flush=True)
            torch.save({'student': student.state_dict(), 'opt': opt.state_dict(),
                        'step': step}, OUT + '/ckpt.pt')
            json.dump(hist, open(OUT + '/history.json', 'w'))
        if step >= STEPS:
            break

torch.save({'student': student.state_dict(), 'opt': opt.state_dict(), 'step': step},
           OUT + '/ckpt.pt')
print('DONE %s n=%d at step %d' % (METHOD, NPAIRS, step), flush=True)
