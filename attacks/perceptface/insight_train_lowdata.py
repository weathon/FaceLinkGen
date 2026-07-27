"""Low-data distillation attack on PerceptFace.

The target argues that limiting how many protected photos are released prevents an
adversary from distilling the mapping. This is that setting: the adversary only ever
sees 50 protected/original pairs.

Same recipe as insight_train.py except:
  - training set = the first 50 images of the train split (FFHQ is one image per
    identity, so 50 images = 50 identities)
  - no LR decay, constant lr 1e-5
  - 5 epochs, batch 16, training set NOT repeated
  -> 3 optimiser steps per epoch (drop_last=True), 15 steps in total
Validation is the same held-out split as the full-data run, so the two are comparable.

Evaluate with:  python insight_test.py log_lowdata50
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from onnx2torch import convert
from PIL import Image
import tqdm
import wandb

ONNX = '../../checkpoints/model.onnx'
PROT = '/path/to/perceptface_work/protected224'
LOG = 'log_lowdata50'
N_TRAIN = 50
EPOCHS = 5
BATCH = 16
device = 'cuda'
os.makedirs(LOG, exist_ok=True)

teacher = pickle.load(open('log/teacher_embeddings_insight.pkl', 'rb'))
train_names = pickle.load(open('log/train_paths.pkl', 'rb'))[:N_TRAIN]
val_names = pickle.load(open('log/val_paths.pkl', 'rb'))
print('train %d val %d' % (len(train_names), len(val_names)), flush=True)

tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])


class Protected(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = tf(Image.open(os.path.join(PROT, n)).convert('RGB'))
        return n, torch.from_numpy(teacher[n]).float(), img


loader = DataLoader(Protected(train_names), batch_size=BATCH, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(Protected(val_names), batch_size=256, shuffle=False,
                        num_workers=16, pin_memory=True)
print('steps per epoch %d, total %d' % (len(loader), len(loader) * EPOCHS), flush=True)

student = convert(ONNX).to(device)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=2e-2)
wandb.init(project='perceptface_distill_lowdata')

for e in range(EPOCHS):
    student.train()
    tot = np.zeros(3)
    count = 0
    for names, t_emb, img in tqdm.tqdm(loader, desc='epoch %d' % e):
        t_emb = t_emb.to(device, non_blocking=True)
        s_emb = student(img.to(device, non_blocking=True))

        s_n = F.normalize(s_emb, dim=1)
        t_n = F.normalize(t_emb, dim=1)
        neg = t_n[torch.roll(torch.arange(s_n.size(0), device=device), shifts=1)]
        trip = torch.clamp((1 - (s_n * t_n).sum(1)) - (1 - (s_n * neg).sum(1)) + 0.3,
                           min=0).mean() * 10
        cos = (1 - (s_n * t_n).sum(1)).mean()
        mae = F.l1_loss(s_n, t_n) * 10
        loss = cos + mae + trip

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tot += [trip.item(), cos.item(), mae.item()]
        count += 1
        wandb.log({'triplet': trip.item(), 'cosine': cos.item(), 'mae': mae.item()})

    student.eval()
    val_emb = {}
    vtot = np.zeros(3)
    vcount = 0
    with torch.no_grad():
        for names, t_emb, img in tqdm.tqdm(val_loader, desc='val %d' % e):
            t_emb = t_emb.to(device, non_blocking=True)
            s_emb = student(img.to(device, non_blocking=True))
            s_n = F.normalize(s_emb, dim=1)
            t_n = F.normalize(t_emb, dim=1)
            neg = t_n[torch.roll(torch.arange(s_n.size(0), device=device), shifts=1)]
            vtot += [torch.clamp((1 - (s_n * t_n).sum(1)) - (1 - (s_n * neg).sum(1)) + 0.3,
                                 min=0).mean().item() * 10,
                     (1 - (s_n * t_n).sum(1)).mean().item(),
                     F.l1_loss(s_n, t_n).item() * 10]
            vcount += 1
            for k, nm in enumerate(names):
                val_emb[nm] = s_emb[k].cpu().numpy().astype(np.float32)
    wandb.log({'epoch': e, 'epoch_cosine': tot[1] / count,
               'val_epoch_cosine': vtot[1] / vcount})
    print('epoch %d  train_cos %.4f  val_cos %.4f  (loss terms, lower is better)'
          % (e, tot[1] / count, vtot[1] / vcount), flush=True)

    with open(LOG + '/insight_student_embeddings_val_epoch%d.pkl' % e, 'wb') as f:
        pickle.dump(val_emb, f)
    torch.save(student.state_dict(), 'student_lowdata50.pth')

print('DONE', flush=True)
