"""Distillation attack on PerceptFace.

Student = onnx2torch(Antelopev2 glintr100), all parameters trainable.
Input   = the PROTECTED image, Resize(112) + ToTensor (i.e. [0,1]).
Target  = the Antelopev2 embedding of the ORIGINAL crop.
Loss    = cos + 10*mae + 10*triplet(margin 0.3, in-batch roll negative).
AdamW lr 1e-5 / wd 2e-2, cosine schedule, batch 128, 10 epochs, train set repeated twice.

Same recipe as attacks/canfg/insight_train.py, except that a missing teacher embedding
raises instead of silently becoming a zero vector, and the run is resumable.

PerceptFace's own Limitation 4 predicts this: "an adversary can collect a large number
of paired protected and unprotected faces. They can invalidate our method by training an
en-decoder network."
"""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from onnx2torch import convert
from PIL import Image
import tqdm
import wandb

ONNX = '../../checkpoints/model.onnx'
PROT = '../../data/perceptface/protected224'
EPOCHS = 10
device = 'cuda'
os.makedirs('log', exist_ok=True)

teacher = pickle.load(open('log/teacher_embeddings_insight.pkl', 'rb'))
train_names = pickle.load(open('log/train_paths.pkl', 'rb'))
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


loader = DataLoader(Protected(train_names * 2), batch_size=128, shuffle=True,
                    num_workers=32, pin_memory=True, drop_last=True)
val_loader = DataLoader(Protected(val_names), batch_size=256, shuffle=False,
                        num_workers=16, pin_memory=True)

student = convert(ONNX).to(device)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=2e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * EPOCHS)

start_epoch = 0
if os.path.exists('ckpt.pt'):
    ck = torch.load('ckpt.pt', map_location='cpu', weights_only=False)
    student.load_state_dict(ck['student'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    start_epoch = ck['epoch'] + 1
    wandb.init(project='perceptface_distill', id=ck['wandb_id'], resume='must')
    print('resumed at epoch %d' % start_epoch, flush=True)
else:
    wandb.init(project='perceptface_distill')

for e in range(start_epoch, EPOCHS):
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
        scheduler.step()
        tot += [trip.item(), cos.item(), mae.item()]
        count += 1
        wandb.log({'triplet': trip.item(), 'cosine': cos.item(), 'mae': mae.item(),
                   'lr': scheduler.get_last_lr()[0]})
    wandb.log({'epoch': e, 'epoch_triplet': tot[0] / count,
               'epoch_cosine': tot[1] / count, 'epoch_mae': tot[2] / count})

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
    wandb.log({'epoch': e, 'val_epoch_triplet': vtot[0] / vcount,
               'val_epoch_cosine': vtot[1] / vcount, 'val_epoch_mae': vtot[2] / vcount})
    print('epoch %d  train_cos %.4f  val_cos %.4f  (loss terms, lower is better)'
          % (e, tot[1] / count, vtot[1] / vcount), flush=True)

    with open('log/insight_student_embeddings_val_epoch%d.pkl' % e, 'wb') as f:
        pickle.dump(val_emb, f)
    torch.save({'student': student.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': e,
                'wandb_id': wandb.run.id}, 'ckpt.pt')
    torch.save(student.state_dict(), 'student.pth')

print('DONE', flush=True)
