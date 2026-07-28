"""PerceptFace Stage 2 (PEIT): train ONLY the ID_transform MLP, on the FFHQ512
protect-train split. Copied verbatim from /home/wg25r/face_deid/PerceptFace/
train_stage2_peit.py; only SRC/MASKS/OUT, the split restriction, the frozen netG source,
and the wandb project differ.

netG (the Stage-1 G retrained here, not the official 90000_net_G.pth) and ArcFace are frozen; gradients reach T through G.
Objective: 5*L_pri + 5*L_per
  L_pri = max(eps, cos(E_id(x), E_id(x_hat))),  eps = 0.1
  L_per = LPIPS(alex) + sum_i alpha_i * MSE over region i
          alpha = [brow .192, eye .223, nose .183, mouth .229, skin .174]
Adam(0.99, 0.99), lr 4e-4, batch 16.

Undocumented in the paper, chosen here: E_id is the SimSwap ArcFace shipped with
PerceptFace (the only differentiable recogniser in the released pipeline); the region
term is a mean squared error over the masked pixels of the [0,1] image.
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import cv2
from PIL import Image
import lpips
import wandb

HF = '/home/wg25r/face_deid/PerceptFace/upstream/hfspace'
sys.path.insert(0, HF)
from fs_networks_fix import Generator_Adain_Upsample
from AIDPro_MSE import ID_transform

SRC = '/raid/wg25r/redteam_work/crops/ffhq/224'
MASKS = '/raid/wg25r/redteam_work/masks/ffhq/224'
SPLIT = '/raid/wg25r/redteam_work/splits/ffhq_protect_train.txt'
STAGE1 = '/raid/wg25r/redteam_work/ckpt/pf_stage1/ckpt.pt'
OUT = '/raid/wg25r/redteam_work/ckpt/pf_stage2'
MAX_SECONDS = 6 * 3600
TOTAL_STEP = 400000
BATCH = 16
EPS = 0.1
ALPHA = torch.tensor([0.192, 0.223, 0.183, 0.229, 0.174])
device = 'cuda'
os.makedirs(OUT + '/samples', exist_ok=True)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class Crops(data.Dataset):
    def __init__(self):
        self.names = sorted(open(SPLIT).read().split())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = tf(Image.open(os.path.join(SRC, n)).convert('RGB'))
        lab = torch.from_numpy(cv2.imread(os.path.join(MASKS, n), 0).astype(np.int64))
        return img, lab


loader = data.DataLoader(Crops(), batch_size=BATCH, shuffle=True, num_workers=16,
                         pin_memory=True, drop_last=True, persistent_workers=True)

netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9).to(device)
# The Stage-1 G retrained here on FFHQ512, NOT the official 90000_net_G.pth.
netG.load_state_dict(torch.load(STAGE1, map_location='cpu', weights_only=False)['G'])
netG.eval()
netG.requires_grad_(False)

netArc = torch.load(HF + '/pretrained_models/arcface_checkpoint.tar',
                    map_location='cpu', weights_only=False).to(device)
netArc.eval()
netArc.requires_grad_(False)

WI = ID_transform(512).to(device)
loss_lpips = lpips.LPIPS(net='alex').to(device)
loss_lpips.requires_grad_(False)

opt = torch.optim.Adam(WI.parameters(), lr=4e-4, betas=(0.99, 0.99))

step = 0
if os.path.exists(OUT + '/ckpt.pt'):
    ck = torch.load(OUT + '/ckpt.pt', map_location='cpu', weights_only=False)
    WI.load_state_dict(ck['WI'])
    opt.load_state_dict(ck['opt'])
    step = ck['step']
    wandb.init(project='redteam_pf_stage2', id=ck['wandb_id'], resume='must')
    print('resumed at step %d' % step, flush=True)
else:
    wandb.init(project='redteam_pf_stage2')

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
alpha = ALPHA.to(device)

t0 = time.time()
stop = False
while not stop:
    for x, lab in loader:
        x = x.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)

        with torch.no_grad():
            z_id = F.normalize(netArc(F.interpolate(x, (112, 112), mode='bicubic')), p=2, dim=1)
        T_id = F.normalize(WI(z_id), p=2, dim=1)
        xh = netG(x, T_id)

        zh = F.normalize(netArc(F.interpolate(xh, (112, 112), mode='bicubic')), p=2, dim=1)
        cos = (z_id * zh).sum(1).mean()
        loss_pri = torch.clamp(cos, min=EPS)

        x01 = x * imagenet_std + imagenet_mean
        xh01 = xh * imagenet_std + imagenet_mean
        l_lpips = loss_lpips(x01 * 2 - 1, xh01 * 2 - 1).mean()
        se = (x01 - xh01) ** 2
        l_region = torch.stack([alpha[k] * se[(lab == k + 1)[:, None].expand_as(se)].mean()
                                for k in range(5)]).sum()
        loss = 5 * loss_pri + 5 * (l_lpips + l_region)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        wandb.log({'step': step, 'loss': loss.item(), 'cos_id': cos.item(),
                   'loss_pri': loss_pri.item(), 'lpips': l_lpips.item(),
                   'l_region': l_region.item()})
        step += 1

        if step % 1000 == 0:
            print('step %d  cos %.4f  lpips %.4f  region %.5f  (%.0f s)'
                  % (step, cos.item(), l_lpips.item(), l_region.item(), time.time() - t0),
                  flush=True)
            vutils.save_image(torch.cat([x01[:8], xh01[:8]]),
                              OUT + '/samples/step%07d.png' % step, nrow=8)
            torch.save({'WI': WI.state_dict(), 'opt': opt.state_dict(), 'step': step,
                        'wandb_id': wandb.run.id}, OUT + '/ckpt.pt')

        if time.time() - t0 > MAX_SECONDS or step >= TOTAL_STEP:
            stop = True
            break

torch.save({'WI': WI.state_dict(), 'opt': opt.state_dict(), 'step': step,
            'wandb_id': wandb.run.id}, OUT + '/ckpt.pt')
print('STOPPED at step %d after %.0f s' % (step, time.time() - t0), flush=True)
