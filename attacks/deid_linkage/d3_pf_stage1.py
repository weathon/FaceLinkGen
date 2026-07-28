"""PerceptFace Stage 1 (APIM): train E_attr + G + D from scratch on the FFHQ512
protect-train split. Copied verbatim from /home/wg25r/face_deid/PerceptFace/
train_stage1_apim.py; only SRC/OUT, the split restriction, and the wandb project differ.

Objective (paper Sec. III): L_adv + 30*L_id + 10*L_attr + 10*L_fus
  L_adv  : WGAN-GP, lambda_gp = 10
  L_id   : 1 - cos(E_id(G(x, z_id^src)), z_id^src)
  L_attr : weak feature matching = L1 on the LAST 3 discriminator layers
  L_fus  : ||G(x, z_id^own) - x||_1, on half the steps (self reconstruction)
Adam(0.5, 0.99), lr 4e-4, batch 16, 224x224, ImageNet normalisation.

FFHQ has one image per identity, so the swap source identity is the batch rolled by one.
Training set is the 64281-image protect-train split, disjoint from gate-val and attack-train.
Deviation from a literal pix2pixHD MultiscaleDiscriminator: InstanceNorm instead of
BatchNorm, because WGAN-GP's per-sample gradient penalty is not valid under BatchNorm.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import wandb

HF = '/home/wg25r/face_deid/PerceptFace/upstream/hfspace'
SS = '/home/wg25r/face_deid/PerceptFace/upstream/SimSwap'
sys.path.insert(0, HF)
sys.path.insert(0, SS)
from fs_networks_fix import Generator_Adain_Upsample
from models.networks import MultiscaleDiscriminator

SRC = '/raid/wg25r/redteam_work/crops/ffhq/224'
SPLIT = '/raid/wg25r/redteam_work/splits/ffhq_protect_train.txt'
OUT = '/raid/wg25r/redteam_work/ckpt/pf_stage1'
MAX_SECONDS = 6 * 3600
TOTAL_STEP = 400000
BATCH = 16
LAMBDA_ID, LAMBDA_ATTR, LAMBDA_FUS, LAMBDA_GP = 30.0, 10.0, 10.0, 10.0
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
        return tf(Image.open(os.path.join(SRC, self.names[i])).convert('RGB'))


loader = data.DataLoader(Crops(), batch_size=BATCH, shuffle=True, num_workers=16,
                         pin_memory=True, drop_last=True, persistent_workers=True)

netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9).to(device)
netD = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                               use_sigmoid=False, num_D=2, getIntermFeat=True).to(device)
netArc = torch.load(HF + '/pretrained_models/arcface_checkpoint.tar',
                    map_location='cpu', weights_only=False).to(device)
netArc.eval()
netArc.requires_grad_(False)

opt_G = torch.optim.Adam(netG.parameters(), lr=4e-4, betas=(0.5, 0.99), eps=1e-8)
opt_D = torch.optim.Adam(netD.parameters(), lr=4e-4, betas=(0.5, 0.99), eps=1e-8)

step = 0
if os.path.exists(OUT + '/ckpt.pt'):
    ck = torch.load(OUT + '/ckpt.pt', map_location='cpu', weights_only=False)
    netG.load_state_dict(ck['G'])
    netD.load_state_dict(ck['D'])
    opt_G.load_state_dict(ck['opt_G'])
    opt_D.load_state_dict(ck['opt_D'])
    step = ck['step']
    wandb.init(project='redteam_pf_stage1', id=ck['wandb_id'], resume='must')
    print('resumed at step %d' % step, flush=True)
else:
    wandb.init(project='redteam_pf_stage1')

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)


def critic(feats):
    return torch.stack([f[-1].mean() for f in feats]).mean()


last_fus = float('nan')   # a resume starting at step 999 prints before any even step
t0 = time.time()
stop = False
while not stop:
    for real in loader:
        real = real.to(device, non_blocking=True)
        with torch.no_grad():
            arc = F.normalize(netArc(F.interpolate(real, (112, 112), mode='bicubic')), p=2, dim=1)
        src_id = torch.roll(arc, 1, dims=0)

        # ---- D: WGAN-GP ----
        fake = netG(real, src_id).detach()
        eps = torch.rand(real.size(0), 1, 1, 1, device=device)
        mix = (eps * real + (1 - eps) * fake).requires_grad_(True)
        g = torch.autograd.grad(critic(netD(mix)), mix, create_graph=True)[0]
        gp = ((g.view(g.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        loss_D = critic(netD(fake)) - critic(netD(real)) + LAMBDA_GP * gp
        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()

        # ---- G ----
        fake = netG(real, src_id)
        fake_feats = netD(fake)
        with torch.no_grad():
            real_feats = netD(real)
        loss_adv = -critic(fake_feats)
        fake_arc = F.normalize(netArc(F.interpolate(fake, (112, 112), mode='bicubic')), p=2, dim=1)
        loss_id = (1 - (fake_arc * src_id).sum(1)).mean()
        loss_attr = torch.stack([F.l1_loss(fa[k], re[k])
                                 for fa, re in zip(fake_feats, real_feats)
                                 for k in (-3, -2, -1)]).mean()
        loss_G = loss_adv + LAMBDA_ID * loss_id + LAMBDA_ATTR * loss_attr

        # L_fus is only computed on half the steps, so it is only reported on those steps.
        # Reading it unconditionally logs the previous step's value on odd steps and
        # NameErrors outright when a resume starts on an odd step.
        fus = None
        if step % 2 == 0:
            recon = netG(real, arc)
            loss_fus = F.l1_loss(recon, real)
            loss_G = loss_G + LAMBDA_FUS * loss_fus
            fus = loss_fus.item()
        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

        logd = {'step': step, 'loss_D': loss_D.item(), 'gp': gp.item(),
                'loss_adv': loss_adv.item(), 'loss_id': loss_id.item(),
                'loss_attr': loss_attr.item()}
        if fus is not None:
            logd['loss_fus'] = fus
        wandb.log(logd)
        step += 1

        if fus is not None:
            last_fus = fus
        if step % 1000 == 0:
            # step was already incremented, so the step that just ran is odd and never
            # computes L_fus. Print the most recent one that did; wandb has the real series.
            print('step %d  D %.3f  adv %.3f  id %.4f  attr %.4f  fus(last) %.4f  (%.0f s)'
                  % (step, loss_D.item(), loss_adv.item(), loss_id.item(),
                     loss_attr.item(), last_fus, time.time() - t0), flush=True)
            with torch.no_grad():
                netG.eval()
                grid = torch.cat([real[:8], netG(real[:8], torch.roll(arc[:8], 1, dims=0)),
                                  netG(real[:8], arc[:8])])
                netG.train()
            vutils.save_image(grid * imagenet_std + imagenet_mean,
                              OUT + '/samples/step%07d.png' % step, nrow=8)
            torch.save({'G': netG.state_dict(), 'D': netD.state_dict(),
                        'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
                        'step': step, 'wandb_id': wandb.run.id}, OUT + '/ckpt.pt')

        if time.time() - t0 > MAX_SECONDS or step >= TOTAL_STEP:
            stop = True
            break

torch.save({'G': netG.state_dict(), 'D': netD.state_dict(),
            'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict(),
            'step': step, 'wandb_id': wandb.run.id}, OUT + '/ckpt.pt')
print('STOPPED at step %d after %.0f s' % (step, time.time() - t0), flush=True)
