"""PerceptFace protected images from the OFFICIAL released weights -- no retraining.

Forward pass copied verbatim from methods/perceptface/gen_protected.py; only the source
paths and the split restriction differ:
  ImageNet-normalised 224 input
  -> netArc(interp 112, bicubic) -> L2 norm   = original_id
  -> ID_transform (states['WI'])  -> L2 norm  = T_id
  -> netG(img_224, T_id) -> denormalise -> save png

Alignment matches by construction: c1_crop_224.py produces crops/*/224 with PerceptFace's
own Face_detect_crop (SCRFD, mode='None' arcface alignment, app.get(img, 224)) -- the same
code path as methods/perceptface/prep_crops.py -- so these crops are what the released
weights expect. Whether that carries over to LFW is what the gate measures.

Fully deterministic: no key, no randomness, so the paired distillation attack applies.

Usage: python d7_gen_perceptface_official.py
"""
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

REPO = '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen'
sys.path.insert(0, REPO + '/third_party/perceptface')
from fs_networks_fix import Generator_Adain_Upsample
from AIDPro_MSE import ID_transform

CKPT = REPO + '/checkpoints/perceptface'
WORK = '/raid/wg25r/redteam_work'
METHOD = 'perceptface_official'
device = 'cuda'

netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9).to(device)
netG.load_state_dict(torch.load(CKPT + '/90000_net_G.pth', map_location='cpu'))
netG.eval()

# arcface_checkpoint.tar is a pickled nn.Module, not a state_dict, so weights_only=False
# is required on torch >= 2.6 and models/arcface_models.py must be importable.
netArc = torch.load(CKPT + '/arcface_checkpoint.tar',
                    map_location='cpu', weights_only=False).to(device)
netArc.eval()

WI = ID_transform(512).to(device)
WI.load_state_dict(torch.load(CKPT + '/MSE_new_all_loss_id_5_rec_5_wa_5_step_40000.pt',
                              map_location='cpu')['WI'])
WI.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

JOBS = [
    ('ffhq', WORK + '/splits/ffhq_attack_2000.txt'),
    ('ffhq', WORK + '/splits/ffhq_gate_val.txt'),
    ('lfw', WORK + '/splits/lfw_query.txt'),
]


class Crops(data.Dataset):
    def __init__(self, src, names):
        self.src, self.names = src, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return (tf(Image.open(os.path.join(self.src, self.names[i])).convert('RGB')),
                self.names[i])


for ds, split in JOBS:
    src = '%s/crops/%s/224' % (WORK, ds)
    dst = '%s/protected/%s/%s' % (WORK, METHOD, ds)
    os.makedirs(dst, exist_ok=True)
    rels = open(split).read().split()
    names = [os.path.splitext(r.replace('/', '__'))[0] + '.png' for r in rels]
    todo = sorted({n for n in names if not os.path.exists(os.path.join(dst, n))})
    print('%s: %d in split, %d todo' % (os.path.basename(split), len(names), len(todo)),
          flush=True)

    loader = data.DataLoader(Crops(src, todo), batch_size=32, shuffle=False,
                             num_workers=16, pin_memory=True)
    done = 0
    with torch.no_grad():
        for imgs, batch_names in loader:
            imgs = imgs.to(device, non_blocking=True)
            emb = netArc(F.interpolate(imgs, (112, 112), mode='bicubic'))
            original_id = F.normalize(emb, p=2, dim=1)
            T_id = F.normalize(WI(original_id), p=2, dim=1)
            fake = netG(imgs, T_id)
            out = fake * imagenet_std + imagenet_mean
            for j, nm in enumerate(batch_names):
                vutils.save_image(out[j], os.path.join(dst, nm), nrow=1)
            done += len(batch_names)
            if done % 1600 < 32:
                print('  %d/%d' % (done, len(todo)), flush=True)
    print('  DONE %s -> %d on disk' % (dst, len(os.listdir(dst))), flush=True)
