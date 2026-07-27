"""Generate the PerceptFace-protected half of the (original, protected) pairs.

Faithful to the released generating_protected_faces_224.py / AIDPro_MSE.py:
  ImageNet-normalised 224 input
  -> netArc(interp 112, bicubic) -> L2 norm            = original_id
  -> ID_transform -> L2 norm                           = T_id
  -> netG(img_224, T_id) -> denormalise -> save png

PerceptFace is fully deterministic: no key, no randomness, the same input always maps to
the same protected image, which is what makes the paired distillation attack possible.
"""
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

PERCEPTFACE = '/path/to/PerceptFace_hf_space'
sys.path.insert(0, PERCEPTFACE)
from fs_networks_fix import Generator_Adain_Upsample
from AIDPro_MSE import ID_transform

SRC = '/path/to/perceptface_work/crops224'
DST = '/path/to/perceptface_work/protected224'
device = 'cuda'

os.makedirs(DST, exist_ok=True)
names = sorted(n for n in os.listdir(SRC) if n.endswith('.png'))
todo = [n for n in names if not os.path.exists(os.path.join(DST, n))]
print('crops %d, todo %d' % (len(names), len(todo)), flush=True)

netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9).to(device)
netG.load_state_dict(torch.load(PERCEPTFACE + '/pretrained_models/90000_net_G.pth',
                                map_location='cpu'))
netG.eval()

# arcface_checkpoint.tar is a pickled nn.Module, not a state_dict, so weights_only=False
# is required on torch >= 2.6, and models/arcface_models.py must be importable.
netArc = torch.load(PERCEPTFACE + '/pretrained_models/arcface_checkpoint.tar',
                    map_location='cpu', weights_only=False).to(device)
netArc.eval()

WI = ID_transform(512).to(device)
WI.load_state_dict(torch.load(
    PERCEPTFACE + '/pretrained_models/MSE_new_all_loss_id_5_rec_5_wa_5_step_40000.pt',
    map_location='cpu')['WI'])
WI.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class Crops(data.Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return tf(Image.open(os.path.join(SRC, self.names[i])).convert('RGB')), self.names[i]


loader = data.DataLoader(Crops(todo), batch_size=32, shuffle=False,
                         num_workers=16, pin_memory=True)

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

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
            vutils.save_image(out[j], os.path.join(DST, nm), nrow=1)
        done += len(batch_names)
        if done % 1600 == 0:
            print('%d/%d' % (done, len(todo)), flush=True)

print('DONE protected on disk = %d' % len(os.listdir(DST)), flush=True)
