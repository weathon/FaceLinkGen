"""CanFG and CanFG-Ano protected images from the official checkpoints.

canfg      G from seed85_anonymized_100_id_0_em_500_lp_10.pt -- the released CanFG:
           physical identity removed AND a virtual identity embedded.
canfg_ano  anonymized_rec100_id_10_em_0_lp_0_G.pt -- the stage-1 PID remover only:
           physical identity removed, no virtual identity embedded. CanFG.py loads this
           exact file as its frozen self.Ano.

Neither is retrained. The upstream training scripts are not touched (Remover.py does not
even import: its load() has an `except` with no `try`).

Input is the 128 crop produced by CanFG's own MTCNN, ToTensor + Normalize(0.5, 0.5), so
[-1,1]; the generator's last layer is tanh, so the output is denormalised by (x+1)/2.

Usage: python d1_gen_canfg.py {canfg|canfg_ano}
"""
import os
import sys
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PRE = '/raid/wg25r/redteam_work/canfg_premodels/extracted'
CANFG = '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/methods/canfg'
WORK = '/raid/wg25r/redteam_work'
sys.path.insert(0, CANFG)
sys.path.insert(0, PRE)            # CanFG.py does `from premodels.irse import Backbone`
os.chdir(PRE)                      # and torch.load('premodels/...') with a relative path
from CanFG import Generator

CKPT = {
    'canfg': ('premodels/seed85_anonymized_100_id_0_em_500_lp_10.pt', 'G'),
    'canfg_ano': ('premodels/anonymized_rec100_id_10_em_0_lp_0_G.pt', None),
}
WHICH = sys.argv[1]
path, subkey = CKPT[WHICH]
device = 'cuda'

state = torch.load(path, map_location='cpu', weights_only=False)
if subkey is not None:
    state = state[subkey]
netG = Generator().to(device)
netG.load_state_dict(state)
netG.eval()
netG.requires_grad_(False)

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

JOBS = [
    ('ffhq', WORK + '/splits/ffhq_attack_2000.txt'),
    ('ffhq', WORK + '/splits/ffhq_gate_val.txt'),
    ('lfw', WORK + '/splits/lfw_gallery.txt'),
]


class Crops(Dataset):
    def __init__(self, src, names):
        self.src, self.names = src, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.names[i], tf(Image.open(os.path.join(self.src, self.names[i])).convert('RGB'))


for ds, split in JOBS:
    src = '%s/crops/%s/128' % (WORK, ds)
    dst = '%s/protected/%s/%s' % (WORK, WHICH, ds)
    os.makedirs(dst, exist_ok=True)
    rels = open(split).read().split()
    names = [os.path.splitext(r.replace('/', '__'))[0] + '.png' for r in rels]
    todo = sorted({n for n in names if not os.path.exists(os.path.join(dst, n))})
    print('%s %s: %d in split, %d todo' % (WHICH, os.path.basename(split), len(names), len(todo)),
          flush=True)

    loader = DataLoader(Crops(src, todo), batch_size=64, shuffle=False, num_workers=16,
                        pin_memory=True)
    done = 0
    for batch_names, imgs in loader:
        with torch.no_grad():
            out = (netG(imgs.to(device, non_blocking=True)) + 1) / 2
        for j, n in enumerate(batch_names):
            vutils.save_image(out[j], os.path.join(dst, n))
        done += len(batch_names)
        if done % 2000 < 64:
            print('  %d/%d' % (done, len(todo)), flush=True)
    print('  DONE %s -> %d on disk' % (dst, len(os.listdir(dst))), flush=True)
