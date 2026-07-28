"""BiSeNet (CelebAMask-HQ) face-region masks for Stage 2's L_region.

Adapted from /home/wg25r/face_deid/PerceptFace/precompute_masks.py. The label-grouping and
tensor ops are kept verbatim so the masks are identical in convention; only the paths
change, and it runs over the protect-train split rather than every crop on disk.

Single-channel PNG per crop, values 0 none / 1 brow / 2 eye / 3 nose / 4 mouth / 5 skin,
matching the alpha order [0.192, 0.223, 0.183, 0.229, 0.174] in the paper.
CelebAMask-HQ labels: skin 1, l_brow 2, r_brow 3, l_eye 4, r_eye 5, nose 10,
mouth 11, u_lip 12, l_lip 13.
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/SimSwap')
from parsing_model.model import BiSeNet

WORK = '/raid/wg25r/redteam_work'
SRC = WORK + '/crops/ffhq/224'
DST = WORK + '/masks/ffhq/224'
PARSER = '/raid/wg25r/perceptface_work/parser/79999_iter.pth'
device = 'cuda'
os.makedirs(DST, exist_ok=True)

names = sorted(open(WORK + '/splits/ffhq_protect_train.txt').read().split())
todo = [n for n in names if not os.path.exists(os.path.join(DST, n))]
print('protect-train %d todo %d' % (len(names), len(todo)), flush=True)

net = BiSeNet(n_classes=19).to(device)
net.load_state_dict(torch.load(PARSER, map_location='cpu'))
net.eval()

tf = transforms.Compose([
    transforms.Resize((512, 512)),
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


loader = data.DataLoader(Crops(todo), batch_size=32, shuffle=False, num_workers=16,
                         pin_memory=True)

GROUPS = [[2, 3], [4, 5], [10], [11, 12, 13], [1]]

done = 0
with torch.no_grad():
    for imgs, batch_names in loader:
        lab = net(imgs.to(device, non_blocking=True))[0].argmax(1)
        lab = F.interpolate(lab[:, None].float(), (224, 224), mode='nearest')[:, 0].long()
        out = torch.zeros_like(lab)
        for gi, ids in enumerate(GROUPS):
            m = torch.zeros_like(lab, dtype=torch.bool)
            for c in ids:
                m |= lab == c
            out[m] = gi + 1
        out = out.byte().cpu().numpy()
        for j, nm in enumerate(batch_names):
            cv2.imwrite(os.path.join(DST, nm), out[j])
        done += len(batch_names)
        if done % 6400 == 0:
            print('%d/%d' % (done, len(todo)), flush=True)

print('DONE masks on disk = %d' % len(os.listdir(DST)), flush=True)
cov = np.stack([(cv2.imread(os.path.join(DST, n), 0) == k).mean()
                for n in names[:200] for k in range(6)]).reshape(-1, 6).mean(0)
print('mean pixel coverage [none, brow, eye, nose, mouth, skin] = %s' % np.round(cov, 4))
