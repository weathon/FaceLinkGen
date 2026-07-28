"""FFHQ512 / LFW -> 128 crops with CanFG's OWN MTCNN alignment.

The released CanFG checkpoint was trained on CelebA aligned this exact way, and CanFG is
very sensitive to the crop, so this must be the upstream code path -- methods/canfg/
data_pre-processing/mtcnn.py, align_multi(min_face_size=64, crop_size=(128,128)), run on
the ORIGINAL images (not on the 224 SCRFD crops, which would be a double alignment).

align_multi returns every detected face; upstream preprocess_your_images.py saves them all
to one filename and lets them overwrite each other. Here an image with anything other than
exactly one detection is skipped whole and recorded. MTCNN is on cuda:0 at module scope.

Usage: python c2_crop_128.py {ffhq|lfw}
"""
import os
import sys
from PIL import Image

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/methods/canfg/data_pre-processing')
sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/methods/canfg/data_pre-processing/mtcnn_pytorch')
from mtcnn import MTCNN

WORK = '/raid/wg25r/redteam_work'
SETS = {
    'ffhq': '/raid/wg25r/ffhq512_hf/images/FFHQ512/FFHQ512',
    'lfw': '/raid/wg25r/lfw/lfw-deepfunneled',
}
WHICH = sys.argv[1]
SRC = SETS[WHICH]
DST = WORK + '/crops/' + WHICH + '/128'
SKIPFILE = WORK + '/crops/' + WHICH + '/skipped_128.txt'

os.makedirs(DST, exist_ok=True)
if WHICH == 'ffhq':
    rels = sorted(n for n in os.listdir(SRC) if n.endswith('.png'))
else:
    rels = sorted(p + '/' + f for p in os.listdir(SRC) if not p.startswith('.')
                  for f in os.listdir(os.path.join(SRC, p)))

known = set()
if os.path.exists(SKIPFILE):          # resume state; absent on the first run
    known = set(open(SKIPFILE).read().split())
todo = [r for r in rels
        if r not in known
        and not os.path.exists(os.path.join(DST, os.path.splitext(r.replace('/', '__'))[0] + '.png'))]
print('%s: candidates %d, known-skipped %d, todo %d'
      % (WHICH, len(rels), len(known), len(todo)), flush=True)

mtcnn = MTCNN()
skipped = []
for i, rel in enumerate(todo):
    img = Image.open(os.path.join(SRC, rel)).convert('RGB')
    faces = mtcnn.align_multi(img, min_face_size=64, crop_size=(128, 128))
    if faces is None or len(faces) != 1:
        n = 0 if faces is None else len(faces)
        skipped.append(rel)
        print('SKIP %s faces=%d' % (rel, n), flush=True)
    else:
        faces[0].save(os.path.join(DST, os.path.splitext(rel.replace('/', '__'))[0] + '.png'))
    if (i + 1) % 2000 == 0:
        print('%d/%d skipped=%d' % (i + 1, len(todo), len(skipped)), flush=True)

with open(SKIPFILE, 'w') as f:
    for r in sorted(known | set(skipped)):
        f.write(r + '\n')
print('DONE %s cropped=%d skipped_this_run=%d skipped_total=%d on_disk=%d'
      % (WHICH, len(todo) - len(skipped), len(skipped),
         len(known | set(skipped)), len(os.listdir(DST))), flush=True)
