"""Pre-extract Antelopev2 embeddings for the PerceptFace pairs and write the split.

  teacher_embeddings_insight.pkl    original crops   -> the distillation target
  protected_embeddings_insight.pkl  protected images -> the pre-attack baseline

glintr100 is run through onnx2torch on the GPU because the ONNXRuntime CPU build takes
about two hours for this set. Preprocessing replicates insightface's
ArcFaceONNX.get_feat exactly: cv2 bilinear resize to 112, BGR->RGB, (x - 127.5) / 127.5.
Running the teacher through the same torch graph as the student also means the
pre-attack and post-attack numbers come out of one implementation.

Unlike the other extract_embeddings.py in this repo, a failed read raises instead of
being stored as a zero vector.
"""
import os
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
from onnx2torch import convert

ONNX = '../../checkpoints/model.onnx'
CROPS = '/path/to/perceptface_work/crops224'
PROT = '/path/to/perceptface_work/protected224'
N_TRAIN = 8000
device = 'cuda'

names = sorted(set(os.listdir(CROPS)) & set(os.listdir(PROT)))
print('pairs %d' % len(names), flush=True)
jobs = [(CROPS, n) for n in names] + [(PROT, n) for n in names]


class Faces(data.Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        root, name = self.jobs[i]
        img = cv2.imread(os.path.join(root, name))
        if img is None:
            raise RuntimeError('unreadable ' + os.path.join(root, name))
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return torch.from_numpy((img - 127.5) / 127.5).permute(2, 0, 1), i


loader = data.DataLoader(Faces(jobs), batch_size=256, shuffle=False, num_workers=32,
                         pin_memory=True)

rec = convert(ONNX).to(device).eval()

teacher = {}
protected = {}
done = 0
with torch.no_grad():
    for x, idx in loader:
        emb = rec(x.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
        for k, j in enumerate(idx.tolist()):
            root, name = jobs[j]
            if root == CROPS:
                teacher[name] = emb[k]
            else:
                protected[name] = emb[k]
        done += len(idx)
        if done % 2560 == 0:
            print('%d/%d' % (done, len(jobs)), flush=True)

os.makedirs('log', exist_ok=True)
with open('log/teacher_embeddings_insight.pkl', 'wb') as f:
    pickle.dump(teacher, f)
with open('log/protected_embeddings_insight.pkl', 'wb') as f:
    pickle.dump(protected, f)

# FFHQ has one image per identity, so a deterministic split by filename already keeps
# identities disjoint between train and val.
train_names = names[:N_TRAIN]
val_names = names[N_TRAIN:]
with open('log/train_paths.pkl', 'wb') as f:
    pickle.dump(train_names, f)
with open('log/val_paths.pkl', 'wb') as f:
    pickle.dump(val_names, f)
print('teacher %d protected %d  train %d val %d'
      % (len(teacher), len(protected), len(train_names), len(val_names)), flush=True)
