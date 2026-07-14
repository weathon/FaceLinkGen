import pickle
import numpy as np
import torch
import torch.nn.functional as F

embedding_root = '/path/to/casia-webface/insight_embeddings'

with open("val_frac_lfw.pkl", "rb") as f:
    d = pickle.load(f)

student_embeddings = d["student_embeddings"]  # list of (B,512) tensors
filenames = d["filenames"]                    # list of tuples of paths (per batch)

student = torch.cat([e if torch.is_tensor(e) else torch.tensor(e) for e in student_embeddings], dim=0).float()
flat_names = [p for batch in filenames for p in batch]
assert len(flat_names) == student.shape[0], (len(flat_names), student.shape[0])

teacher = []
for p in flat_names:
    npy = embedding_root + "/" + p.replace("/path/to/casia-webface/", "").replace("/", "_").replace(".jpg", ".npy")
    teacher.append(np.load(npy))
teacher = torch.from_numpy(np.stack(teacher)).float()

s = F.normalize(student, dim=1)
t = F.normalize(teacher, dim=1)
cos = (s * t).sum(dim=1)

print("N =", len(flat_names))
print("mean cosine sim   =", cos.mean().item())
print("mean cosine loss  =", (1 - cos).mean().item())
print("median cosine sim =", cos.median().item())
print("min/max cos sim   =", cos.min().item(), cos.max().item())
