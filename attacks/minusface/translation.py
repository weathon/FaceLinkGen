import os
import sys
sys.path.insert(0, "../../methods/minusface")
sys.path.insert(0, "../../third_party/tface/recognition")
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from minusface import MinusBackbone
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pandas as pd
import tqdm

ANOTHER = False

import cv2
from insightface.app import FaceAnalysis
import torch

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

def get_embedding(path):
    image = cv2.imread(path)
    faces = app.get(image)
    if len(faces) == 0:
        return torch.zeros(1, 512)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    return faceid_embeds


device = "cuda" if torch.cuda.is_available() else "cpu"

teacher = InceptionResnetV1(pretrained='vggface2').eval().to(device)
student = InceptionResnetV1(pretrained='vggface2').to(device)
if os.path.exists("vgg_student.pth"):
    student.load_state_dict(torch.load("vgg_student.pth", map_location='cpu'))
    student.eval()
    student = student.to(device)
    # wandb.init(project="student_distill", resume="must", id="kcftuxvl")


for p in teacher.parameters():
    p.requires_grad = False

conversion_model = MinusBackbone(mode='stage1')
conversion_model.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
conversion_model = conversion_model.eval().to(device)

tf_teacher = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
])

tf_student = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
])

tf_conv = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])
all_root = '/path/to/celeba_aligned'

df = pd.read_csv('/path/to/Identity_CelebA.txt', sep=' ')
df.columns = ["col0","img_name","id"]

def get_another_image(df, img_name): 
    row = df[df["img_name"] == img_name]
    if row.empty:
        return None
    id_val = row["id"].iloc[0]
    imgs = df[df["id"] == id_val]["img_name"].tolist()
    imgs = [x for x in imgs if x != img_name]
    if not imgs:
        return None 
    return imgs[0]

def select_teacher_image(path):
    img_name = os.path.basename(path)
    alt = get_another_image(df, img_name)
    if alt is not None:
        return os.path.join(all_root, alt)
    return path

def list_images(root):
    out = []
    exts = (".jpg",".jpeg",".png")
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts):
                out.append(os.path.join(r,f))
    return out

class FaceDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        p_t = select_teacher_image(p) if ANOTHER else p
        p_s = select_teacher_image(p) if ANOTHER else p
        img_t = Image.open(p_t).convert("RGB")
        img_s = Image.open(p_s).convert("RGB")
        buffalo_l_embeddings = get_embedding(p_s)
        t_img = tf_teacher(img_t)
        c_img = tf_conv(img_s)
        
        return p, t_img, c_img, buffalo_l_embeddings

def convert_batch(conv_raw):
    conv_raw = conv_raw.to(device)
    with torch.no_grad():
        out = conversion_model(conv_raw)
    out = out[5]
    out = out.permute(0,2,3,1).cpu().numpy()
    out = (out - out.min(axis=(1,2,3),keepdims=True)) / (out.max(axis=(1,2,3),keepdims=True) - out.min(axis=(1,2,3),keepdims=True) + 1e-8)
    out = np.clip(out[...,:3],0,1)
    out = (out*255).astype(np.uint8)
    imgs = [Image.fromarray(x) for x in out]
    final = torch.stack([tf_student(im) for im in imgs]).to(device)
    return final

def cosine_triplet(a, p, n, margin=0.3):
    a = F.normalize(a, dim=1)
    p = F.normalize(p, dim=1)
    n = F.normalize(n, dim=1)
    d_ap = 1 - (a * p).sum(dim=1)
    d_an = 1 - (a * n).sum(dim=1)
    return torch.clamp(d_ap - d_an + margin, min=0).mean()

def cosine_sim_loss(a, b):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (1 - (a * b).sum(dim=1)).mean()

def loss_fn(s, t):
    s_n = F.normalize(s, dim=1)
    t_n = F.normalize(t, dim=1)
    idx = torch.arange(s.size(0), device=s.device)
    neg = t_n[torch.roll(idx, shifts=1)]
    trip = cosine_triplet(s_n, t_n, neg)
    cos = cosine_sim_loss(s, t)
    mae = F.l1_loss(s_n, t_n)
    return trip, cos, mae

paths = list_images("training")
dataset = FaceDataset(paths)
loader = DataLoader(dataset, batch_size=256 + 128, shuffle=False, num_workers=32, pin_memory=True)

epochs = 20
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * epochs)

student_embeddings = []
teacher_embeddings = []
buffalo_l_embeddings_list = []
filenames = []
with torch.no_grad(): 
    for filenames, t_img, conv_raw, buffalo_l_embeddings in tqdm.tqdm(loader):
        t_img = t_img.to(device)
        t_emb = teacher(t_img)
        s_img = convert_batch(conv_raw)
        s_emb = student(s_img)
        trip, cos, mae = loss_fn(s_emb, t_emb)
        student_embeddings.append(s_emb.cpu())
        teacher_embeddings.append(t_emb.cpu())
        filenames.extend(filenames)
        buffalo_l_embeddings_list.append(buffalo_l_embeddings)
        print(f"Triplet Loss: {trip.item():.4f}, Cosine Loss: {cos.item():.4f}, MAE Loss: {mae.item():.4f}")

filename = "embeddings_another.pkl" if ANOTHER else "embeddings.pkl"
with open(filename, "wb") as f:
    import pickle
    pickle.dump({
        "student": torch.cat(student_embeddings, dim=0),
        "teacher": torch.cat(teacher_embeddings, dim=0),
        "buffalo_l": torch.cat(buffalo_l_embeddings_list, dim=0),
        "filenames": filenames
    }, f)
