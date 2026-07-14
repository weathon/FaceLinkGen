import os
import sys
sys.path.insert(0, "../../methods/minusface")
sys.path.insert(0, "../../third_party/tface/recognition")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

TEST_ONLY = True
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
from insightface.app import FaceAnalysis
from datasets import load_from_disk
from tasks.partialface.utils import dct_transform, idct_transform

import cv2
from insightface.app import FaceAnalysis 
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# student = InceptionResnetV1(pretrained='vggface2').train().to(device)
# student = torch.nn.Sequential(
#     student,
#     torch.nn.Linear(512, 1024),
#     torch.nn.ReLU(),
#     torch.nn.Linear(1024, 512)
# ).to(device)
# if os.path.exists("student.pth"):
#     student.load_state_dict(torch.load("student.pth", map_location='cpu'))
#     wandb.init(project="student_distill_insight", resume="must", id="lqwdb3s7")
import torch
from onnx2torch import convert

onnx_model_path = "../../checkpoints/model.onnx"
student = convert(onnx_model_path).to(device)
# for name, param in student.named_parameters():
#     if "layer1" in name or "layer2" in name or "layer3" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
# student = torch.nn.Sequential(
#     torch.nn.Conv2d(192, 3, kernel_size=3, padding=1),
#     student,
# ).to(device)

conversion_model = MinusBackbone(mode='stage1')
conversion_model.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
conversion_model = conversion_model.eval().to(device)



tf_student = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

tf_conv = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])

df = pd.read_csv('/path/to/Identity_CelebA.txt', sep=' ')
df.columns = ["col0","img_name","id"]
 
def get_another_image(df, img_name):
    return None
    # row = df[df["img_name"] == img_name]
    # if row.empty: 
    #     return None
    # id_val = row["id"].iloc[0]
    # imgs = df[df["id"] == id_val]["img_name"].tolist()
    # imgs = [x for x in imgs if x != img_name]
    # if not imgs:
    #     return None
    # return random.choice(imgs) 

def select_teacher_image(path):
    # if random.random() < 0.5:
    #     img_name = os.path.basename(path)
    #     alt = get_another_image(df, img_name)
    #     if alt is not None:
    #         return os.path.join(os.path.dirname(path), alt)
    return path


def list_images(root): 
    out = []
    for sample in ds:
        img_name = sample["image_name"]
        if sample["embedding"] is None:
            continue
        full_path = os.path.join(root, img_name)
        if os.path.exists(full_path):
            out.append(full_path)
    return out

from deepface import DeepFace

embedding_root = '/path/to/casia-webface/insight_embeddings'
pre_loaded_teachers = {}
class FaceDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        # print("Loading embedding from:", p_t)
        embedding_teacher = pre_loaded_teachers[p]  # (1, 512)
        img_s = DeepFace.extract_faces(p, detector_backend = "opencv", enforce_detection=False)[0]["face"]
        img_s = img_s[..., ::-1]
        img_s = Image.fromarray((img_s * 255).astype("uint8"))
        # img_s = Image.open(p).convert("RGB")
        c_img = tf_conv(img_s)
        # print("Converted image shape:", c_img.shape)
        return p, embedding_teacher, c_img
    
import time
# def convert_batch(conv_raw): 
#     time_s = time.time()
#     conv_raw = conv_raw.to(device) 
#     with torch.no_grad():
#         out = conversion_model(conv_raw)
#     out = out[5]
#     out = out.permute(0,2,3,1).cpu().numpy() 
#     # out = (out - out.min(axis=(1,2,3),keepdims=True)) / (out.max(axis=(1,2,3),keepdims=True) - out.min(axis=(1,2,3),keepdims=True) + 1e-8)
#     # out = np.clip(out[...,:3],0,1)
#     # out = (out*255).astype(np.uint8) 
#     # imgs = [Image.fromarray(x) for x in out] 
#     # final = torch.stack([tf_student(im) for im in imgs]).to(device)
#     imgs = torch.from_numpy(out).permute(0, 3, 1, 2).float()
#     final = tf_student(imgs).to(device)

#     print("Conversion time:", time.time() - time_s) #after putting the time the idel time decreased wtf
#     return final

import torch.nn.functional as F

def convert_batch(conv_raw):
    conv_raw = conv_raw.to(device)
    with torch.no_grad():
        out = conversion_model(conv_raw)[5]

    imgs = out.float()
    # print(imgs.shape)
    # imgs = F.interpolate(imgs, size=(112, 112), mode="bilinear")

    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    imgs = (imgs - 0.5) / 0.5  # Normalize to [-1, 1]

    return imgs


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
    return trip * 10, cos, mae * 10

paths = []
root = '/path/to/casia-webface'
with open("../../data_splits/index.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        filename, split = line.strip().split()
        if split == "train":
            paths.append(os.path.join(root, filename))
        

dataset = FaceDataset(paths)  


val_paths = []
root = '/path/to/casia-webface'
with open("../../data_splits/index.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        filename, split = line.strip().split()
        if split != "train":
            val_paths.append(os.path.join(root, filename)) 
        

val_dataset = FaceDataset(val_paths)  

import pickle
for path in tqdm.tqdm(val_paths, desc="Preloading embeddings"):
    embedding = np.load(embedding_root+"/"+path.replace("/path/to/casia-webface/","").replace("/", "_").replace(".jpg", ".npy"))[None,:]
    pre_loaded_teachers[path] = torch.from_numpy(embedding).float()


import pickle
for path in tqdm.tqdm(paths, desc="Preloading embeddings"):
    embedding = np.load(embedding_root+"/"+path.replace("/path/to/casia-webface/","").replace("/", "_").replace(".jpg", ".npy"))[None,:]
    pre_loaded_teachers[path] = torch.from_numpy(embedding).float()

print("Dataset size:", len(dataset))
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
wandb.init(project="student_distill_insight") 
epochs = 10
# optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
# Use Adam, lr 5e-5 for backbone and 1e-4 for head 
optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)#, weight_decay=4e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * epochs)
os.makedirs("log", exist_ok=True)

templates = []
# generate data
# print("🚨🚨🚨🚨🚨🚨🚨 Data Gen, not normal run")
# for fn, embedding_teacher, conv_raw in tqdm.tqdm(loader):
#     embedding_teacher = embedding_teacher.to(device)
#     s_img = convert_batch(conv_raw)
#     templates.append(s_img.detach().cpu()) 

# with open(f"log/insight_student_templates.pkl", "wb") as f:
#     pickle.dump({
#         "templates": templates
#     }, f)
    
# quit()
for e in range(epochs): 
    student = student.train()
    total_trip = 0
    total_cos = 0
    count = 0 
    total_mae = 0
    for fn, embedding_teacher, conv_raw in tqdm.tqdm(loader):
        embedding_teacher = embedding_teacher.to(device)
        s_img = convert_batch(conv_raw)
        # print(s_img.shape) 
        s_emb = student(s_img)
        # print(embedding_teacher.shape, s_emb.shape)
        trip, cos, mae = loss_fn(s_emb, embedding_teacher.squeeze(1))
        loss = cos# + mae# + trip
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_trip += trip.item()
        total_cos += cos.item()
        total_mae += mae.item()
        count += 1
        wandb.log({"triplet": trip.item(), "cosine": cos.item(), "mae": mae.item(), "lr": scheduler.get_last_lr()[0]})
    torch.save(student.state_dict(), "student.pth")
    wandb.log({"epoch_triplet": total_trip/count, "epoch_cosine": total_cos/count, "epoch_mae": total_mae/count})

    student = student.eval()
    print("Validation on val set")
    total_trip = 0
    total_cos = 0
    count = 0
    total_mae = 0
    teacher_embeddings = []
    student_embeddings = []
    filenames = [] 
    templates = [] 
    with torch.no_grad():
        for fn, embedding_teacher, conv_raw in tqdm.tqdm(val_loader):
            embedding_teacher = embedding_teacher.to(device)
            s_img = convert_batch(conv_raw)
            s_emb = student(s_img)
            templates.append(s_img.detach().cpu()) 
            teacher_embeddings.append(embedding_teacher.detach().cpu())
            student_embeddings.append(s_emb.detach().cpu())
            filenames.append(fn)
            trip, cos, mae = loss_fn(s_emb, embedding_teacher.squeeze(1))
            total_trip += trip.item()
            total_cos += cos.item()
            total_mae += mae.item()
            count += 1
    wandb.log({"val_epoch_triplet": total_trip/count, "val_epoch_cosine": total_cos/count, "val_epoch_mae": total_mae/count})
    with open(f"log/insight_student_embeddings_val_epoch{e}.pkl", "wb") as f:
        pickle.dump({
            "filenames": filenames,
            "student_embeddings": student_embeddings,
            "teacher_embeddings": teacher_embeddings,
            "templates": templates
        }, f)
