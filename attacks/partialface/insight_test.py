import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
TEST_ONLY = True
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pandas as pd
import tqdm

import cv2
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
student = torch.nn.Sequential(
    torch.nn.Conv2d(27, 3, kernel_size=3, padding=1),
    student,
).to(device)



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
        embedding_teacher = torch.zeros((1, 512))#pre_loaded_teachers[p]  # (1, 512)
        img_s = DeepFace.extract_faces(p, detector_backend = "opencv", enforce_detection=False)[0]["face"]
        img_s = Image.fromarray((img_s * 255).astype("uint8"))
        # img_s = Image.open(p).convert("RGB")
        c_img = tf_conv(img_s)
        # print("Converted image shape:", c_img.shape)
        return p, embedding_teacher, c_img
    
import time
import processing_utils as util

def convert_batch(conv_raw): 
    time_s = time.time()
    final = util.form_training_batch(conv_raw, [1] * len(conv_raw))[0]
    
    print("Conversion time:", time.time() - time_s) #after putting the time the idel time decreased wtf
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
    return trip * 10, cos, mae * 10

MAX_SAMPLES = 100_000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAIRFACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "fracface", "soft"))
FAIRFACE_SPLIT = os.environ.get("FAIRFACE_SPLIT", "train")
paths = os.listdir(os.path.join(FAIRFACE_ROOT, FAIRFACE_SPLIT))[:MAX_SAMPLES]
paths = [os.path.join(FAIRFACE_ROOT, FAIRFACE_SPLIT, p) for p in paths]
# paths = []
# root = "/path/to/lfw"
# for id in os.listdir(root):
#     id_folder = os.path.join(root, id)
#     if not os.path.isdir(id_folder):
#         continue
#     for file in os.listdir(id_folder):
#         if file.endswith(".jpg"):
#             paths.append(os.path.join(id_folder, file))
# paths = sorted(paths)
# random.seed(42)
# random.shuffle(paths) 
# paths = paths[:1000]
# val_dataset = FaceDataset(paths)  

# val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
# paths = []
# with open("/path/to/lfw/lfw_ann.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip().split(" ")
#         # if line[0] != "1":
#         #     continue
#         img1 = "/path/to/lfw/" + line[1]
#         img2 = "/path/to/lfw/" + line[2]
#         paths.extend([img1, img2])

paths = sorted(paths)
random.seed(42)
random.shuffle(paths)
# paths = paths[:1000]
dataset = FaceDataset(paths)  

print("Dataset size:", len(dataset))
val_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

os.makedirs("log", exist_ok=True)

student_embeddings = []
filenames = []
student.load_state_dict(torch.load("student.pth", map_location='cpu'))
# student.load_state_dict(torch.load("/path/to/FaceLinkGen/fracface/blackbox/student.pth.bak", map_location='cpu'))
student = student.eval()

with torch.no_grad():
    for fn, embedding_teacher, conv_raw in tqdm.tqdm(val_loader):
        embedding_teacher = embedding_teacher.to(device)
        s_img = convert_batch(conv_raw.to(device))
        s_emb = student(s_img) 
        # embedding_teacher is a placeholder; keep only student embeddings
        student_embeddings.append(s_emb.detach().cpu())
        filenames.append(fn)
import pickle
output_name = "test_fairface.pkl" if FAIRFACE_SPLIT == "train" else "test_fairface_val.pkl"
with open(output_name, "wb") as f:
    pickle.dump({
        "filenames": filenames,
        "student_embeddings": student_embeddings,
    }, f)
