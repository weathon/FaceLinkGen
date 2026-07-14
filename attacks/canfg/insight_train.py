import os

import sys
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
student = convert(onnx_model_path)
# print(student)
# student.load_state_dict(torch.load("./blackbox/student.pth.bak", map_location='cpu'))
# for name, param in student.named_parameters():
#     if "layer1/layer1/0" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# only defroze the first 2 layers of student model
# for param in student.parameters():



post_linear = torch.nn.Linear(512, 512)
with torch.no_grad():
    post_linear.weight.copy_(torch.eye(512))
    post_linear.bias.zero_()

student = torch.nn.Sequential(
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

import pickle
print("Loading pre-extracted teacher embeddings...")
with open("log/teacher_embeddings_insight.pkl", "rb") as f:
    pre_loaded_teachers = pickle.load(f)
print(f"Loaded {len(pre_loaded_teachers)} teacher embeddings.")

class FaceDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        emb = pre_loaded_teachers.get(p)
        if emb is None:
            embedding_teacher = torch.zeros(512, dtype=torch.float32)
        else:
            embedding_teacher = torch.from_numpy(emb).float()
        img_s = Image.open(p).convert("RGB")
        # img_s = Image.open("dataset_processed/" + p.split("/")[-1].replace("png", "jpg")).convert("RGB")
        c_img = tf_conv(img_s)
        return p, embedding_teacher, c_img 
    


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


with open("log/train_paths.pkl", "rb") as f:
    train_paths = pickle.load(f)
with open("log/val_paths.pkl", "rb") as f:
    val_paths = pickle.load(f)
print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

paths = train_paths * 2
dataset = FaceDataset(paths)
val_dataset = FaceDataset(val_paths)

print("Dataset size:", len(dataset))
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=32, pin_memory=True)
wandb.init(project="student_distill_insight") 
epochs = 10


# for name, param in student.named_parameters():
#     if "layer1" in name or "layer2" in name or "layer3" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
# print("🚨🚨🚨🚨🚨🚨🚨 Data Gen, not normal run")
# templates = []
# for fn, embedding_teacher, conv_raw in tqdm.tqdm(loader):
#     embedding_teacher = embedding_teacher.to(device)
#     templates.append(conv_raw.detach().cpu()) 

# with open(f"log/insight_student_templates.pkl", "wb") as f:
#     pickle.dump({
#         "templates": templates
#     }, f)
    


optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=2e-2) 
scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * epochs)
os.makedirs("log", exist_ok=True)
for e in range(epochs):
    student = student.train()
    total_trip = 0
    total_cos = 0
    count = 0
    total_mae = 0
    for fn, embedding_teacher, conv_raw in tqdm.tqdm(loader): 
        embedding_teacher = embedding_teacher.to(device)
        # print(s_img.shape) 
        # img_s = data2npy.preprocess_and_return(conv_raw.to(device), 1)  # (B, 81, 112, 112) 
        s_emb = student(conv_raw.to(device)) 
        # print(embedding_teacher.shape, s_emb.shape)
        trip, cos, mae = loss_fn(s_emb, embedding_teacher.squeeze(1))
        loss = cos + mae + trip
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
            s_emb = student(conv_raw.to(device)) 
            templates.append(conv_raw.detach().cpu())
            teacher_embeddings.append(embedding_teacher.detach().cpu())
            student_embeddings.append(s_emb.detach().cpu())
            filenames.append(fn)
            trip, cos, mae = loss_fn(s_emb, embedding_teacher.squeeze(1))
            total_trip += trip.item()
            total_cos += cos.item()
            total_mae += mae.item()
            count += 1
    wandb.log({"val_epoch_triplet": total_trip/count, "val_epoch_cosine": total_cos/count, "val_epoch_mae": total_mae/count})
import pickle
with open(f"log/insight_student_embeddings_val_epoch{4}.pkl", "wb") as f:
    pickle.dump({
        "filenames": filenames,
        "student_embeddings": student_embeddings,
        "teacher_embeddings": teacher_embeddings,
        "templates": templates
    }, f)
