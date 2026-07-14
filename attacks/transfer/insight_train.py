import os
import cv2
cv2.ocl.setUseOpenCL(False)
print("OpenCV built with OpenCL:", cv2.ocl.useOpenCL()) 
import sys
sys.path.append("../../methods/fracface")
import data2npy
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


import torch
import torch.nn.functional as F
import random
from torchjpeg import dct

def dct_transform(x, chs_remove=None, chs_pad=False):
    assert x.shape[1] == 3
    size = random.choice([4, 8, 16]); stride = size; pad = 0; dilation = 1; ratio = size

    x = x * 0.5 + 0.5

    # up-sample
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)

    # convert to the YCbCr color domain, required by DCT
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    # perform block discrete cosine transform (BDCT)
    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)
    chs_remove = list(range(random.randint(1, x_freq.shape[2]-1))) 
    channels = list(set([i for i in range(x_freq.shape[2])]) - set(chs_remove)) 
    # channels = random.sample(channels, random.randint(min(1, len(channels)), len(channels)))
    x_freq = x_freq[:, :, channels, :, :] 

    x_freq = x_freq.reshape(b, -1, n_block, n_block)
    x_freq = x_freq.mean(dim=1)
    # noise = torch.randn_like(x_freq) * x_freq.std() * 0.1
    # x_freq = x_freq + noise
    return x_freq




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
# for name, param in student.named_parameters():
#     if "layer1" in name or "layer2" in name or "layer3" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# only defroze the first 2 layers of student model
# for param in student.parameters():


import torchvision.transforms as T

bw_aug = T.Compose([
    # T.RandomApply([
    #     T.ColorJitter( 
    #         brightness=0.2,
    #     )
    # ], p=0.8),
    T.RandomAutocontrast(p=1.0),
    T.RandomAdjustSharpness(sharpness_factor=3.0, p=0.4),
    T.RandomApply([
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
    ], p=0.4),
    # T.RandomResizedCrop((112,112), scale=(0.9, 1.1), ratio=(0.95, 1.05))
])


# test_t = T.Compose([
#     T.RandomAutocontrast(p=1.0),
# ])
 


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


from deepface import DeepFace
import torchvision.transforms.functional as TF

def highpass(img):
    blurred = TF.gaussian_blur(img, (5, 5))
    highpass_img = img - blurred 
    return highpass_img

import torchvision.transforms.functional as TF

embedding_root = '/path/to/casia-webface/insight_embeddings'
pre_loaded_teachers = {}
class FaceDataset(Dataset):
    def __init__(self, paths, split):
        self.paths = paths
        self.split = split

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        # print("Loading embedding from:", p_t)
        embedding_teacher = pre_loaded_teachers[p]  # (1, 512)
        # try:
        # except Exception as e:
        #     print("Error extracting face for:", p, e)
        img_s = DeepFace.extract_faces(p, detector_backend = "opencv", enforce_detection=False)[0]["face"]
        img_s = Image.fromarray((img_s * 255).astype("uint8")).resize((112,112))
        # img_s = Image.open(p).convert("RGB")

        if self.split == "train":
            c_img = torch.tensor(np.array(img_s)).permute(2,0,1).float() / 255.0
            c_img = dct_transform(c_img[None]).mean(0, keepdim=True).repeat(3,1,1)
            c_img = bw_aug(c_img)
            # c_img *= random.random() * 2
            t = random.uniform(50, 100)

        if self.split != "train":
            c_img = cache_templates[p]
            c_img = c_img.reshape(-1, 112, 112)
            c_img = c_img.mean(0, keepdim=True).repeat(3,1,1)
            # c_img = test_t(c_img)
            t = random.uniform(50, 100)
            
        c_img = (c_img - c_img.mean()) / (c_img.std() + 1e-8)
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

paths = []
root = '/path/to/casia-webface'
with open("../../data_splits/index.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        filename, split = line.strip().split()
        if split == "train":
            paths.append(os.path.join(root, filename))

paths = paths * 5
dataset = FaceDataset(paths, split="train")


val_paths = []
root = '/path/to/casia-webface'
with open("../../data_splits/index.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        filename, split = line.strip().split()
        if split != "train":
            val_paths.append(os.path.join(root, filename)) 


random.seed(42)
val_paths = random.sample(val_paths, k=30)
val_dataset = FaceDataset(val_paths, split="val")  

cache_templates = {}
for i in tqdm.tqdm(range(len(val_paths)), desc="Preloading cache_templates"):
    p = val_paths[i]
    img_s = DeepFace.extract_faces(p, detector_backend = "opencv", enforce_detection=False)[0]["face"]
    img_s = Image.fromarray((img_s * 255).astype("uint8")).resize((112,112))
    img_s = Image.open(p).convert("RGB")
    template = data2npy.preprocess_and_return(img_s, 1)[0]  
    cache_templates[p] = template

import pickle
for path in tqdm.tqdm(val_paths, desc="Preloading embeddings"):
    embedding = np.load(embedding_root+"/"+path.replace("/path/to/casia-webface/","").replace("/", "_").replace(".jpg", ".npy"))[None,:]
    pre_loaded_teachers[path] = torch.from_numpy(embedding).float()


import pickle
for path in tqdm.tqdm(paths, desc="Preloading embeddings"):
    embedding = np.load(embedding_root+"/"+path.replace("/path/to/casia-webface/","").replace("/", "_").replace(".jpg", ".npy"))[None,:]
    pre_loaded_teachers[path] = torch.from_numpy(embedding).float()

print("Dataset size:", len(dataset))
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True, num_workers=32, pin_memory=True)
wandb.init(project="student_distill_insight") 
epochs = 5
lr = 3e-5
wd = 0.0005 
optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)
print("learning rate:", lr, " weight decay:", wd)
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
        # conv_raw = util.form_training_batch(conv_raw.to(device), [1] * len(conv_raw))[0]
        # conv_raw = conv_raw.mean(1, keepdim=True).repeat(1,3,1,1)
        # conv_raw = (conv_raw - conv_raw.amin(dim=(1,2,3), keepdim=True)) / (conv_raw.amax(dim=(1,2,3), keepdim=True) - conv_raw.amin(dim=(1,2,3), keepdim=True) + 1e-8)
        # conv_raw = dct_transform(conv_raw.to(device))
        s_emb = student(conv_raw.to(device)) 
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
        conv_raw_disp = (conv_raw - conv_raw.min()) / (conv_raw.max() - conv_raw.min() + 1e-8)
        wandb.log({"triplet": trip.item(),
                   "cosine": cos.item(),
                   "mae": mae.item(),
                   "lr": scheduler.get_last_lr()[0],
                   "input": wandb.Image(conv_raw_disp[0].permute(1,2,0).cpu().numpy())
                   })
        if count % 50 == 0:
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
                    conv_raw_disp = (conv_raw - conv_raw.min()) / (conv_raw.max() - conv_raw.min() + 1e-8)
                    wandb.log({"val_input": wandb.Image(conv_raw_disp[0].permute(1,2,0).cpu().numpy())})     
            wandb.log({"val_epoch_triplet": total_trip/count, "val_epoch_cosine": total_cos/count, "val_epoch_mae": total_mae/count})
    torch.save(student.state_dict(), "student.pth")
    wandb.log({"epoch_triplet": total_trip/count, "epoch_cosine": total_cos/count, "epoch_mae": total_mae/count})

with open(f"log/insight_student_embeddings_val_epoch{4}.pkl", "wb") as f:
    pickle.dump({
        "filenames": filenames,
        "student_embeddings": student_embeddings,
        "teacher_embeddings": teacher_embeddings,
        "templates": templates
    }, f)
