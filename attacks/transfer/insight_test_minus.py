
import os
import sys
sys.path.append("../partialface")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FORCE_GPU", "2")
import sys
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone

# "val" uses teacher embeddings; "test" has no teacher but identical preprocessing.
DATASET_MODE = "val"
USE_TEACHER = True
TRAIN_VAL_SAMPLE = 30
EVAL_VAL_SAMPLE = 300

conversion_model = MinusBackbone(mode='stage1')
conversion_model.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
conversion_model = conversion_model.eval().to(device)
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
import torchvision.transforms.functional as TF

import cv2
import torch
import torch
import torch.nn.functional as F
import random




def _auto_kernel_from_sigma(sigma):
    # 3-sigma rule for an odd kernel size.
    return int(2 * round(3 * sigma) + 1)


def highpass(img, strength=2.0, kernel_size=9, sigma=None):
    if kernel_size is None:
        if sigma is None:
            kernel_size = 5
        else:
            kernel_size = _auto_kernel_from_sigma(sigma)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if sigma is None:
        blurred = TF.gaussian_blur(img, (kernel_size, kernel_size))
    else:
        blurred = TF.gaussian_blur(img, (kernel_size, kernel_size), sigma=[sigma, sigma])
    highpass_img = img - blurred
    highpass_img = highpass_img * strength
    return highpass_img

def zscore_batch(imgs, eps=1e-6, clamp_val=5.0):
    mean = imgs.mean(dim=(1, 2, 3), keepdim=True)
    std = imgs.std(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - mean) / (std + eps)
    return imgs.clamp(-clamp_val, clamp_val)

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


tf_conv = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])

embedding_root = '/path/to/casia-webface/insight_embeddings'
pre_loaded_teachers = {}
import sys

class FaceDataset(Dataset):
    def __init__(self, paths, split, has_teacher):
        self.paths = paths
        self.split = split
        self.has_teacher = has_teacher

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img_s = Image.open(p).convert("RGB").resize((112,112))
        c_img = tf_conv(img_s)
        c_img = c_img.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        if self.has_teacher:
            embedding_teacher = pre_loaded_teachers[p]
            return p, embedding_teacher, c_img
        return p, c_img
    


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

has_teacher = DATASET_MODE == "val" and USE_TEACHER
rng = random.Random(42)
if DATASET_MODE == "val":
    val_paths = []
    root = '/path/to/casia-webface'
    with open("../../data_splits/index.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            filename, split = line.strip().split()
            if split != "train":
                val_paths.append(os.path.join(root, filename))
    train_val_paths = rng.sample(val_paths, k=TRAIN_VAL_SAMPLE)
    remaining_val_paths = [p for p in val_paths if p not in set(train_val_paths)]
    eval_val_paths = rng.sample(remaining_val_paths, k=EVAL_VAL_SAMPLE)
    dataset = FaceDataset(eval_val_paths, split="val", has_teacher=has_teacher)
    if has_teacher:
        for path in tqdm.tqdm(eval_val_paths, desc="Preloading embeddings"):
            embedding = np.load(
                embedding_root + "/" + path.replace("/path/to/casia-webface/","").replace("/", "_").replace(".jpg", ".npy")
            )[None, :]
            pre_loaded_teachers[path] = torch.from_numpy(embedding).float()
else:
    # test_paths = os.listdir("../training-tpdne")
    # test_paths = [os.path.join("../training-tpdne", p) for p in test_paths]
    # test_paths = sorted(test_paths)
    # rng.shuffle(test_paths)
    paths = os.listdir("/path/to/lfw/lfw_112x112/")
    paths = [os.path.join("/path/to/lfw/lfw_112x112/", p) for p in paths]
    # dataset = FaceDataset(test_paths[:300], split="test", has_teacher=has_teacher)
    dataset = FaceDataset(paths, split="test", has_teacher=has_teacher)  

print("Dataset size:", len(dataset))
loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=32, pin_memory=True)


# def convert_batch(conv_raw):
#     conv_raw = conv_raw.to(device)
#     with torch.no_grad():
#         out = conversion_model(conv_raw)[5] 

#     imgs = out.float()
#     print(imgs.shape)
#     # imgs = F.interpolate(imgs, size=(112, 112), mode="bilinear"
#     imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-6)
#     Image.fromarray((imgs[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")).save("raw_debug.png")
    
#     imgs_new = []
#     for img in imgs:
#         img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
#         img = np.array(img.cpu()).astype(np.uint8)
#         img = torch.tensor(img).float() / 255
#         img = img.contiguous()
#         img = img.mean(0, keepdim=True).repeat(3,1,1)
#         img_freq = dct_transform(img.unsqueeze(0))#
        
#         # img = cv2.imread("raw_debug.png")
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # img = torch.tensor(img) 
#         # img = (img - img.min()) / (img.max() - img.min()) 
#         # img_freq = dct_transform(img.unsqueeze(0).permute(0,3,1,2).float())
#         img_freq = (img_freq - img_freq.min()) / (img_freq.max() - img_freq.min() + 1e-6)
#         img_freq = img_freq.mean(1, keepdim=True).repeat(1,3,1,1) # mean at last
#         img_freq = (img_freq - img_freq.min()) / (img_freq.max() - img_freq.min() + 1e-6)

#         imgs_new.append(img_freq)
#     imgs = imgs_new
#     imgs = torch.cat(imgs, dim=0) 
    
#     # minv = imgs.amin(dim=(1, 2, 3), keepdim=True) 
#     # maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
#     # imgs = (imgs - minv) / (maxv - minv + 1e-6)  
#     Image.fromarray((imgs[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")).save("debug_minus.png")
#     # quit()
#     return imgs
tf_conv = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor()
])


def convert_batch(conv_raw):
    conv_raw = conv_raw.to(device)
    with torch.no_grad():
        out = conversion_model(conv_raw)[5]

    imgs = out.float()
    imgs = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    imgs = (imgs - minv) / (maxv - minv + 1e-6)
    imgs = (imgs - 0.5) / 0.5
    return imgs

student = student.eval()
student.load_state_dict(torch.load(os.environ.get("CKPT", "student.pth"), map_location='cpu'))
print("Validation on test set")
total_trip = 0
total_cos = 0
count = 0
total_mae = 0
teacher_embeddings = [] if has_teacher else None
student_embeddings = []
filenames = []
templates = []
import pickle
with torch.no_grad():
    if has_teacher:
        for fn, embedding_teacher, conv_raw in tqdm.tqdm(loader):
            conv_raw = convert_batch(conv_raw)
            conv_raw = highpass(conv_raw)
            embedding_teacher = embedding_teacher.to(device)
            conv_raw = zscore_batch(conv_raw.to(device))
            s_emb = student(conv_raw) 
            templates.append(conv_raw.detach().cpu())
            teacher_embeddings.append(embedding_teacher.detach().cpu())
            student_embeddings.append(s_emb.detach().cpu())
            filenames.append(fn)
            trip, cos, mae = loss_fn(s_emb, embedding_teacher.squeeze(1))
            total_trip += trip.item()
            total_cos += cos.item()
            total_mae += mae.item()
            count += 1
    else:
        for fn, conv_raw in tqdm.tqdm(loader):
            conv_raw = convert_batch(conv_raw)
            conv_raw = highpass(conv_raw)
            conv_raw = zscore_batch(conv_raw.to(device))
            s_emb = student(conv_raw) 
            templates.append(conv_raw.detach().cpu())
            student_embeddings.append(s_emb.detach().cpu())
            filenames.append(fn)

if has_teacher and count > 0:
    print(f"RESULT minus cosine_loss={total_cos/count:.4f} triplet={total_trip/count:.4f} mae={total_mae/count:.4f} n_batches={count}")

variant = "minus"
output_path = f"{DATASET_MODE}_{variant}_lfw.pkl"
with open(output_path, "wb") as f:
    pickle.dump({
        "filenames": filenames,
        "student_embeddings": student_embeddings,
        "teacher_embeddings": teacher_embeddings,
        "templates": templates
    }, f)
