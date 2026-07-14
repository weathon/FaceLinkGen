import os
import sys
import cv2
cv2.ocl.setUseOpenCL(False)
print("OpenCV built with OpenCL:", cv2.ocl.useOpenCL()) 
sys.path.append("../../methods/fracface")
import data2npy
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import tqdm
import concurrent.futures
from deepface import DeepFace
from onnx2torch import convert

# ==========================================
# 1. Setup & Seeding
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 2. Models (Standard Baseline)
# ==========================================
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature=512, hidden_size=1024):
        super(DomainDiscriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layer(x)

# Student is now just the base model (Input: 3 channels)
onnx_model_path = "../../checkpoints/model.onnx"
student = convert(onnx_model_path).to(device)

student = torch.nn.Sequential(
    student,
).to(device)
student.load_state_dict(torch.load("./student.pth", map_location='cpu'))
discriminator = DomainDiscriminator().to(device)

# ==========================================
# 3. Data Preparation
# ==========================================
embedding_root = '/path/to/casia-webface/insight_embeddings'
img_root = '/path/to/casia-webface'

paths = []
with open("../../data_splits/index.txt","r") as f:
    for line in f:
        filename, split = line.strip().split()
        if split == "train":
            paths.append(os.path.join(img_root, filename))

pre_loaded_teachers = {}
print("Preloading Teacher Embeddings...")
for path in tqdm.tqdm(paths): 
    emb_path = embedding_root + "/" + path.replace("/path/to/casia-webface/", "").replace("/", "_").replace(".jpg", ".npy")
    try:
        embedding = np.load(emb_path)[None, :]
        pre_loaded_teachers[path] = torch.from_numpy(embedding).float()
    except:
        pass

valid_paths = [p for p in paths if p in pre_loaded_teachers]
print(f"Valid Source Images: {len(valid_paths)}")

random.seed(42)
val_sample_paths = random.sample(valid_paths, 30)
train_paths = [p for p in valid_paths if p not in val_sample_paths]

print("Generating Fixed Validation Set...")
val_pairs = []

def process_val_item(path):
    try:
        # faces = DeepFace.extract_faces(path, detector_backend="opencv", enforce_detection=False)
        # if not faces: return None
        # img = Image.fromarray((faces[0]["face"] * 255).astype("uint8"))
        img = Image.open(path).convert("RGB")
        raw_target = data2npy.preprocess_and_return(img, 1)[0]
        if not isinstance(raw_target, torch.Tensor):
            raw_target = torch.from_numpy(raw_target).float()
        
        target_tensor = raw_target.mean(dim=0, keepdim=True).repeat(3, 1, 1)

        # Normalize to [-1, 1] to fix zero similarity issue
        mean_val = target_tensor.mean()
        std_val = target_tensor.std()
        if std_val > 1e-6:
            target_tensor = (target_tensor - mean_val) / std_val
        
        return {"target_img": target_tensor, "teacher_emb": pre_loaded_teachers[path]}
    except Exception as e:
        print("Error processing val item:", path, e)
        return None

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    results = list(tqdm.tqdm(executor.map(process_val_item, val_sample_paths), total=len(val_sample_paths)))
    val_pairs = [r for r in results if r is not None]

# ==========================================
# 4. Unpaired Online Dataset
# ==========================================
tf_source = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

class UnpairedOnlineDataset(Dataset):
    def __init__(self, file_paths, teacher_dict):
        self.source_paths = file_paths
        self.teacher_dict = teacher_dict
        self.target_paths = file_paths.copy()
        # random.shuffle(self.target_paths) 
        
    def __len__(self):
        return len(self.source_paths)

    def get_image(self, path):
        try:
            # faces = DeepFace.extract_faces(path, detector_backend="opencv", enforce_detection=False)
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            print("Error loading image:", path, e)
            return Image.new('RGB', (112, 112))

    def __getitem__(self, idx):
        domain_choice = random.randint(0, 1)
        
        if domain_choice == 0:
            # --- SOURCE FLOW ---
            path = self.source_paths[idx]
            img = self.get_image(path)
            
            # Standard RGB Input: (3, 112, 112)
            input_tensor = tf_source(img)
            
            gt_emb = self.teacher_dict[path]
            domain_label = torch.tensor(0, dtype=torch.float)
            
            return input_tensor, gt_emb, domain_label
            
        else:
            # --- TARGET FLOW ---
            path = self.target_paths[idx] 
            img = self.get_image(path)
            
            try:
                # 81 channels -> Collapse to 3
                raw_target = data2npy.preprocess_and_return(img, 1)[0] 
                if not isinstance(raw_target, torch.Tensor):
                    raw_target = torch.from_numpy(raw_target).float()
                input_tensor = raw_target.mean(dim=0, keepdim=True).repeat(3, 1, 1)

                # Normalize to [-1, 1] to fix zero similarity issue
                mean_val = input_tensor.mean()
                std_val = input_tensor.std() 
                if std_val > 1e-6:
                    input_tensor = (input_tensor - mean_val) / std_val

            except Exception as e:
                print("Error processing target image:", path, e)
                input_tensor = torch.zeros(3, 112, 112)

            dummy_emb = torch.zeros(1, 512)
            domain_label = torch.tensor(1, dtype=torch.float)
            
            return input_tensor, dummy_emb, domain_label

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

dataset = UnpairedOnlineDataset(train_paths, pre_loaded_teachers)
loader = DataLoader(dataset, batch_size=512, shuffle=True, 
                    num_workers=64, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)

# ==========================================
# 5. Training
# ==========================================
def cosine_sim_loss(s, t):
    s = F.normalize(s, dim=1)
    t = F.normalize(t, dim=1)
    return (1 - (s * t).sum(dim=1)).mean()

bce_loss = nn.BCEWithLogitsLoss()

wandb.init(project="student_distill_insight_dat_avg")
epochs = 5
optimizer = torch.optim.AdamW([
    {'params': student.parameters(), 'lr': 1e-4},
    {'params': discriminator.parameters(), 'lr': 2e-4} 
], weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * epochs)

path = train_paths[0]
img = Image.open(path).convert("RGB")
input_tensor = tf_source(img).unsqueeze(0).to(device) # (1, 3, 112, 112)


student.eval()
with torch.no_grad():
    student_emb = student(input_tensor)
    teacher_emb = pre_loaded_teachers[path].to(device)

sim = F.cosine_similarity(student_emb, teacher_emb).item()
print(f"Initial Cosine Similarity: {sim:.4f}")

for name, param in student.named_parameters():
    if "layer1" in name or "layer2" in name or "layer3" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print("Start Training...")
for e in range(epochs):
    student.train()
    discriminator.train()
    
    pbar = tqdm.tqdm(loader)
    
    for i, (inputs, gt_embs, domains) in enumerate(pbar):
        # inputs is now always (B, 3, 112, 112) for both source and target
        inputs = inputs.to(device).float()
        gt_embs = gt_embs.to(device).squeeze(1)
        domains = domains.to(device).unsqueeze(1)
        
        # 1. Forward Pass (No domain input needed anymore)
        features = student(inputs)
        
        # 2. Alignment Loss (Source Only)
        source_mask = (domains.squeeze() == 0)
        align_loss = torch.tensor(0.0, device=device)
        
        if source_mask.sum() > 0:
            s_feats_src = features[source_mask]
            t_feats_src = gt_embs[source_mask]
            align_loss = cosine_sim_loss(s_feats_src, t_feats_src)
        
        # 3. Adversarial Loss (All Samples)
        # p = float(e * len(loader) + pbar.n) / (epochs * len(loader))
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1.

        reversed_feat = grad_reverse(features, 1)
        domain_preds = discriminator(reversed_feat)
        adv_loss = bce_loss(domain_preds, domains)
        
        # 4. Total Loss
        loss = align_loss + adv_loss * 0.4
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        wandb.log({
            "train/align_loss": align_loss.item(),
            "train/adv_loss": adv_loss.item(),
            "lr": scheduler.get_last_lr()[0]
        })
        pbar.set_description(f"E{e} Align:{align_loss.item():.4f} Adv:{adv_loss.item():.4f}")

        if i % 30 == 0:
            # 5. Validation using the fixed set
            student.eval()
            val_sims = []
            
            with torch.no_grad():
                for pair in val_pairs:
                    t_img = pair["target_img"].unsqueeze(0).to(device) # Now (1, 3, 112, 112)
                    teacher_emb = pair["teacher_emb"].to(device)
                    
                    s_emb = student(t_img)
                    
                    sim = F.cosine_similarity(s_emb, teacher_emb)
                    val_sims.append(sim.item())
            student.train()
            assert len(val_sims) != 0
            avg_val_sim = sum(val_sims) / len(val_sims) if val_sims else 0
            wandb.log({"val/avg_cosine_sim": avg_val_sim, "epoch": e})
            print(f"Epoch {e} Validation Sim: {avg_val_sim:.4f}")
    
    torch.save(student.state_dict(), f"../student_epoch_{e}.pth")
