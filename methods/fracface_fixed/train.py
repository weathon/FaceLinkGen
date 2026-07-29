import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from models.arc_margin import ArcMarginProduct
from torchkit.backbone import get_model
from datasets.fractal_dataset import FractalDataset
from models.attention import FSMSelectAttention
import os
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import gc


class FracFaceModel(nn.Module):
    def __init__(self, num_classes, use_unet=True, generator=UNet, recognizer=None):
        super().__init__()
        self.use_unet = use_unet

        if use_unet:
            self.generator = generator(in_channels=81, out_channels=3)
            self.recognizer = get_model('ir_50', in_channels=3)([112, 112]) if recognizer is None else recognizer
        else:
            self.fsm_attention = FSMSelectAttention(channels=81, reduction=9)
            self.recognizer = get_model('ir_50', input_channel=81, input_size=(112, 112)) if recognizer is None else recognizer

        self.arc_margin = ArcMarginProduct(512, num_classes, s=16.0, m=0.3)

    def forward(self, x, labels):
        x = x.to(x.device)
        labels = labels.to(x.device)

        if self.use_unet:
            x_FracGen = self.generator(x)
            embeddings = self.recognizer(x_FracGen)
            logits = self.arc_margin(F.normalize(embeddings, p=2, dim=1), labels)
            return logits, embeddings, x_FracGen
        else:
            x_attn = self.fsm_attention(x)
            embeddings = self.recognizer(x_attn)
            logits = self.arc_margin(F.normalize(embeddings, p=2, dim=1), labels)
            return logits, embeddings


def evaluate(model, val_loader, device, writer=None, epoch=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            dummy_labels = torch.zeros_like(labels).to(device)
            output = model(images, dummy_labels)
            logits = output[0] if isinstance(output, tuple) else output
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"[Eval] Accuracy: {acc:.2f}%")
    if writer is not None and epoch is not None:
        writer.add_scalar("Val/Accuracy", acc, epoch)

    gc.collect()  
    torch.cuda.empty_cache()


class FracFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.recognition_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.recognition_loss(logits, labels)


def train_epoch(model, train_loader, optimizer, criterion, epoch, scaler, device, writer=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=True):
            output = model(images, labels)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if writer is not None:
            writer.add_scalar("Train/Loss", total_loss / (i + 1), epoch * len(train_loader) + i)
            writer.add_scalar("Train/Accuracy", 100 * correct / total, epoch * len(train_loader) + i)

    gc.collect()  
    torch.cuda.empty_cache()

    return total_loss / len(train_loader), 100 * correct / total


def save_checkpoint(model, optimizer, epoch, save_dir, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    filename = "best_model.pth" if best else f"checkpoint_epoch_{epoch}.pth"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    print(f" Saved checkpoint to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        'batch_size': ..., # input the batch size
        'lr': ..., # input the learning rate
        'epochs': ..., # input the epochs
        'weight_decay': 1e-4,
        'eval_freq': ...,
        'save_freq': 5,
        'num_workers': ...,  # input the num_workwes, which depend on your GPU used
        'pin_memory': True,
        'persistent_workers': False,
        'prefetch_factor': ...,
        'drop_last': False,
        'use_unet': False
    }

    log_dir = "./tensorboard"
    writer = SummaryWriter(log_dir=log_dir)

    train_dataset = FractalDataset(
        index_file='/path/to/your/index.txt',
        dataset_root='/path/to/your/dataset',
        remap_labels=True
    )

    val_dataset = FractalDataset(
        index_file='/path/to/your/index.txt',
        dataset_root='/path/to/your/dataset',
        remap_labels=False,
        label_map=train_dataset.label_map
    )

    num_classes = train_dataset.num_classes
    model = FracFaceModel(num_classes=num_classes, use_unet=config['use_unet']).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = FracFaceLoss().to(device)
    scaler = GradScaler()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=config['prefetch_factor'],
        drop_last=config['drop_last']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'] // 2,
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=int(config['prefetch_factor'] // 1.5),
        drop_last=config['drop_last']
    )

    
    model.eval()
    import time
    images, labels = next(iter(train_loader))

    start = time.time()
    images = images.to(device)
    labels = labels.to(device)

    start = time.time()
    with autocast(enabled=True):
        logits = model(images, labels)[0]

    start = time.time()
    loss = criterion(logits, labels)
    loss.backward()

    model.train()

    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, scaler, device, writer)

        if (epoch + 1) % config['eval_freq'] == 0:
            evaluate(model, val_loader, device, writer, epoch)

        scheduler.step()

        if (epoch + 1) % config['save_freq'] == 0:
            save_checkpoint(model, optimizer, epoch + 1, save_dir="./checkpoints")

        gc.collect()
        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    main()
