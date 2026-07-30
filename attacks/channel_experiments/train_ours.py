"""Train the embedding-distillation attack for one protection/channel setting."""

import argparse
import json
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
import torch.nn.functional as F
from onnx2torch import convert
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "methods", "minusface"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "tface", "recognition"))
sys.path.insert(0, os.path.join(ROOT, "attacks", "partialface"))
sys.path.insert(0, os.path.join(ROOT, "methods", "fracface"))

import data2npy
import processing_utils
from methods.fracface_fixed import data2npy as data2npy_fixed
from minusface import MinusBackbone


class DistillationDataset(Dataset):
    def __init__(self, paths, data_root, method, channel_mode):
        self.paths = paths
        self.data_root = data_root
        self.method = method
        self.channel_mode = channel_mode
        self.fixed_channel = channel_mode == "fixed"
        self.to_tensor = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        relative = os.path.relpath(path, self.data_root)
        embedding_path = os.path.join(
            self.data_root,
            "insight_embeddings",
            relative.replace("/", "_").replace(".jpg", ".npy"),
        )
        teacher = torch.from_numpy(np.load(embedding_path)).float()

        if self.method == "fracface":
            image = Image.open(path).convert("RGB")
            protected = data2npy.preprocess_and_return(
                image, 1, fixed_channel=self.fixed_channel
            )[0]
            return path, teacher, protected
        if self.method == "fracface_fixed":
            image = Image.open(path).convert("RGB")
            if self.channel_mode == "random_per_sample":
                protected = data2npy_fixed.preprocess_and_return(
                    image, 1, fixed_channel=False
                )[0]
                return path, teacher, protected
            protected = data2npy_fixed.preprocess_fcr_and_return(image)[0]
            return path, teacher, protected

        image = Image.open(path).convert("RGB")
        return path, teacher, self.to_tensor(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["fracface", "fracface_fixed", "partialface", "minusface"])
    parser.add_argument(
        "--channel-mode",
        required=True,
        choices=["fixed", "random", "random_per_sample"],
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.method == "minusface":
        assert args.channel_mode == "random"
    if args.channel_mode == "random_per_sample":
        assert args.method == "fracface_fixed"

    fixed_channel = args.channel_mode == "fixed"
    device = "cuda"
    os.makedirs(args.output, exist_ok=True)

    paths = []
    with open(os.path.join(ROOT, "data_splits", "index.txt")) as f:
        for line in f:
            filename, split = line.strip().split()
            if split == "train":
                paths.append(os.path.join(args.data_root, filename))

    dataset = DistillationDataset(
        paths, args.data_root, args.method, args.channel_mode
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    print("method=%s channel=%s train=%d" % (
        args.method, args.channel_mode, len(dataset)
    ), flush=True)

    backbone = convert(os.path.join(ROOT, "checkpoints", "model.onnx"))
    if args.method in ["fracface", "fracface_fixed"]:
        student = torch.nn.Sequential(
            torch.nn.Conv2d(81, 3, kernel_size=3, padding=1),
            backbone,
        ).to(device)
    else:
        student = backbone.to(device)

    conversion_model = None
    if args.method == "minusface":
        conversion_model = MinusBackbone(mode="stage1")
        conversion_model.load_state_dict(torch.load(
            os.path.join(ROOT, "checkpoints", "minusface_stage1.pth"),
            map_location="cpu",
        ))
        conversion_model = conversion_model.eval().to(device)

    epochs = 2
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=5e-4, weight_decay=5e-3
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * epochs)
    resume_path = os.path.join(args.output, "resume.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        assert state["method"] == args.method
        assert state["channel_mode"] == args.channel_mode
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        print("resuming at epoch %d" % start_epoch, flush=True)

    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump({
            "method": args.method,
            "channel_mode": args.channel_mode,
            "epochs": epochs,
            "batch_size": 256,
            "optimizer": "AdamW",
            "lr": 5e-4,
            "weight_decay": 5e-3,
            "scheduler": "CosineAnnealingLR",
            "train_samples": len(dataset),
        }, f, indent=2)

    for epoch in range(start_epoch, epochs):
        student.train()
        total_cosine = 0.0
        batches = 0
        progress = tqdm(loader, desc="ours %s %s epoch %d" % (
            args.method, args.channel_mode, epoch
        ))
        for filenames, teacher, attack_input in progress:
            teacher = teacher.to(device)

            if args.method == "fracface_fixed":
                if args.channel_mode != "random_per_sample":
                    attack_input = data2npy_fixed.form_training_batch_with_fractal(
                        attack_input,
                        [1] * attack_input.shape[0],
                        fixed_channel=fixed_channel,
                    )[0]
                attack_input = attack_input.to(device)
            elif args.method == "partialface":
                attack_input = attack_input.to(device)
                attack_input = processing_utils.form_training_batch(
                    attack_input,
                    [1] * attack_input.shape[0],
                    fixed_channel=fixed_channel,
                )[0]
                attack_input = attack_input.mean(
                    dim=1, keepdim=True
                ).repeat(1, 3, 1, 1)
                minimum = attack_input.amin(dim=(1, 2, 3), keepdim=True)
                maximum = attack_input.amax(dim=(1, 2, 3), keepdim=True)
                if torch.any(maximum == minimum):
                    raise RuntimeError(
                        "constant PartialFace attack input for %s" % (
                            list(filenames),
                        )
                    )
                attack_input = (
                    attack_input - minimum
                ) / (maximum - minimum)
            elif args.method == "minusface":
                with torch.no_grad():
                    attack_input = conversion_model(
                        attack_input.to(device)
                    )[5].float()
                minimum = attack_input.amin(dim=(1, 2, 3), keepdim=True)
                maximum = attack_input.amax(dim=(1, 2, 3), keepdim=True)
                if torch.any(maximum == minimum):
                    raise RuntimeError(
                        "constant MinusFace attack input for %s" % (
                            list(filenames),
                        )
                    )
                attack_input = (
                    attack_input - minimum
                ) / (maximum - minimum)
                attack_input = (attack_input - 0.5) / 0.5
            else:
                attack_input = attack_input.to(device)

            prediction = student(attack_input)
            loss = (
                1 - (
                    F.normalize(prediction, dim=1)
                    * F.normalize(teacher, dim=1)
                ).sum(dim=1)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_cosine += loss.item()
            batches += 1
            progress.set_postfix(cosine=total_cosine / batches)

        checkpoint_path = os.path.join(
            args.output, "model_epoch_%02d.pth" % (epoch + 1)
        )
        torch.save(student.state_dict(), checkpoint_path)
        with open(os.path.join(
            args.output, "metrics_epoch_%02d.json" % (epoch + 1)
        ), "w") as f:
            json.dump({
                "epoch": epoch + 1,
                "train_cosine_loss": total_cosine / batches,
                "lr": scheduler.get_last_lr()[0],
                "batches": batches,
            }, f, indent=2)
        torch.save({
            "method": args.method,
            "channel_mode": args.channel_mode,
            "epoch": epoch,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, resume_path)
        print(
            "epoch=%d cosine=%.6f checkpoint=%s" % (
                epoch + 1, total_cosine / batches, checkpoint_path
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
