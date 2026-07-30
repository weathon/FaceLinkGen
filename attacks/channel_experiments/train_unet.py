"""Train the U-Net reconstruction baseline for one protection/channel setting."""

import argparse
import json
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
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


class ReconstructionDataset(Dataset):
    def __init__(self, paths, method, channel_mode):
        self.paths = paths
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
        image = Image.open(path).convert("RGB")
        raw = self.to_tensor(image)
        if self.method == "fracface":
            protected = data2npy.preprocess_and_return(
                image, 1, fixed_channel=self.fixed_channel
            )[0]
            return path, protected, raw
        if self.method == "fracface_fixed":
            if self.channel_mode == "random_per_sample":
                protected = data2npy_fixed.preprocess_and_return(
                    image, 1, fixed_channel=False
                )[0]
                return path, protected, raw
            protected = data2npy_fixed.preprocess_fcr_and_return(image)[0]
            return path, protected, raw
        return path, raw


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
    parser.add_argument("--epochs", required=True, type=int)
    args = parser.parse_args()

    assert args.epochs >= 5
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

    dataset = ReconstructionDataset(paths, args.method, args.channel_mode)
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

    input_channels = {
        "fracface": 81,
        "fracface_fixed": 81,
        "partialface": 27,
        "minusface": 3,
    }[args.method]
    unet_repository = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "torch",
        "hub",
        "mateuszbuda_brain-segmentation-pytorch_master",
    )
    model = torch.hub.load(
        unet_repository,
        "unet",
        source="local",
        in_channels=input_channels,
        out_channels=3,
        init_features=3,
        pretrained=False,
    ).to(device)

    conversion_model = None
    if args.method == "minusface":
        conversion_model = MinusBackbone(mode="stage1")
        conversion_model.load_state_dict(torch.load(
            os.path.join(ROOT, "checkpoints", "minusface_stage1.pth"),
            map_location="cpu",
        ))
        conversion_model = conversion_model.eval().to(device)

    epochs = args.epochs
    constant_epochs = epochs - 5
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=5e-3
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(loader) * (epochs - constant_epochs)
    )
    resume_path = os.path.join(args.output, "resume.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        assert state["method"] == args.method
        assert state["channel_mode"] == args.channel_mode
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        print("resuming at epoch %d" % start_epoch, flush=True)

    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump({
            "method": args.method,
            "channel_mode": args.channel_mode,
            "epochs": epochs,
            "constant_epochs": constant_epochs,
            "cosine_epochs": epochs - constant_epochs,
            "batch_size": 256,
            "optimizer": "AdamW",
            "lr": 5e-4,
            "weight_decay": 5e-3,
            "loss": "L1",
            "train_samples": len(dataset),
        }, f, indent=2)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_l1 = 0.0
        batches = 0
        progress = tqdm(loader, desc="unet %s %s epoch %d" % (
            args.method, args.channel_mode, epoch
        ))
        for batch in progress:
            if args.method in ["fracface", "fracface_fixed"]:
                _, protected, raw = batch
                if (
                    args.method == "fracface_fixed"
                    and args.channel_mode != "random_per_sample"
                ):
                    protected = data2npy_fixed.form_training_batch_with_fractal(
                        protected,
                        [1] * protected.shape[0],
                        fixed_channel=fixed_channel,
                    )[0]
                protected = protected.to(device)
                raw = raw.to(device)
            else:
                filenames, raw = batch
                raw = raw.to(device)
                if args.method == "partialface":
                    protected = processing_utils.form_training_batch(
                        raw,
                        [1] * raw.shape[0],
                        fixed_channel=fixed_channel,
                    )[0]
                else:
                    with torch.no_grad():
                        protected = conversion_model(raw)[5].float()
                    minimum = protected.amin(dim=(1, 2, 3), keepdim=True)
                    maximum = protected.amax(dim=(1, 2, 3), keepdim=True)
                    if torch.any(maximum == minimum):
                        raise RuntimeError(
                            "constant MinusFace U-Net input for %s" % (
                                list(filenames),
                            )
                        )
                    protected = (
                        protected - minimum
                    ) / (maximum - minimum)
                    protected = (protected - 0.5) / 0.5

            prediction = model(protected)
            loss = torch.nn.functional.l1_loss(prediction, raw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch >= constant_epochs:
                scheduler.step()

            total_l1 += loss.item()
            batches += 1
            progress.set_postfix(l1=total_l1 / batches)

        checkpoint_path = os.path.join(
            args.output, "model_epoch_%02d.pth" % (epoch + 1)
        )
        torch.save(model.state_dict(), checkpoint_path)
        with open(os.path.join(
            args.output, "metrics_epoch_%02d.json" % (epoch + 1)
        ), "w") as f:
            json.dump({
                "epoch": epoch + 1,
                "train_l1": total_l1 / batches,
                "lr": optimizer.param_groups[0]["lr"],
                "batches": batches,
            }, f, indent=2)
        torch.save({
            "method": args.method,
            "channel_mode": args.channel_mode,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, resume_path)
        print(
            "epoch=%d l1=%.6f lr=%.8f checkpoint=%s" % (
                epoch + 1,
                total_l1 / batches,
                optimizer.param_groups[0]["lr"],
                checkpoint_path,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
