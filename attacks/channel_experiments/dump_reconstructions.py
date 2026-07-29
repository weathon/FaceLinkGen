"""Write the 300 paired inputs for Arc2Face or direct U-Net reconstructions."""

import argparse
import json
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
from onnx2torch import convert
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "methods", "minusface"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "tface", "recognition"))
sys.path.insert(0, os.path.join(ROOT, "attacks", "partialface"))
sys.path.insert(0, os.path.join(ROOT, "methods", "fracface"))

import data2npy
import processing_utils
from minusface import MinusBackbone


class EvaluationDataset(Dataset):
    def __init__(self, paths, method, fixed_channel):
        self.paths = paths
        self.method = method
        self.fixed_channel = fixed_channel
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
            return idx, path, protected, raw
        return idx, path, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=["ours", "unet"])
    parser.add_argument("--method", required=True, choices=["fracface", "partialface", "minusface"])
    parser.add_argument("--channel-mode", required=True, choices=["fixed", "random"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.method == "minusface":
        assert args.channel_mode == "random"

    fixed_channel = args.channel_mode == "fixed"
    device = "cuda"
    os.makedirs(args.output, exist_ok=True)

    paths = []
    with open(os.path.join(
        ROOT, "data_splits", "val_minus_lfw_filenames.txt"
    )) as f:
        for line in f:
            path = line.strip().replace(
                "/path/to/casia-webface", args.data_root
            )
            paths.append(path)
    paths = paths[:300]

    manifest_path = os.path.join(args.output, "manifest.jsonl")
    completed = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                record = json.loads(line)
                completed.add(record["index"])
                assert os.path.exists(record["output"])

    pending_paths = [
        path for idx, path in enumerate(paths) if idx not in completed
    ]
    pending_indices = [
        idx for idx in range(len(paths)) if idx not in completed
    ]
    dataset = EvaluationDataset(
        pending_paths, args.method, fixed_channel
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    print("attack=%s method=%s channel=%s pending=%d total=%d" % (
        args.attack,
        args.method,
        args.channel_mode,
        len(dataset),
        len(paths),
    ), flush=True)

    if args.attack == "ours":
        backbone = convert(os.path.join(ROOT, "checkpoints", "model.onnx"))
        if args.method == "fracface":
            model = torch.nn.Sequential(
                torch.nn.Conv2d(81, 3, kernel_size=3, padding=1),
                backbone,
            ).to(device)
        else:
            model = backbone.to(device)
    else:
        input_channels = {
            "fracface": 81,
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

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    conversion_model = None
    if args.method == "minusface":
        conversion_model = MinusBackbone(mode="stage1")
        conversion_model.load_state_dict(torch.load(
            os.path.join(ROOT, "checkpoints", "minusface_stage1.pth"),
            map_location="cpu",
        ))
        conversion_model = conversion_model.eval().to(device)

    pending_offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="dump"):
            if args.method == "fracface":
                _, source_paths, attack_input, raw = batch
                attack_input = attack_input.to(device)
                raw = raw.to(device)
            else:
                _, source_paths, raw = batch
                raw = raw.to(device)
                if args.method == "partialface":
                    attack_input = processing_utils.form_training_batch(
                        raw,
                        [1] * raw.shape[0],
                        fixed_channel=fixed_channel,
                    )[0]
                    if args.attack == "ours":
                        attack_input = attack_input.mean(
                            dim=1, keepdim=True
                        ).repeat(1, 3, 1, 1)
                        minimum = attack_input.amin(
                            dim=(1, 2, 3), keepdim=True
                        )
                        maximum = attack_input.amax(
                            dim=(1, 2, 3), keepdim=True
                        )
                        if torch.any(maximum == minimum):
                            raise RuntimeError(
                                "constant PartialFace attack input for %s" % (
                                    list(source_paths),
                                )
                            )
                        attack_input = (
                            attack_input - minimum
                        ) / (maximum - minimum)
                else:
                    attack_input = conversion_model(raw)[5].float()
                    minimum = attack_input.amin(
                        dim=(1, 2, 3), keepdim=True
                    )
                    maximum = attack_input.amax(
                        dim=(1, 2, 3), keepdim=True
                    )
                    if torch.any(maximum == minimum):
                        raise RuntimeError(
                            "constant MinusFace attack input for %s" % (
                                list(source_paths),
                            )
                        )
                    attack_input = (
                        attack_input - minimum
                    ) / (maximum - minimum)
                    attack_input = (attack_input - 0.5) / 0.5

            outputs = model(attack_input).detach().cpu()
            for batch_idx in range(outputs.shape[0]):
                original_index = pending_indices[pending_offset + batch_idx]
                source = source_paths[batch_idx]
                if args.attack == "ours":
                    output_path = os.path.join(
                        args.output, "%04d_embedding.npy" % original_index
                    )
                    np.save(output_path, outputs[batch_idx].numpy())
                else:
                    output_path = os.path.join(
                        args.output, "%04d.png" % original_index
                    )
                    array = (
                        outputs[batch_idx]
                        .clamp(0, 1)
                        .permute(1, 2, 0)
                        .numpy()
                        * 255
                    ).astype("uint8")
                    Image.fromarray(array).save(output_path)

                with open(manifest_path, "a") as f:
                    f.write(json.dumps({
                        "index": original_index,
                        "source": source,
                        "output": output_path,
                    }) + "\n")
            pending_offset += outputs.shape[0]

    assert pending_offset == len(pending_paths)
    print("wrote %d records to %s" % (len(paths), manifest_path), flush=True)


if __name__ == "__main__":
    main()
