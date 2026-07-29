import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchjpeg import dct
from torch.nn import functional as F

def dct_transform(x: torch.Tensor, chs_remove=None, chs_pad=False,
                  size=8, stride=8, pad=0, dilation=1, ratio=8) -> torch.Tensor:
    """
    Transform a spatial image into frequency domain (DCT).

    Args:
        x: Tensor of shape [B, 3, H, W], RGB image.
        chs_remove: List of channel indices to prune.
        chs_pad: Whether to keep shape and pad removed channels with zeros.
        size, stride, pad, dilation: DCT block config.
        ratio: Upsample ratio.

    Returns:
        Tensor of shape [B, C, H', W'] where C is frequency channels.
    """
    assert x.shape[1] == 3, "Input must be RGB image of shape [B, 3, H, W]"
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)
    x = dct.to_ycbcr(x * 255) - 128 

    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=stride)
    x = x.transpose(1, 2).view(b, c, -1, size, size)
    x_freq = dct.block_dct(x).view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    if chs_remove is not None:
        if chs_pad:
            for channel in chs_remove:
                plane, frequency = divmod(channel, 64)
                x_freq[:, plane, frequency] = 0
        else:
            kept_planes = []
            for plane in range(3):
                removed = {
                    channel - plane * 64
                    for channel in chs_remove
                    if plane * 64 <= channel < (plane + 1) * 64
                }
                kept = sorted(set(range(64)) - removed)
                kept_planes.append(x_freq[:, plane, kept])
            x_freq = torch.stack(kept_planes, dim=2)

    return x_freq.reshape(b, -1, n_block, n_block)


def create_square_subsets(x: torch.Tensor, chs_prune_per_layer=None):
    """
    Divide frequency channels into two 9x9 square subsets (81 each).

    Args:
        x: Tensor [B, 3, H, W]
        chs_prune_per_layer: List of pruned channels per layer (default: low-freq)

    Returns:
        part1, part2: [B, 81, H', W']
    """
    if chs_prune_per_layer is None:
        chs_prune_per_layer = [
            [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],
            [64, 65, 66, 67, 71, 72, 73, 80, 81, 88],
            [128, 129, 130, 131, 136, 137, 138, 144, 145, 152]
        ]
    pruned = [i for layer in chs_prune_per_layer for i in layer]
    x_freq = dct_transform(x, chs_remove=pruned, chs_pad=False)
    part1_indices, part2_indices = generate_snake_indices()
    return x_freq[:, part1_indices], x_freq[:, part2_indices]


def generate_snake_indices():
    """
    Generate snake-like scanning indices for 18x9=162 DCT channels.

    Returns:
        part1_indices, part2_indices: Lists of length 81
    """
    size = 9
    indices = np.zeros((size * 2, size), dtype=int)
    for row in range(size * 2):
        if row % 2 == 0:
            indices[row] = np.arange(row * size, (row + 1) * size)
        else:
            indices[row] = np.arange((row + 1) * size - 1, row * size - 1, -1)
    flat = indices.flatten()
    return flat[:81].tolist(), flat[81:].tolist()


def generate_fsm():
    """
    Generate the paper-aligned depth-2 fractal index matrix.

    Returns:
        A 9x9 channel index matrix.
    """
    M0 = np.random.randint(1, 10, size=(3, 3))
    L0 = np.random.permutation(9).reshape(3, 3)
    M0 = M0.flatten()[L0.flatten()].reshape(3, 3)
    fractal = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            fractal[i, j] = (
                (M0[i // 3, j // 3] - 1) * 9
                + (M0[i % 3, j % 3] - 1)
            )
    return fractal


def apply_fractal_transform(x: torch.Tensor, fractal) -> torch.Tensor:
    """
    Apply fractal permutation on a [B, 81, H, W] tensor.

    Returns:
        Transformed tensor of shape [B, 81, H, W]
    """
    B, C, H, W = x.shape
    assert C == 81, "Input must be 81 channels (9x9)."
    reshaped = x.view(B, 9, 9, H, W)
    transformed = torch.zeros_like(reshaped)

    for i in range(9):
        for j in range(9):
            idx = fractal[i, j] % 81
            src_i, src_j = divmod(idx, 9)
            transformed[:, i, j] = reshaped[:, src_i, src_j]

    return transformed.view(B, 81, H, W)


def visualize_fractal_parts(part1: torch.Tensor, part2: torch.Tensor, cmap='viridis'):
    """Visualize part1 and part2 frequency maps in log scale."""
    def to_grid(tensor):
        tensor = tensor[0].cpu().numpy().reshape(9, 9, 112, 112)
        return np.concatenate([np.concatenate(tensor[i], axis=1) for i in range(9)], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(np.log(np.abs(to_grid(part1)) + 1e-6), cmap=cmap)
    ax1.axis("off")
    ax2.imshow(np.log(np.abs(to_grid(part2)) + 1e-6), cmap=cmap)
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig("fractal_parts_visualization.png")


def load_image_tensor(image_path: str, size=(112, 112)) -> torch.Tensor:
    """Load and preprocess image to tensor [1, 3, H, W]"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dim


def main():
    image_path = "/path/to/celeba_aligned/054957.jpg"
    img_tensor = load_image_tensor(image_path).cuda()

    part1, part2 = create_square_subsets(img_tensor)
    fractal = generate_fsm()
    part1_fsm = apply_fractal_transform(part1, fractal)
    part2_fsm = apply_fractal_transform(part2, fractal)
    print(part1_fsm.shape, part2_fsm.shape)
    
    visualize_fractal_parts(part1_fsm, part2_fsm)


if __name__ == '__main__':
    main()
