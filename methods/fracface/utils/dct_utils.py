import torch
import torch.nn.functional as F
from torchjpeg import dct


def dct_transform(x, chs_remove=None, chs_pad=False,
                  size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
    Applies block-based Discrete Cosine Transform (DCT) to an input RGB image tensor.
    Optionally removes or zeroes out specific frequency channels.

    Args:
        x (Tensor): Input image tensor of shape [B, 3, H, W], with values in range [-1, 1].
        chs_remove (list or None): Indices of frequency channels (0–63) to remove.
        chs_pad (bool): If True, pruned channels are zero-padded instead of removed.
        size (int): DCT block size (typically 8).
        stride (int): Stride for block DCT, controls downsampling.
        pad (int): Padding added before applying unfolding.
        dilation (int): Dilation factor for unfolding blocks.
        ratio (int): Upsampling ratio before applying DCT (to enhance frequency resolution).

    Returns:
        Tensor: Frequency-domain tensor of shape [B, C, H', W'], where C depends on pruning.
    """
    # Ensure input has 3 channels (RGB)
    assert x.shape[1] == 3

    # Normalize input range to [0, 1] as required by TorchJPEG
    x = x * 0.5 + 0.5

    # Upsample to increase spatial resolution (improves DCT quality)
    x = F.interpolate(x, scale_factor=ratio, mode='bilinear', align_corners=True)

    # Convert to YCbCr color space and center around 0
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    # Apply block-wise unfolding (for BDCT)
    b, c, h, w = x.shape
    n_block = h // stride  # number of DCT blocks per dimension
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)

    # Apply DCT on each 8x8 block
    x_freq = dct.block_dct(x)

    # Rearrange frequency components: [B, C, 64, H_blocks, W_blocks]
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    # Optional: remove or zero out selected channels
    if chs_remove is not None:
        channels = list(set(range(64)) - set(chs_remove))
        if not chs_pad:
            # Remove channels directly
            x_freq = x_freq[:, :, channels, :, :]
        else:
            # Zero-out the kept channels
            x_freq[:, :, channels] = 0 

    # Merge color and frequency dimensions: shape [B, C*F, H_blocks, W_blocks]
    x_freq = x_freq.reshape(b, -1, n_block, n_block)

    return x_freq


def create_square_subsets(x, chs_prune_per_layer=None):
    """
    Split the DCT frequency-domain tensor into two 81-channel subsets,
    removing predefined low-frequency channels in each layer.

    Args:
        x (Tensor): Input image tensor of shape [B, 3, H, W].
        chs_prune_per_layer (list[list[int]]): List of channel indices (10 per layer, 3 layers).

    Returns:
        Tuple[Tensor, Tensor]: Two frequency channel subsets of shape [B, 81, H', W'] each.
    """
    if chs_prune_per_layer is None:
        # Default pruning indices (10 channels per DCT layer × 3 layers)
        chs_prune_per_layer = [
            [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],         # Layer 1
            [64, 65, 66, 67, 71, 72, 73, 80, 81, 88],    # Layer 2
            [128, 129, 130, 131, 136, 137, 138, 144, 145, 152]  # Layer 3
        ]

    # Flatten all pruned channels into a single list
    all_pruned = [ch for layer in chs_prune_per_layer for ch in layer]

    # Apply DCT with channel pruning
    x_freq = dct_transform(x, chs_remove=all_pruned, chs_pad=False)

    # Generate indices for two square subsets (81 each) using snake pattern
    part1_indices, part2_indices = generate_snake_indices()

    # Slice into two parts
    part1 = x_freq[:, part1_indices, :, :]
    part2 = x_freq[:, part2_indices, :, :]

    return part1, part2


def idct_transform(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
    Inverse DCT: Reconstructs a spatial image from its frequency-domain representation.

    Args:
        x (Tensor): Frequency-domain input tensor of shape [B, 192, H', W'].
        size (int): DCT block size.
        stride (int): Stride used during DCT transform.
        pad (int): Padding used during DCT transform.
        dilation (int): Dilation factor used in unfolding.
        ratio (int): Downsampling ratio (inverse of upsampling during DCT).

    Returns:
        Tensor: Reconstructed spatial image tensor of shape [B, 3, H, W], range [0, 1].
    """
    b, _, h, w = x.shape

    # Reshape to original block format: [B, 3, 64, H', W']
    x = x.view(b, 3, 64, h, w)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(b, 3, h * w, size, size)

    # Apply inverse DCT block-wise
    x = dct.block_idct(x)

    # Fold back to full image resolution
    x = x.view(b * 3, h * w, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * ratio, 112 * ratio),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(b, 3, 112 * ratio, 112 * ratio)

    # Postprocessing: convert back to RGB, normalize to [0, 1]
    x = x + 128
    x = dct.to_rgb(x)
    x = x / 255
    x = F.interpolate(x, scale_factor=1 / ratio, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)

    return x
