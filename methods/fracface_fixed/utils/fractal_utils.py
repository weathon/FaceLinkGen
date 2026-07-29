import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchjpeg import dct
from utils.dct_utils import dct_transform


def generate_snake_indices():
    """
    Generate snake-like traversal indices for a 2D grid.
    This ordering is used to split DCT frequency channels into two sets 
    in a structured, interleaved fashion.

    Returns:
        part1_indices (np.ndarray): Indices for the first 81 channels.
        part2_indices (np.ndarray): Indices for the remaining 81 channels.
    """
    size = 9
    total_channels = 162
    indices = np.zeros((size * 2, size), dtype=int)

    # Generate snake-like indices row by row
    for row in range(size * 2):
        if row % 2 == 0:  # Even rows: left to right
            indices[row] = np.arange(row * size, (row + 1) * size)
        else:  # Odd rows: right to left
            indices[row] = np.arange((row + 1) * size - 1, row * size - 1, -1)

    # Flatten and split into two equal parts
    flat_indices = indices.flatten()
    part1_indices = flat_indices[:81]
    part2_indices = flat_indices[81:]

    return part1_indices, part2_indices


def create_square_subsets(x, chs_prune_per_layer=None):
    """
    Divide DCT frequency channels into two 9x9 square subsets after pruning low-frequency bands.

    Args:
        x (torch.Tensor): Input image tensor of shape [B, C, H, W].
        chs_prune_per_layer (list[list[int]], optional): List of 10 channels to prune per layer. 
            If None, default values are used for three layers (30 total channels).

    Returns:
        part1 (torch.Tensor): First subset of shape [B, 81, H, W].
        part2 (torch.Tensor): Second subset of shape [B, 81, H, W].
    """
    # Default channel indices to prune from each of the 3 frequency layers
    if chs_prune_per_layer is None:
        chs_prune_per_layer = [
            [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],      # Layer 1
            [64, 65, 66, 67, 71, 72, 73, 80, 81, 88], # Layer 2
            [128, 129, 130, 131, 136, 137, 138, 144, 145, 152]  # Layer 3
        ]

    # Flatten all pruned channels into a single list
    all_pruned = [ch for layer in chs_prune_per_layer for ch in layer]

    # 1. Apply DCT transform and remove specified channels
    x_freq = dct_transform(x, chs_remove=all_pruned, chs_pad=False)  # Output: [B, 162, H, W]

    # 2. Generate snake indices
    part1_indices, part2_indices = generate_snake_indices()

    # 3. Extract two subsets of frequency channels
    part1 = x_freq[:, part1_indices]  # [B, 81, H, W]
    part2 = x_freq[:, part2_indices]  # [B, 81, H, W]

    return part1, part2


def randomize_A1():
    """
    Randomly generate a base 3x3 matrix A1 with values in [1, 9].
    Used as the foundation for building fractal index matrices.

    Returns:
        A1 (np.ndarray): A 3x3 matrix with random integers.
    """
    A1 = np.random.randint(1, 10, size=(3, 3))
    return A1


def generate_fsm(iterations=4):
    """
    Generate a fractal structure matrix (FSM) by recursively expanding a random base matrix.

    Args:
        iterations (int): Number of recursive expansion steps. 
            Each step increases the resolution by a factor of 3.

    Returns:
        E (list[np.ndarray]): List of fractal index matrices from each iteration.
    """
    E = [None] * iterations
    A1 = randomize_A1()  # Base matrix
    E[0] = A1  # Initial 3x3 matrix

    for k in range(1, iterations):
        q = k - 1
        factor = 3 ** (2 * k)

        # Create sub-blocks using A1 offsets and previous level matrix
        B1 = (A1[0, 0] - 1) * factor + E[q]
        B2 = (A1[0, 1] - 1) * factor + E[q]
        B3 = (A1[0, 2] - 1) * factor + E[q]
        B4 = (A1[1, 0] - 1) * factor + E[q]
        B5 = (A1[1, 1] - 1) * factor + E[q]
        B6 = (A1[1, 2] - 1) * factor + E[q]
        B7 = (A1[2, 0] - 1) * factor + E[q]
        B8 = (A1[2, 1] - 1) * factor + E[q]
        B9 = (A1[2, 2] - 1) * factor + E[q]

        # Stack into a 3x3 block matrix
        E[k] = np.block([
            [B1, B2, B3],
            [B4, B5, B6],
            [B7, B8, B9]
        ])

        # Adjustment for level-2 matrix (9x9) to fit 81-channel format
        if k == 2:
            E[k] = E[k] - 1

    return E


def apply_fractal_transform(feature_tensor, E, iteration_level=2):
    """
    Apply fractal transformation on input tensor based on fractal index matrix.

    Args:
        feature_tensor (torch.Tensor): Input feature tensor of shape [B, 81, H, W].
        E (list[np.ndarray]): Fractal index matrices from `generate_fsm`.
        iteration_level (int): Which fractal level to apply (default 2 -> 9x9).

    Returns:
        transformed_tensor (torch.Tensor): Output tensor with the same shape [B, 81, H, W].
    """
    B, C, H, W = feature_tensor.shape
    assert C == 81, "Input must have 81 channels (9x9 fractal layout)"

    fractal_matrix = E[iteration_level].astype(int)

    # Reshape to 2D spatial grid
    feature_reshaped = feature_tensor.view(B, 9, 9, H, W)
    transformed = torch.zeros_like(feature_reshaped)

    # Perform rearrangement according to fractal matrix
    for i in range(9):
        for j in range(9):
            idx = fractal_matrix[i, j] % 81  # Wrap index to 0–80
            src_i, src_j = divmod(idx, 9)    # Map to 2D coordinates
            transformed[:, i, j] = feature_reshaped[:, src_i, src_j]

    return transformed.view(B, 81, H, W)


def form_training_batch_with_fractal(inputs, labels):
    """
    Generate a training batch by applying a shared fractal transformation 
    to all input samples using a single random FSM.

    Args:
        inputs (torch.Tensor): Input tensor [B, 81, H, W].
        labels (torch.Tensor): Corresponding labels.

    Returns:
        transformed_inputs (torch.Tensor): Fractal-transformed inputs.
        labels (torch.Tensor): Labels (unchanged).
    """
    b, _, _, _ = inputs.shape
    E = generate_fsm()  # Shared FSM
    transformed_inputs = apply_fractal_transform(inputs, E, iteration_level=2)
    return transformed_inputs, labels


# Alternative version (commented out):
# Applies a unique FSM to each sample in the batch independently.
# def form_training_batch_with_fractal(inputs, labels):
#     b, c, h, w = inputs.shape
#     transformed_inputs = torch.zeros_like(inputs)
#     
#     for i in range(b):
#         E = generate_fsm()
#         transformed_inputs[i] = apply_fractal_transform(
#             inputs[i].unsqueeze(0), E, iteration_level=2
#         ).squeeze(0)
#     
#     return transformed_inputs, labels
