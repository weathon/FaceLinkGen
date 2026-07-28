"""A paper-aligned variant of FracFace's FCR + FFM.

We evaluate the officially released implementation as our primary setting. While working
with it we noticed several points where the released code and the paper description differ.
We take no position on which is intended. This module implements the paper-side reading at
each of those points, so the attack can be repeated under both and the result shown not to
depend on how they are resolved.

This file is standalone and does not modify the FracFace repository, so the released code
remains citable in its original state.

Points addressed (see RESULTS_fracface_ablation.md for the full statement):

1. FRACTAL ITERATION DEPTH. The released code builds E[2] = (A1-1)*3**4 + E[1] - 1 and
   applies `% 81`; since 3**4 == 81 the level-2 term leaves the result unchanged modulo 81,
   so k=2 and k=1 induce the same mapping. The paper is inconsistent here: main-text Eq. 1
   uses beta_k = 3^{2k} (matching the code), Appendix Eq. 15 uses beta_1 = 1,
   beta_l = prod b_s. This module adopts Eq. 15, under which k=2 is non-degenerate:
       E2[i,j] = (M0[i//3, j//3] - 1) * 9 + (M0[i%3, j%3] - 1)
   spanning 0..80, so the modulus is inert.

2. SCOPE OF CHANNEL PRUNING. `chs_prune_per_layer` lists three groups; the layer-2/-3
   entries are >= 64 while `dct_transform` selects via `set(range(64)) - set(chs_remove)`,
   so the layer-1 set applies to all three planes. Read as per-plane global indices the
   intended sets differ on one channel of 162 (Cb offset 7 vs 10). Applied per-plane here.

3. INDEX LATTICE L0. Described in the paper alongside M0 as a secret component; not present
   in the repository, and initialised but not subsequently referenced in the paper's
   Algorithm 2. Implemented here as a secret ordering -- our interpretation.

4. INPUT RANGE CONVENTION. `dct_transform` documents [-1,1] and applies x*0.5+0.5, while
   `preprocess_and_save` supplies [0,1]. This module supplies [-1,1] as documented.

5. BASIS OF THE TWO-GROUP SPLIT. See `dct_transform_freqmajor` below.

`released_equivalent=True` reproduces the released behaviour for A/B comparison.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchjpeg import dct

# Per-plane pruning sets, reading the released constants as global indices.
PRUNE_PER_PLANE = [
    [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],   # Y
    [0, 1, 2, 3, 7, 8, 9, 16, 17, 24],    # Cb  (released code applies the Y set here)
    [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],   # Cr
]


def dct_transform_corrected(x, prune_per_plane=PRUNE_PER_PLANE, size=8, stride=8, ratio=8):
    """Block-DCT with per-plane channel pruning. Input x must be in [-1, 1]."""
    assert x.shape[1] == 3
    x = x * 0.5 + 0.5                       # -> [0,1]; correct only if input really is [-1,1]
    x = F.interpolate(x, scale_factor=ratio, mode="bilinear", align_corners=True)
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), stride=(stride, stride))
    x = x.transpose(1, 2).view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    # Per-plane pruning (the released code applies one set to all three planes).
    kept = []
    for plane in range(3):
        keep = sorted(set(range(64)) - set(prune_per_plane[plane]))
        kept.append(x_freq[:, plane][:, keep])
    x_freq = torch.cat(kept, dim=1)          # [B, 162, n_block, n_block]
    return x_freq


def generate_snake_indices():
    size, indices = 9, np.zeros((18, 9), dtype=int)
    for row in range(18):
        if row % 2 == 0:
            indices[row] = np.arange(row * size, (row + 1) * size)
        else:
            indices[row] = np.arange((row + 1) * size - 1, row * size - 1, -1)
    flat = indices.flatten()
    return flat[:81], flat[81:]


def generate_fsm_corrected(rng=None, use_L0=True):
    """Return a 9x9 index matrix over 0..80 -- a genuine two-level fractal expansion.

    M0 is the secret 3x3 fractal kernel; L0 is the secret 3x3 index lattice giving the
    ordering of elements within M0 (absent from the released code).
    """
    rng = np.random.RandomState() if rng is None else rng
    M0 = rng.randint(1, 10, size=(3, 3))
    if use_L0:
        L0 = rng.permutation(9).reshape(3, 3)      # secret ordering
        flat = M0.flatten()
        M0 = flat[L0.flatten()].reshape(3, 3)

    E2 = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            # Outer level selects a 3x3 block, inner level selects within it.
            E2[i, j] = (M0[i // 3, j // 3] - 1) * 9 + (M0[i % 3, j % 3] - 1)
    return E2                                       # already spans 0..80: % 81 is a no-op


def generate_fsm_released():
    """The released construction, for A/B comparison (k=2 coincides with k=1 under % 81)."""
    A1 = np.random.randint(1, 10, size=(3, 3))
    E = [None] * 4
    E[0] = A1
    for k in range(1, 4):
        factor = 3 ** (2 * k)
        E[k] = np.block([[(A1[r, c] - 1) * factor + E[k - 1] for c in range(3)] for r in range(3)])
        if k == 2:
            E[k] = E[k] - 1
    return E[2][:9, :9]


def apply_fractal(feature, index_matrix):
    """Gather channels according to the 9x9 index matrix (non-injective by design)."""
    B, C, H, W = feature.shape
    assert C == 81
    fr = feature.view(B, 9, 9, H, W)
    out = torch.zeros_like(fr)
    for i in range(9):
        for j in range(9):
            idx = int(index_matrix[i, j]) % 81
            si, sj = divmod(idx, 9)
            out[:, i, j] = fr[:, si, sj]
    return out.view(B, 81, H, W)


def protect(img_01, released_equivalent=False, rng=None):
    """Full FracFace protection of a [0,1] image tensor [B,3,112,112] -> [B,81,112,112]."""
    x = img_01 * 2 - 1                              # enforce the documented [-1,1] contract
    if released_equivalent:
        from utils.dct_utils import dct_transform
        x_freq = dct_transform(img_01, chs_remove=sum(
            [[0, 1, 2, 3, 8, 9, 10, 16, 17, 24],
             [64, 65, 66, 67, 71, 72, 73, 80, 81, 88],
             [128, 129, 130, 131, 136, 137, 138, 144, 145, 152]], []), chs_pad=False)
        idx_mat = generate_fsm_released()
    else:
        x_freq = dct_transform_corrected(x)
        idx_mat = generate_fsm_corrected(rng=rng)

    p1, _ = generate_snake_indices()
    part1 = x_freq[:, p1]
    return apply_fractal(part1, idx_mat)


if __name__ == "__main__":
    # Sanity checks for the two repairs that matter most.
    E_corr = generate_fsm_corrected(rng=np.random.RandomState(0), use_L0=False)
    print("corrected E2 range: %d..%d  (mod 81 is a no-op: %s)" % (
        E_corr.min(), E_corr.max(), bool(((E_corr % 81) == E_corr).all())))

    rel = generate_fsm_released()
    # Released: level-2 coincides with level-1 under the modulus.
    A_check = []
    for _ in range(200):
        E = [None] * 4
        A1 = np.random.randint(1, 10, size=(3, 3))
        E[0] = A1
        for k in range(1, 4):
            f = 3 ** (2 * k)
            E[k] = np.block([[(A1[r, c] - 1) * f + E[k - 1] for c in range(3)] for r in range(3)])
            if k == 2:
                E[k] = E[k] - 1
        A_check.append((((E[2][:9, :9]) % 81) == ((E[1] - 1) % 81)).all())
    print("released E2 coincides with E1 in %d/200 draws" % sum(A_check))

    x = torch.rand(2, 3, 112, 112)
    out = protect(x)
    print("corrected protect() ->", tuple(out.shape))


# ---------------------------------------------------------------------------
# Variant: FREQUENCY-MAJOR split.
#
# The snake index is built over an 18x9 grid of the 162 channels, which only
# yields a genuine *frequency-band* split if the channel axis is frequency-major.
# The released reshape produces PLANE-major order (Y 54 | Cb 54 | Cr 54), so the
# split falls along colour planes and routes every surviving Y channel into part1. This variant reorders to frequency-major first, so part1 becomes the
# lowest 27 surviving frequencies across all three planes -- i.e. what the paper
# describes as partitioning "the frequency bands".
#
# The paper never states the channel layout, so which reading is intended is our
# inference. Provided so the attack can be measured under both.
# ---------------------------------------------------------------------------

def dct_transform_freqmajor(x, prune_per_plane=PRUNE_PER_PLANE, size=8, stride=8, ratio=8):
    """Same as dct_transform_corrected but emits frequency-major channel order."""
    assert x.shape[1] == 3
    x = x * 0.5 + 0.5
    x = F.interpolate(x, scale_factor=ratio, mode="bilinear", align_corners=True)
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128
    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = F.unfold(x, kernel_size=(size, size), stride=(stride, stride))
    x = x.transpose(1, 2).view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)

    kept = []
    for plane in range(3):
        keep = sorted(set(range(64)) - set(prune_per_plane[plane]))
        kept.append(x_freq[:, plane][:, keep])          # [B,54,n,n] each
    stacked = torch.stack(kept, dim=2)                  # [B,54,3,n,n]  freq-major
    return stacked.reshape(b, 162, n_block, n_block)


def protect_freqmajor(img_01, rng=None):
    x = img_01 * 2 - 1
    x_freq = dct_transform_freqmajor(x)
    p1, _ = generate_snake_indices()
    return apply_fractal(x_freq[:, p1], generate_fsm_corrected(rng=rng))
