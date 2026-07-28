# FracFace U-Net reconstruction attack — released implementation, plus a paper-aligned variant

We evaluate the **officially released implementation** (commit `3984dfe`, 2025-05-23),
verified byte-identical to the `anonymous.4open.science/r/FracFace` copy cited by the paper.
This is our primary setting: we follow the code as released.

While working with it we noticed several points where the released code and the paper
description differ. We do not take a position on which is intended. To confirm that our
results do not depend on how those points are resolved, we additionally implemented a
variant that follows the paper's description at each of them, and repeated the attack.

Attack: U-Net (81->3, init_features=3), L1 to the original image, AdamW lr 5e-4, cosine,
10 epochs, batch 256. Split from `fracface/index.txt` (CASIA-WebFace), 70,692 train /
17,994 val. Identical hyperparameters throughout; the only variable is the protection code.
The protected representation is regenerated with a fresh random secret **per image** in both
settings, following the released `preprocess_and_save`.

## Pixel-level metrics (identical 500 val images, matched by filename)

| Protection | SSIM | PSNR (dB) |
|---|---|---|
| Released code, fixed secret (seeded) | 0.6956 | 17.95 |
| Released code, per-image random secret | 0.6137 | 17.01 |
| Paper-aligned variant | 0.6197 | 17.14 |
| *Baseline: constant mean face* | *0.3676* | *12.84* |
| *Baseline: constant grey 0.5* | *—* | *11.35* |
| *Baseline: uniform random noise* | *—* | *7.97* |

## Identity leakage (Face++ compare, 1000 val images)

Success = a face detected in **both** images **and** confidence above the Face++ 1e-5
threshold.

| Protection | 1e-5 success | rate | SE |
|---|---|---|---|
| Released code, per-image random secret | 578 / 1000 | 57.8% | ±1.6 pp |
| Paper-aligned variant | 590 / 1000 | 59.0% | ±1.6 pp |

The 1.2 pp difference sits inside a paired SE of ~2.3 pp: the two settings are
**statistically indistinguishable**. Our conclusions do not depend on how the code/paper
differences are resolved.

CAVEAT: the two Face++ runs drew from different orderings of the same val set and overlap
on only 57 images, so this comparison is unpaired. The pixel metrics above are strictly
matched by filename. For a camera-ready number, re-run both on an identical filename list.

Full per-pair API responses: `facepp_frac_results.pkl/.json` (released),
`facepp_frac_allfixes.pkl/.json` (paper-aligned variant).

## Points where the released code and the paper differ

Stated as observations, not as defects. In each case the released behaviour is what we
evaluate in our primary results; the variant adopts the paper-side reading.

1. **Fractal iteration depth.** The code builds `E[2] = (A1-1)*3**4 + E[1] - 1` and applies
   `% 81` in `apply_fractal_transform`. Since 3^4 = 81, the level-2 term leaves the result
   unchanged modulo 81: we observe `E[2][:9,:9] % 81 == (E[1]-1) % 81` in 2000/2000 draws,
   so k=2 and k=1 induce the same mapping. The paper is itself inconsistent here — main-text
   Eq. 1 uses beta_k = 3^{2k} (matching the code), Appendix Eq. 15 uses beta_1 = 1,
   beta_l = prod b_s. The variant adopts Eq. 15, the reading under which k=2 is
   non-degenerate. (`E[2]` is 27x27 while the transform loop iterates `range(9)`.)
2. **Scope of channel pruning.** `chs_prune_per_layer` lists three groups; the layer-2 and
   layer-3 entries are >= 64, while `dct_transform` selects via
   `set(range(64)) - set(chs_remove)`, so the layer-1 set is applied to all three planes.
   Read as per-plane global indices, the intended sets differ from this on one channel of
   162 (Cb offset 7 vs 10). The variant applies them per-plane.
3. **Index lattice L0.** The paper describes L0 alongside M0 as a secret component;
   `grep -rn "L0\|lattice"` over the repository returns no hits. L0 is also initialised but
   not subsequently referenced in the paper's Algorithm 2. The variant implements it as a
   secret ordering, which is our interpretation.
4. **Input range convention.** `dct_transform` documents `[-1,1]` input and applies
   `x*0.5+0.5`, while `preprocess_and_save` supplies `[0,1]`, so images enter as
   `[0.5,1.0]`. The variant supplies `[-1,1]` as documented.
5. **Basis of the two-group split.** The snake index is constructed over an 18x9 grid,
   which partitions by frequency if the channel axis is frequency-major; the reshape
   produces plane-major order (Y 54 | Cb 54 | Cr 54), so `part1` comprises all 54 surviving
   Y channels plus 27 Cb, and `part2` the remaining 27 Cb plus all 54 Cr. As sets,
   `part1 == {0..80}` and `part2 == {81..161}` exactly. The paper describes the split as
   partitioning frequency bands; the variant reorders to frequency-major, giving `part1` =
   27 Y + 27 Cb + 27 Cr. The paper does not state the channel layout, so which reading is
   intended is our inference.

## Why the outcome is insensitive to all five

Points 1, 3 and 5 change only *which* channels are retained and in *what order*. Because
`dct_transform` upsamples 8x (`ratio=8`) before the block-DCT, each of the 81 channels is a
full-resolution 112x112 spatial map, so identity survives any permutation of a set of
face-shaped feature maps. Point 2 moves one channel of 162. Point 4 restores contrast, which
if anything favours the attacker.

Worth stating in the paper: FCR removes the 10 lowest frequencies including DC, so the
retained luminance has near-zero linear correlation with the greyscale image (measured
-0.004) and appears as noise — yet the attack still recovers identity at ~59% Face++ 1e-5.
Low pixel-space similarity is not evidence of privacy.

## Interpretation caveats to state explicitly

- The paper's main text and appendix give different beta schedules; we note both and adopt
  the appendix reading for the variant only.
- The paper does not state the channel layout, so point 5's paper-side reading is inferred.
- The paper does not define how L0 is applied, so point 3's implementation is ours.

Provenance of the evaluated code: `fracface/FRACFACE_PROVENANCE.txt`.
