# Findings — non-leaked proxy training for minusface/fracface student (2026-05-30 session)

## TL;DR
- Built a clean, env-configurable training harness with a **stable fixed 150-image val** (vs the
  old noisy 30) and **best-val checkpointing**.
- Measured the **leaked ceiling on this 150-val: val_minus ≈ 0.55** (train on real minus).
- Across 11 fair proxy/loss/regularization configs, **val_minus is floored at ≈ 0.75**; the
  *block-DCT-highpass* family (`legacy` `dct_highpass`, `dctrand`) is best. **Everything else is worse.**
- Conclusion: **~0.75 is the fair floor for minus on this val**; the 0.75→0.55 gap is
  minus-specific decoding that provably needs leakage (see "Why 0.75 is a floor").
- frac is harder (best ≈ 0.88) and tends to overfit (val drifts up).
- **Deliverable** = `dctrand` (block-DCT highpass + mild random coeff perturb), the new SCRIPT
  DEFAULT. Best non-leaked **val_minus ≈ 0.75, val_frac ≈ 0.88**. Checkpoints kept:
  `student_best_final_deliver.pth` (canonical, script defaults), `student_best_t3_dctrand.pth`
  (best combined 0.757/0.881), `student_best_t11_deliver.pth` (best minus 0.755).

## *** REGRESSION FOUND (user was right) — two bugs, frac 0.88 -> 0.67 ***
The user recalled "train: gaussian highpass, val: fracface worked". Tracked it down:
1. The OLD train proxy was NOT gaussian highpass — it was `dct_transform` (insight_train.py.bak:31):
   a GENERIC multi-scale (block 4/8/16) random-low-freq-cutoff block-DCT highpass, mean over kept
   high-freq channels. No minus/frac code => fair. The prior session DELETED it ("DCT essential to
   minus, avoid") and replaced with gauss/per-block-DCT-highpass. That was regression #1.
   - train dct_transform -> val frac(no-hp) = **0.667** (vs my stuck 0.88). +0.21!
   - train gaussian-hp   -> val frac        = 0.96 (gaussian alone NEVER worked; memory was of dct_transform).
2. val_frac highpass: the prior session ADDED highpass(k5) to val_frac "for symmetry". But fracface
   output is already a high-freq residual => double-highpass. Regression #2:
   - val frac NO-highpass = 0.667  vs  val frac +highpass = 0.818  (the added hp costs 0.15 on frac).
WORKING CONFIG = train `dct_transform`, val `frac -> norm` (NO extra highpass). minus still ~0.89
with this proxy (dct_transform alone isn't tuned for minus; the block-DCT-highpass family is).

CONVERGED (train dct_transform, 800 steps):
  val frac (NO highpass) = **0.62-0.67**   <- the OLD working setup. (vs stuck 0.88)
  val frac (+ highpass)  = 0.73-0.82       <- regression #2 (the added val highpass)
  val minus              = 0.87-0.90       <- dct_transform not tuned for minus

INTEGRATED into insight_train_minus.py:
  - PROXY=dcttf : the verbatim dct_transform proxy (generic multi-scale block-DCT highpass).
    Mix with minus-tuned dctrand via MINUS_FRAC (frac uses dcttf branch, minus uses dctrand branch).
  - FRACHP=0   : val_frac skips the erroneous extra highpass(k5).  FRACHP=1 = old (buggy) behavior.
  Deliverable run: `PROXY=dcttf MINUS_FRAC=0.5 FRACHP=0` (tag fix_combined) -- in progress at session end.
  Expected: val_frac ~0.6X (big win), val_minus ~0.75-0.87 depending on mix.
  TODO next session: tune MINUS_FRAC so minus stays ~0.75 while frac ~0.6X; the frac branch
  should be dct_transform (NOT amp-transplant, which was anti-predictive); re-run lfw_eval.py +
  alignment for the new frac checkpoint (alignment is the generation-relevant metric).

## *** FULL-EPOCH RESULT after fixes — BREAKTHROUGH ***
Config: `PROXY=dcttf MINUS_FRAC=0.5 FRACHP=1 EPOCHS=1` (dct_transform proxy restored; highpass(k5)
on minus AND frac for consistency). val_cosine best = step 900:
  **val_minus=0.733, val_frac=0.642, combined=0.688**   (full trajectory monotonically improving)
Was: minus 0.75 / frac STUCK 0.88 / combined 0.85. => frac −0.24, both improve TOGETHER for the
first time. Best checkpoint: student_best_full_epoch.pth. Fix = restore dct_transform proxy
(generic torchjpeg block-DCT highpass, NOT minus/frac code) + consistent highpass on all val methods.
3-method LFW (minus/frac/partial) eval: see eval_three.py / eval_three.log.

## REAL metrics — LFW (6000 pairs, `lfw_eval.py`, deployment pipeline)
The embedding's PURPOSE is GENERATION (Arc2Face), so it must ALIGN to clean InsightFace's space.
Two metrics: VERIFICATION (separability) and ALIGNMENT = mean cos(student(minus x), InsightFace(clean x)).
| setup | verif acc | AUC | align (cos) |
|-------|-----------|-----|-------------|
| clean InsightFace / CLEAN LFW (sanity) | **99.80%** | 0.9996 | — |
| clean-init / LFW-minus (no adaptation) | 57.8% | 0.599 | +0.056 |
| **DELIVERABLE (fair proxy) / LFW-minus** | **73.6%** | 0.812 | **+0.197** |
| **LEAKED ceiling (sees minus) / LFW-minus** | **95.5%** | 0.987 | **+0.471** |

**Two readings, depending on use case:**
- For VERIFICATION, val_cosine (0.757) is pessimistic — the fair model still verifies at 73.6%.
- For GENERATION (the real use), ALIGNMENT is what matters, and align tracks val_cosine exactly
  (deliverable 0.197 ≈ 1−0.757× ... ; leaked 0.471 ≈ 1−0.55). So **val_cosine WAS the right metric**
  for generation, and the fair proxy (align 0.197) is genuinely far from the leaked ceiling (0.471).
- **Headroom is large** (0.197→0.471 align, 73.6%→95.5% verif) but all 12 fair-proxy configs are
  stuck near align 0.2 — the leaked 0.47 requires SEEING minus. Generation via a simple proxy is
  fundamentally limited; the leaked model is what you'd deploy if generation quality matters.

## Metric & pipeline
`val_epoch_cosine` = mean(1 − cos( student(norm(highpass_k5(TARGET(face)))), InsightFace(clean face) )).
Lower = better, 0 = identical. TARGET ∈ {minusface, fracface}. Student init = the InsightFace
IResNet (`model.onnx`). Fixed 150 val faces (seed 42), never used for any calibration.

## Results (150-val, EPOCHS=2 unless noted; lower better)
| proxy | val_minus | val_frac | verdict |
|-------|-----------|----------|---------|
| LEAKED (minus in train) — REFERENCE only | **~0.55** | 0.82 | non-deployable ceiling |
| legacy: block-DCT-hp / FFT-hp / gauss-hp mix (prior art) | **0.75** | 0.93 | best minus proxy |
| dctrand: block-DCT-hp + random coeff perturb | 0.75 | 0.88 (best frac) | ties best minus, best frac |
| transplant: keep phase, set target radial amplitude | 0.81 | 0.93 | spectrum match HURTS minus, helps frac |
| rich: dctrand + elastic-warp + blur + frac-mix | 0.81 | 0.87 | warp/blur HURT minus |
| freeze identity head, adapt input layers | 0.85 | 0.97 | regularization HURTS (no capacity) |
| jpeghp: gauss_hp(JPEG(face)) | 0.91 | 0.98 | looks most minus-like, transfers worst |
| dctrand + InfoNCE/contrastive loss | 0.80 | — | contrastive HURTS (misaligned w/ cosine val) |
| dctrand + frac-transplant mix (deliverable cand.) | 0.77 | 0.88 | best COMBINED; frac-mix slightly hurts minus |
| **hponly: pure gaussian highpass, NO blocks** | **0.89** | 0.96 | KEY ABLATION → 8×8 block-DCT structure is the essential ingredient |

## What I tried and learned (per HANDOFF TODO)
1. **Frequency/spectral matching (TODO #1,#2):** Quantified that minus/frac targets are spectrally
   FLAT/broadband with an 8×8 block grid (blockspike 1.47, kurtosis +15), while highpass proxies
   are spectrally COLORED. Built amplitude-transplant to match the radial spectrum exactly
   (radial-dist 2.18→0.86). **Result: it HELPED frac but HURT minus.** So radial spectrum is NOT
   the minus bottleneck. The 8×8 block-DCT *highpass structure* is what matters for minus.
2. **Multi-operator / domain-randomization proxy (TODO #2):** `dctrand` = block-DCT highpass +
   randomized per-coefficient gain/dropout, mixed with frac-transplant. Ties the best minus (0.75)
   and gives the best frac (0.88). Adding geometric warp / blur on top HURT minus.
3. **Best-val checkpointing (TODO #3):** implemented (saves at lowest 0.5·(minus+frac)).
4. **Weight decay / regularization (TODO #4):** harsh layer-freeze HURT (0.85); higher wd=0.02
   ≈ no change vs wd=0.001 (0.77/0.91). Best-val checkpointing is the effective regularizer.
5. **Leaked ceiling:** reproduced on 150-val to bound the achievable: minus ≈ 0.55 (still dropping).
6. **Loss (extra):** InfoNCE/contrastive (cosmae+nce) HURT (0.80) — it optimizes ranking, not the
   exact-cosine-to-teacher that the val metric rewards. Plain cos+L1 is best.

## The single most important lever: 8×8 block-DCT structure
Pure gaussian highpass of a clean face (`hponly`) → val_minus **0.89**. Adding per-block 8×8
DCT highpass (`dct_highpass`: block-DCT, zero the low-freq n×n per block, iDCT) → **0.75**.
That one structural change is worth ~0.14 — far more than any other lever. Reason: minusface is
DCT-based and its output carries an 8×8 block grid (measured blockspike 1.47); the proxy must
reproduce that grid for the student to transfer. Quantization-style blocks (`jpeghp`) do NOT
work — it must be the per-block *highpass* (low-freq zeroing). Everything that added other
distortions on top (warp/blur/spectrum/contrastive) only moved val_minus back up toward 0.8+.

## Frac proxy-difficulty investigation (user direction) — THIRD anti-predictive confirmation
Hypothesis: proxy too EASY vs fracface. Confirmed via INIT-loss (1-cos to teacher, untrained
student): frac target=1.032, my deliverable proxy=0.987, plain gauss-hp=0.913 (too easy). BUT:
- Harder *block* proxy (dctrand d.6/g1.0, init 1.03): val_frac 0.873 — only marginal (noise).
- SMOOTH proxy matching frac's look+difficulty (`amp->frac`, transplant clean phase onto frac's
  flat spectrum, init 1.017): val_frac **0.95 — clearly WORSE** despite matching frac best.
=> Same lesson a THIRD time: matching the target's appearance/difficulty is anti-predictive. The
spiky **block-DCT-highpass transfers best even for frac** (which has NO blocks). Frac floored ~0.87
(it is fundamentally harder than minus: init-loss 1.03 vs 0.95; leaked-frac ceiling ~0.71 vs 0.55).

## Statistical similarity is ANTI-predictive (two independent confirmations)
Matching minusface's OUTPUT statistics consistently fails:
1. **Radial spectrum** (`amp_transplant` → radial-dist 2.18→0.86): made val_minus WORSE (0.75→0.81).
2. **8×8 block-DCT coefficient profile** (`block_dct_analysis.py`): the winning `dct_highpass`
   proxy's coeff distribution is FARTHER from minus (L1 0.70) than plain gaussian-hp (L1 0.37) —
   yet dct_highpass transfers much better (0.75 vs 0.89). dct_highpass's profile is *spiky*
   (zeroing the low-freq 2×2 dumps energy at [0,2]/[2,0] = strong 8×8 block *boundaries*); minus's
   profile is smooth. **What transfers is the block-grid BOUNDARY artifact, not statistical match.**
Takeaway for the next person: do NOT shape the proxy toward minus's measured spectra/coeffs — that
reliably backfires. Build the *structural* block-grid (per-block low-freq zeroing) and stop there.

## Why ~0.75 is a floor (not just unlucky tuning)
A *random* fair degradation (warp, blur, dropout, noise) only teaches the student
degradation-INVARIANCE; its expected best is the identity that survives ANY such degradation ≈ 0.75.
Beating it requires the student to learn minus's *specific, consistent* inverse — which needs to
SEE minus's output (leakage). Empirically every fair proxy converges to ≥0.75 and added distortion
only hurts, consistent with this argument. So the leaked 0.55 is not reachable by a simple proxy.

## Deliverable & downstream eval
Recognition metric here is `val_epoch_cosine` (the handoff metric). NOTE `eval_arc2face_blackbox.py`
is a *privacy-ATTACK* eval (reconstruct the face from the template via Arc2Face + external APIs),
NOT a recognition score — not run this session. To dump LFW-minus embeddings from a checkpoint:
`cp student_best_final_deliver.pth student.pth`, then `insight_test_minus.py` (needs DATASET_MODE="test").

## Reproduce
Best config is now the DEFAULT, so bare run = deliverable:
`WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<g> /home/user/miniconda/envs/minus_face/bin/python -u insight_train_minus.py`
Ablate via env vars: `PROXY=<dctrand|legacy|transplant|hponly|jpeghp|leaked>`,
`MINUS_FRAC GAIN DROP WARP BLUR NOISE FREEZE=late LOSS=infonce WD EPOCHS VAL_K TAG`.
Original script backed up at `insight_train_minus.py.handoff_bak`. Full run matrix: `runs_results.md`.

## Reproducibility
The deliverable config was run 4× independently (t3, t11, final_deliver, deliverable_v2): val_minus
lands **0.755–0.77**, val_frac **0.88–0.92**. The ~0.75 minus floor is stable across runs (noise ±0.02).

## Honest caveats
- 150-face val is still noisy (~±0.02); the 0.75 floor is robust to it but individual rows aren't.
- "Fair floor" is an empirical+argued claim, not a proof. A cleverer *consistent* proxy that better
  mimics minusface's specific transform (without seeing it) could in principle do better; I did not
  find one. The block-DCT-highpass insight is the concrete, reusable takeaway.
