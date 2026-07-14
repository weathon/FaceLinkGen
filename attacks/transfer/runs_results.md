# Non-leaked proxy runs — val on FIXED 150 casia faces (seed 42), EPOCHS=2 unless noted

Metric: val_epoch_cosine (lower=better, 0=identical). Pipeline: target(face)->highpass(k5)->norm->student, cosine-dist to clean InsightFace teacher.
NOTE: numbers here use 150-val and are NOT directly comparable to the handoff table (30-val).
Leaked upper bound (minus in train): ~0.50 minus / ~0.71 frac (from handoff, 30-val).

| tag | proxy | extras | best val_minus | best val_frac | notes |
|-----|-------|--------|----------------|---------------|-------|
| t1_transplant | transplant minus/frac 50/50 | jit0.2 | ~0.81 (stuck) | 0.926↓ (ep3, killed@400) | spectrum match HELPS frac, HURTS minus |
| **t2_legacy** | legacy dct/fft/gauss | prior art, ep2 | **0.75** | **0.93** | BASELINE to beat. block-DCT highpass good for minus |
| t3_dctrand | dctrand minus-only (aggr) | drop.3 gain.5 ep2 | 0.753 | 0.881 (early, drifts up) | minus==legacy; frac best-early but overfits up |
| t4_leaked | LEAKED (minus in train) | REFERENCE | **0.567**↓ | 0.816 | 150-val minus CEILING ~0.55. NOT deployable |
| t5_rich | dctrand+warp+blur+fracmix | g.4 d.2 warp3 blur1.5 | 0.81 | 0.866 | warp+blur HURT minus (distortion doesn't help) |
| t6_legacy_freeze | legacy + freeze identity head | 102/310 trainable | 0.85+ | 0.97 | freeze HURTS (early layers cant adapt to frozen head) |
| t7_jpeghp | gauss_hp(jpeg(face)) | step.1-.4 | 0.91 (worse) | 0.98 | looks most minus-like but trains worst. block-DCT-HIGHPASS != jpeg+gauss-hp |
| t8_deliver | dctrand+fracmix (mild) | g.3 d.15 mf.65 ep3 | 0.761 | 0.88 | combined best @step400 (0.769/0.883); fracmix slightly hurt minus vs pure dctrand |
| t9_nce | dctrand minus-only + InfoNCE | cosmae+nce temp.05 | 0.80 | - | contrastive HURTS (misaligned w/ cosine val metric) |
| t10_hponly | pure gaussian highpass (NO blocks) | - | 0.89 | 0.96 | KEY ABLATION: no block structure => 0.89. Blocks essential! |
| t11_deliver | pure dct_highpass (gain0 drop0) | block-DCT hp only | 0.755 | 0.90 | clean best block op; frac 0.90 (perturb helps frac more) |
| t12_wd | dctrand g.4 d.2 + wd0.02 | TODO#4 higher wd | 0.774 | 0.917 | higher wd ≈ no change vs wd0.001 |
| final_deliver | dctrand defaults g.4 d.2 | canonical run | 0.756 | ~0.93 | milder perturb → worse frac than t3 |

## DELIVERABLE = student_deliverable.pth (copy of t3): val_minus 0.757 / val_frac 0.881.
Script DEFAULTS now reproduce it (PROXY=dctrand GAIN=0.5 DROP=0.3 MINUS_FRAC=1.0 EPOCHS=2).

## block_dct_analysis.py (deep confirm): matching minus's 8x8 DCT coeff profile is ANTI-predictive.
dct_highpass coeff-dist is FARTHER from minus (L1 0.70) than gaussian-hp (0.37), yet transfers
better (0.75 vs 0.89). The block-grid BOUNDARY artifact (per-block low-freq zeroing) is what helps,
NOT statistical similarity. Same lesson as the failed radial-spectrum transplant.

## CONCLUSION (strong): fair floor for minus ~= 0.75 on 150-val (ceiling 0.55).
Best minus proxy = block-DCT-HIGHPASS family (legacy dct_highpass / dctrand): 0.75.
EVERYTHING else is worse: spectrum-transplant 0.81, warp+blur 0.81, freeze 0.85, jpeg+gausshp 0.91.
The 0.75->0.55 gap is minus-specific decoding requiring leakage (random fair degradations only
buy domain-INVARIANCE, capped ~0.75; matching minus's CONSISTENT distortion needs seeing it).

## DECISION-CRITICAL: leaked ceiling minus ~0.55 on 150-val; fair floor ~0.75 => 0.20 HEADROOM.
So 0.75 is NOT the ceiling; keep pushing proxy faithfulness. The 0.75->0.55 gap is minus-specific
decoding that needs seeing minus's within-block content distortion. Approximate it with
blur+warp+block-DCT (minus pre-highpass output is blurred/warped/blocky).

## Pattern: val_minus is STUCK ~0.75 across legacy/dctrand (transplant worse 0.81).
Strongly suggests ~0.75 is near the fair floor for minus on 150-val. frac overfits (drifts up)
=> regularization may help frac/combined. Measuring leaked ceiling to confirm room.

## CORRECTED KEY INSIGHT
Amplitude/radial-spectrum match (transplant) HURT minus (0.81 vs legacy 0.75). The thing that
helps minus is the **8x8 block-DCT structure** (legacy's dct_highpass provides it; minus target
has blockspike 1.47). So: block structure >> radial spectrum for the minus domain.
Transplant still best for frac (whitening matches frac's flat broadband). A good combined proxy
likely = block-DCT family for minus-mode + frac-transplant for frac-mode.

## Key analysis findings (cal_analysis.py, proxy_design.py)
- minus target: blockspike 1.47 (8x8 DCT grid), kurtosis +15.3 (sparse), radial spectrum FLAT/broadband.
- frac target: blockspike ~1.0, kurt +9, flat broadband, anisotropic (vertical).
- proxies (gauss/fft highpass) are spectrally COLORED (peaked) -> big radial-spectrum gap.
- amplitude transplant -> radial dist minus 2.18->0.86, frac 3.53->0.55 (fixes spectrum).
- BUT empirically transplant only helped frac, not minus => minus gap is geometric/phase
  (minusface warps faces; see freq_sample_minus_norm.png blobby warps), not amplitude.

## Hypotheses to test
- WARP (elastic) for minus's geometric distortion.
- Rich domain randomization: warp + transplant + block-dct + noise.
- minus-focused ratio.
