# De-identification linkage attack: FFHQ512 train / LFW test

Paired distillation attack against four protection targets, trained entirely on FFHQ512
and tested entirely on LFW. Every script is run from anywhere (paths are absolute);
artifacts go to `/raid/wg25r/redteam_work`.

## Setting

| | |
|---|---|
| Attack side (teacher = student init) | AdaFace **IR-101 WebFace12M** |
| Protection side | all ArcFace family — PerceptFace = SimSwap ArcFace R50, CanFG/CanFG-Ano = IR-SE50, TIP-IM = IR-SE50 |
| Protection training | none: all four use official released weights |
| Attack training | FFHQ512, 2000/500/200/100 (protected, original) pairs, disjoint from gate-val |
| Test | LFW deepfunneled — gallery 10,723 clean images (5,454 identities), query 1,550 protected images, K = ⌈0.005·10723⌉ = 54 |

The query side is protected; the gallery stays clean and is identical for all four methods.
All splits are restricted to the 224∩128 crop intersection so the four rows share a gallery.

## Targets

| name | what it is |
|---|---|
| `perceptface_official` | released `90000_net_G.pth` + `ID_transform`, arXiv 2509.11249 |
| `canfg` | released `seed85_anonymized_100_id_0_em_500_lp_10.pt`, ACM MM 2024 |
| `canfg_ano` | CanFG's stage-1 PID remover only — physical identity removed, no virtual identity embedded |
| `tipim` | per-image MI-FGSM, no training, ICCV 2021 |

## Run order

```bash
PY=/home/wg25r/face_deid/.venv/bin/python
export PYTHONPATH=/home/wg25r/face_deid/PerceptFace/pylibs

# data + weights
bash a2_fetch_data.sh                  # FFHQ512 (HF) + LFW deepfunneled
bash b1_fetch_models.sh                # AdaFace ckpt, TIP-IM source + IR-SE50
$PY a3_measure.py                      # measure what landed on disk

# crops: 224 = PerceptFace's own SCRFD, 128 = CanFG's own MTCNN, 112 = 224 downscaled
$PY c1_crop_224.py {ffhq|lfw}
$PY c2_crop_128.py {ffhq|lfw}
$PY c3_resize_112.py {ffhq|lfw}
$PY a6_rebuild_splits.py               # splits over the 224-and-128 intersection

# protected images
$PY d7_gen_perceptface_official.py
$PY d1_gen_canfg.py {canfg|canfg_ano}
$PY d6_tipim.py --shard i --nshards 16 # batch 1, parallel shards

# GATE -- do not attack if this fails
$PY e1_gate.py {perceptface_official|canfg|canfg_ano|tipim}

# attack + score
$PY f1_distill.py <method> <n>             # fixed 5000 steps
$PY f2_distill_converge.py <method> <n>    # early stop on held-out val_cos
$PY g1_eval.py <method> <n> [distill|converge]
$PY g2_cosine.py <method> <n> [distill|converge]
```

## Notes on the upstream code

* `c2` catches `ValueError` around CanFG's `align_multi`: upstream `detect_faces` vstacks the
  P-Net output list unchecked, so a no-face image raises instead of taking its own
  `len(landmarks)==0` branch. An image with anything other than exactly one detection is
  skipped whole.
* `d6` keeps `batch_size=1`: upstream `submodular` sizes `gains` by `len(target_feas)==10`
  but indexes it with `batch*10`, so it is only correct at batch 1. Throughput comes from
  parallel shards. `input_diversify.py` in the clone was patched to drop the unused
  `scipy.misc` import and to pass `align_corners=True` explicitly.
* `d3`/`d4` are the self-training reproduction (Stage-1 APIM, Stage-2 PEIT), copied verbatim
  from the parent workspace apart from paths and the frozen netG source. They are **not**
  used for the reported numbers — those all come from the official weights.
* `a1_fetch_digiface.sh` is dead: DigiFace-1M was the original test set, dropped when its
  500K part measured 112×112 RGBA, below the 224 PerceptFace needs.
* `f1_align_lfw.py`, `f1_fetch_hf.py`, `f1_fetch_fairface.py` belong to a separate
  `fracface_rerun` experiment and write to a different work root.

## Metrics

`g1_eval.py` reports, per query, the rank of the best-placed same-identity gallery image:
`top1_hit`, `topK_hit`, `topK_recall` (fraction of that identity's gallery images inside
top K), `avg_rank_best` and `avg_rank_all` (both normalised by gallery size), plus median.
`g2_cosine.py` adds the average query-to-target cosine with an impostor baseline.
Three rows each: `before attack` (protected query through the teacher), `after attack`
(through the student), `upper bound` (clean query through the teacher).
