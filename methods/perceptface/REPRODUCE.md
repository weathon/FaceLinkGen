# Reproducing the PerceptFace attack

Everything needed to go from raw FFHQ512 to the before/after identity-leakage table lives
in this repository. Only two things are not shipped: the FFHQ512 images, and the model
weights (gitignored, fetch instructions below).

Target: **PerceptFace**, arXiv 2509.11249v1, *"Make Identity Unextractable yet
Perceptible"*. Attack: the paired distillation attack used elsewhere in this repo for
CanFG/FracFace/MinusFace/PartialFace.

Results from the run this was written against are in the last section.

---

## 1. What PerceptFace does

A frozen SimSwap generator `G` plus a frozen ArcFace `E_id`; the only trained component is
a 3-layer MLP `T` (called `ID_transform` / `WI` in the code):

```
img (3x224x224, ImageNet norm mean=(.485,.456,.406) std=(.229,.224,.225))
 |- F.interpolate->112x112 (bicubic) -> netArc -> 512d -> F.normalize   = original_id  [no_grad]
 |- original_id -> ID_transform(512->1024->LReLU(.01)->1024->LReLU->512) -> F.normalize = T_id
 `- netG(img_224, T_id) -> protected image -> denormalise -> save png
```

It is **fully deterministic**: no key, no randomness, the same input always maps to the
same protected image, and the protected image is pixel-wise very close to the original.
That is what makes a paired distillation attack possible. The paper's own Limitation 4
predicts it:

> "an adversary can collect a large number of paired protected and unprotected faces.
> They can **invalidate our method by training an en-decoder network**."

Note: the paper describes `T` as a "two-layer perceptron"; the released code has **three**
`nn.Linear` layers. The code is what runs.

---

## 2. Layout

```
third_party/perceptface/          vendored HF Space source (committed, see NOTICE.md)
checkpoints/perceptface/          PerceptFace weights                    (gitignored)
checkpoints/model.onnx            Antelopev2 glintr100, shared with the other attacks
data/perceptface/                 crops, protected images, logs          (gitignored)
methods/perceptface/              pair generation + the pre-attack check
attacks/perceptface/              teacher embeddings, distillation, scoring
```

Every script is run **from its own directory**, as elsewhere in this repo, so the
`../../` paths resolve. Only `SRC` in `methods/perceptface/prep_crops.py` has to be
edited: it points at the raw FFHQ512 image directory.

---

## 3. Weights

The GitHub repo `daizigege/PerceptFace` is missing files its own code imports
(`insightface_func/utils/face_align_ffhqandnewarc.py`, `insightface_func/__init__.py`,
`util/util.py` are all 404). The HuggingFace Space is complete and carries every weight:

```bash
git clone https://huggingface.co/spaces/daizigege/PerceptFace /tmp/perceptface_space
mkdir -p checkpoints/perceptface/antelope
cp /tmp/perceptface_space/pretrained_models/90000_net_G.pth                                  checkpoints/perceptface/
cp /tmp/perceptface_space/pretrained_models/arcface_checkpoint.tar                           checkpoints/perceptface/
cp /tmp/perceptface_space/pretrained_models/MSE_new_all_loss_id_5_rec_5_wa_5_step_40000.pt   checkpoints/perceptface/
cp /tmp/perceptface_space/insightface_func/models/antelope/scrfd_10g_bnkps.onnx              checkpoints/perceptface/antelope/
# Antelopev2 recogniser, shared with the other attacks in this repo:
cp ~/.insightface/models/antelopev2/glintr100.onnx                                           checkpoints/model.onnx
```

| file | size | what it is |
|---|---|---|
| `checkpoints/perceptface/90000_net_G.pth` | 210 MB | `Generator_Adain_Upsample` state_dict (the paper's Stage-1 output) |
| `checkpoints/perceptface/arcface_checkpoint.tar` | 200 MB | **a pickled `nn.Module`**, not a state_dict (SimSwap ArcFace R50) |
| `checkpoints/perceptface/MSE_new_all_loss_id_5_rec_5_wa_5_step_40000.pt` | 191 MB | `ID_transform` weights, under key `states['WI']` |
| `checkpoints/perceptface/antelope/scrfd_10g_bnkps.onnx` | 16 MB | SCRFD detector |
| `checkpoints/model.onnx` | 249 MB | Antelopev2 glintr100 |

The Space also ships `insightface_func/models/antelope/glintr100.onnx`.
`Face_detect_crop` globs every `.onnx` under its model root and loads it as the
recognition model, but **never uses it**, so it is deliberately not copied: with 64
worker processes that is 16 GB of wasted RAM.

---

## 4. Environment

```bash
pip install insightface==0.7.3 onnx2torch lpips wandb pandas scikit-image
```

Gotchas, all of them hit during the original run:

1. `torch.load(arcface_checkpoint.tar)` loads a **whole pickled module**, so torch >= 2.6
   needs an explicit `weights_only=False`, and `models/arcface_models.py` +
   `models/config.py` must be importable — hence the
   `sys.path.insert(0, '../../third_party/perceptface')` at the top of `gen_protected.py`.
2. insightface must be **0.7.3**. The PerceptFace README says `0.2.1`; that value was
   copied from SimSwap and is wrong, the code uses the new SCRFD/ONNX API.
3. `Face_detect_crop` globs the directory literally named **`antelope`**, not
   `antelopev2`.
4. `prepare(det_thresh=0.6, ...)` stores `det_thresh` but it has **no effect** — new
   insightface `detect()` has no threshold parameter.
5. `util/reverse2original.py` uses `np.float`, removed in numpy >= 1.24. This pipeline
   only handles 224 crops and never touches that file, so numpy 2.x is fine.
6. If your onnxruntime build is **CPU-only** (check
   `onnxruntime.get_available_providers()`), do not run glintr100 through ORT: it took
   ~15 min for 2000 images here, and its intra-op thread pool fights with a
   multiprocessing pool. `extract_embeddings.py` therefore runs the same ONNX graph
   through `onnx2torch` on the GPU, replicating `ArcFaceONNX.get_feat` preprocessing by
   hand (cv2 bilinear resize to 112, BGR->RGB, `(x-127.5)/127.5`).
   **This equivalence has never been checked numerically** — see section 8.
7. torch 1.11 does not run on sm_100 (B200); the original run used 2.13.0+cu132.

---

## 5. Data

FFHQ512: 8829 PNGs at 512x512.

* `08828.png` is corrupt (PNG IDAT checksum error) and is excluded by name in
  `prep_crops.py`.
* 7 images get no SCRFD detection and are skipped into `data/perceptface/skipped.txt`:
  `00783 01814 03500 03808 04222 05367 05529`.
* **8821** pairs remain. Split 8000 train / 821 val by sorted filename in
  `extract_embeddings.py`. FFHQ has one image per identity, so that split is already
  identity-disjoint.

### No padding, ever
`det_size` must not exceed the image size. SCRFD upsamples first when it does, and
detection collapses. Measured on 200 FFHQ512 images:

| det_size | detection rate | mean score |
|---|---|---|
| 640 | 0.405 | 0.607 |
| **512** | **1.000** | **0.806** |
| 384 | 1.000 | 0.861 |

`prep_crops.py` uses `(512, 512)`. Do not "fix" a low detection rate with
`cv2.copyMakeBorder`, letterboxing, or any other padding.

### Truncated crops after an interrupted run
`prep_crops.py` resumes by checking whether the output file exists. If a previous run was
killed mid-write, the leftover PNGs exist but cannot be decoded, and the next stage dies
with `OSError: image file is truncated`. Three files hit this in the original run
(`01876 01917 01941`). Find and delete any unreadable file under
`data/perceptface/crops224/` and rerun `prep_crops.py` to regenerate it.

---

## 6. Run order

```bash
cd methods/perceptface
python prep_crops.py                    # FFHQ512 -> data/perceptface/crops224   (~1 min, 64 CPU procs)
python gen_protected.py                 # -> data/perceptface/protected224       (~2 min, GPU)

cd ../../attacks/perceptface
python extract_embeddings.py            # -> log/{teacher,protected}_embeddings_insight.pkl,
                                        #    log/{train,val}_paths.pkl

cd ../../methods/perceptface
python check_protection.py              # GATE — do not continue if this fails

cd ../../attacks/perceptface
python insight_train.py                 # 10 epochs, ~2 min on one B200
python insight_test.py log              # the before/after table

python insight_train_lowdata.py         # optional: the 50-pair setting
python insight_test.py log_lowdata50
```

### The gate
`check_protection.py` exits non-zero if `mean cos(orig, protected) > 0.6`. The paper's
own range is 0.02–0.53; the original run got **0.1342** mean / 0.1333 median, impostor
mean 0.0018, pixel L1 0.0381 (paper: 0.032). A mean above 0.6 almost certainly means the
normalisation or the alignment is wrong — fix that before attacking. Also look at
`panel_protection.png` by eye: 8 pairs, original left / protected right, they should look
nearly identical and definitely not black.

### Resuming
`prep_crops.py`, `gen_protected.py` skip files that already exist. `insight_train.py`
writes `ckpt.pt` every epoch (model + optimizer + scheduler + wandb id) and resumes from
it automatically. All of them can just be restarted.

---

## 7. The attack recipe

Copied from `attacks/canfg/insight_train.py`:

* student: `onnx2torch.convert(glintr100.onnx)`, **all** parameters trainable
* input: the **protected** image, `Resize((112,112))` + `ToTensor()` — i.e. `[0,1]`, no
  further normalisation
* target: the Antelopev2 embedding of the **original** crop
* loss: `cos + 10*mae + 10*triplet`, triplet margin 0.3 with the in-batch
  `torch.roll(idx, 1)` negative; cos/mae computed after L2 normalisation
* AdamW `lr=1e-5, weight_decay=2e-2`, `CosineAnnealingLR(T_max=len(loader)*epochs)`,
  `epochs=10`, `batch=128`, training set repeated twice (`train_paths * 2`)
* **only deviation from the canfg version**: a missing teacher embedding raises instead of
  silently becoming `torch.zeros(512)`, and training is resumable

Note the student's input preprocessing (`[0,1]`, PIL resize) differs from the teacher's
(`(x-127.5)/127.5`, cv2 resize). That inconsistency is inherited from the original
FaceLinkGen recipe and was kept on purpose; the fully fine-tuned student absorbs it.

Scoring in `insight_test.py`: thresholds come from the impostor distribution inside the
validation split itself (one image per identity, so every cross pair is an impostor).
Top-1 is closed-set rank-1 identification — gallery is all 821 validation originals, the
correct answer is always in the gallery, no distractors and no rejection option.

---

## 8. Reference numbers

Validation split, 821 images, 673220 impostor pairs.
FAR=1e-3 threshold 0.2667, FAR=1e-4 threshold 0.3572.

| setting | mean cos | median | TAR@1e-3 | TAR@1e-4 | Top-1 |
|---|---|---|---|---|---|
| before attack | 0.1309 | 0.1278 | 0.0840 | 0.0110 | 0.1657 |
| after attack (8000 pairs, 10 epochs) | 0.6189 | 0.6282 | 0.9976 | 0.9951 | 0.9976 |
| upper bound (original vs itself) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Low-data setting (`insight_train_lowdata.py`, 50 pairs, 15 steps, constant lr):
mean cos 0.1904, TAR@1e-3 0.2314, TAR@1e-4 0.0524, Top-1 0.3593.

How many pairs the attack actually needs — unique-pair sweep at fixed compute
(5000 steps, batch 32, constant lr 1e-5, plus Dropout(0.4) before the final Linear and
p=0.5 horizontal flip; that sweep script is not in this repo):

| unique pairs | val cos | TAR@1e-3 | TAR@1e-4 | Top-1 |
|---|---|---|---|---|
| 50 | 0.3161 | 0.7308 | 0.3264 | 0.7759 |
| 200 | 0.4480 | 0.9829 | 0.8502 | 0.9720 |
| 800 | 0.5563 | 0.9976 | 0.9878 | 0.9951 |
| 3200 | 0.6489 | 0.9988 | 0.9976 | 0.9988 |
| 8000 | 0.7041 | 0.9988 | 0.9976 | 0.9988 |

Top-1 and TAR@1e-3 reach 0.99 at 800 pairs, TAR@1e-4 at 3200. Mean cosine is still rising
at 8000, so it has not saturated within the FFHQ512 budget.

---

## 9. Known gaps

1. **The onnx2torch / ONNXRuntime equivalence was never verified numerically.** Every
   Antelopev2 number above rests on that unchecked assumption. Worth spot-checking a few
   dozen images through both paths before trusting a new run.
2. **There is no genuine same-identity pair anywhere in this setup.** FFHQ has one image
   per identity, so the "upper bound" row is the original embedding against itself and is
   identically 1 by construction — it tests nothing. The only evidence that the extractor
   discriminates at all is the impostor distribution (mean 0.0068, std 0.0676). Moving to
   CASIA-WebFace or CelebA (`data_splits/index.txt` holds the CASIA split used by the
   other attacks in this repo) would close this.
3. 8821 pairs is far smaller than the ~200k CelebA setup the other attacks in this repo
   use.
4. The paper's Stage-1 (APIM) and Stage-2 (PEIT) training-pipeline reproduction is **not**
   in this repository — this repo covers the attack only.
5. The paper's Face++ / Amazon black-box evaluation and its user study were not
   reproduced.
