# U-Net reconstruction attack (baseline)

A direct image-to-image reconstruction attack: a U-Net maps the protected representation
back to the original face under an L1 loss. It is the discriminative-attack baseline
alongside the Arc2Face regeneration evaluation in `evaluation/regeneration/`. Because the
U-Net emits an image directly, its output goes straight to the face-comparison APIs with no
generative step in between.

| File | Purpose |
|---|---|
| `main.ipynb`, `main_frac.ipynb`, `main_par.ipynb` | train against MinusFace / FracFace / PartialFace |
| `test_minus.ipynb`, `test_fracface.ipynb`, `test_par.ipynb` | dump val reconstructions to `artifacts/reconstructions/*.pkl` |
| `corrected_fracface.py` | paper-aligned variant of FracFace's FCR + FFM |
| `train_corrected_frac.py` | train against that variant |
| `eval_facepp_frac.py` | Face++ evaluation of one reconstruction dump |
| `eval_apis_all.py` | paired Face++ and Amazon evaluation across all methods |
| `RESULTS_fracface_ablation.md` | released vs. paper-aligned FracFace results |

Dataset paths are `/path/to/...` placeholders; `eval_apis_all.py` takes `--data_root`.
Checkpoints and reconstruction dumps are not in Git:

```bash
python ../../tools/hf_artifacts.py download checkpoints
python ../../tools/hf_artifacts.py download artifacts
```

Face++ and EdenAI credentials come from the repository-root `.env`.

## Evaluation criteria

**Face++** — a face detected in both images and confidence above the `1e-5` threshold. The
weaker `1e-4` / `1e-3` bands are recorded separately.

**Amazon** (EdenAI `face_compare`, `providers=amazon`) — CompareFaces returns a match.
Rekognition applies its `SimilarityThreshold` server-side, so a non-empty match list means
match. Amazon reports two distinct kinds of non-match and they must not be conflated:

- empty `items` — the faces were compared and judged different.
- `ProviderInvalidInputError` — no face was found in the reconstruction, so CompareFaces
  refused the request. This is an **attack failure, not an API failure**, and it counts in
  the denominator. Excluding it inflates the match rate of precisely the methods whose
  reconstructions are worst, which is not a uniform bias: on a 180-image partial run the
  counts were MinusFace 21, PartialFace 14, FracFace 8, variant 9.

Note that EdenAI reports similarity in `[0, 1]`, not Rekognition's native `0-100`, so the
`confidence > 80` test in `evaluation/regeneration/eval_arc2face_3methods.py` can never
fire and that script's Amazon column should not be relied on.

All methods are scored on `data_splits/val_minus_lfw_filenames.txt`, the 300-image subset
the Arc2Face evaluation uses, so the numbers are paired across methods and comparable with
the regeneration results.

## FracFace: released code vs. paper

We evaluate FracFace's officially released implementation as the primary setting. While
working with it we noticed several points where that code and the paper description differ;
we take no position on which is intended. `corrected_fracface.py` implements the paper-side
reading at each of them so the attack can be measured under both. The result is materially
unchanged — see `RESULTS_fracface_ablation.md` for the table and the interpretation
caveats that belong with it.
