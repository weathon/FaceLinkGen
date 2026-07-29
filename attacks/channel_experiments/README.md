# Fixed/random channel reconstruction experiments

This directory executes `new_plan.md` without reusing prior attack checkpoints,
reconstructions, API records, or visualizations.

The experiment matrix is:

| Protection | Channel modes |
|---|---|
| FracFace | fixed, random |
| PartialFace | fixed, random |
| MinusFace | random |

Each setting trains both attacks from the common pretrained models:

- `ours`: 2 epochs, full 70,692-image training split, batch 256, AdamW,
  learning rate `5e-4`, weight decay `5e-3`, cosine schedule.
- `unet`: 20 epochs, the same split/batch/optimizer/LR/weight decay, L1 loss,
  constant LR for epochs 1–15 and cosine decay for epochs 16–20.

`train_ours.py` and `train_unet.py` save every epoch and resume from
`resume.pt`. `dump_reconstructions.py` writes every one of the 300 evaluation
samples separately. `generate_arc2face.py` generates one face per ours
embedding, matching the single U-Net reconstruction per identity.
`eval_apis.py` appends one raw-response record per identity and reports the
single-image Face++/Amazon rates.

FracFace fixed mode calls `seed_everything(42)` before constructing its FSM;
random mode does not seed. PartialFace fixed mode uses the released hard-coded
6×9 channel partition; random mode draws one new 6×9 partition per batch.
MinusFace already shuffles channels and therefore has no fixed mode.

Training, embedding dumps, direct U-Net reconstruction, and API evaluation use
the `minus_face` conda environment. Arc2Face generation requires its official
Diffusers stack and uses the `arc2face` environment.

CASIA-WebFace is already face-cropped. Both attacks read those aligned images
directly and do not run an additional DeepFace crop.
