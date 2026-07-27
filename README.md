# FaceLinkGen

Code for **FaceLinkGen: Rethinking Identity Leakage in Privacy-Preserving Face Recognition**.

This repository is an organized copy of the research workspace. The experiment code was not refactored: files were moved by purpose, project-relative paths were updated for the new layout, machine-specific dataset paths were replaced with explicit `/path/to/...` placeholders, and API credentials were replaced with `YOUR_...` placeholders.

## Layout

- `attacks/`: distillation attacks for MinusFace, PartialFace, FracFace, CanFG, PerceptFace, and the transfer/proxy experiments.
- `methods/`: source used to produce protected representations for MinusFace, FracFace, CanFG, and PerceptFace.
- `evaluation/`: Arc2Face regeneration evaluation, soft-biometric evaluation, and compact result files.
- `notebooks/`: analysis notebooks grouped by target method or experiment.
- `data_splits/`: dataset split/index files used by the scripts.
- `third_party/`: the Arc2Face package and the subset of TFace imported by the MinusFace code.

## Local paths and models

The scripts intentionally retain their research-code style and hard-coded experiment settings. Replace the `/path/to/...` dataset paths inline before running them. Run a script from its own directory so its relative output and checkpoint paths match the original experiments.

The large datasets, generated images, logs, downloaded model bundles, and trained checkpoints from the original 95 GB workspace are not included. Place the shared Antelopev2 ONNX model at `checkpoints/model.onnx` and the MinusFace stage-1 checkpoint at `checkpoints/minusface_stage1.pth`. Method-specific student checkpoints are read from and written to the corresponding directory under `attacks/`.

The PerceptFace attack is self-contained and does not use the `/path/to/...` convention: its source is vendored at `third_party/perceptface/`, its weights go in `checkpoints/perceptface/`, and its data in `data/perceptface/`. See **`methods/perceptface/REPRODUCE.md`** for the full recipe — weights, environment gotchas, run order, the pre-attack gate, reference numbers, and known gaps. The only path to edit is the raw FFHQ512 directory at the top of `methods/perceptface/prep_crops.py`.

Arc2Face model download locations remain in the comments at the top of the regeneration scripts. Face++ and Eden AI credentials are represented by `YOUR_FACEPP_API_KEY`, `YOUR_FACEPP_API_SECRET`, and `YOUR_EDENAI_TOKEN`.

Third-party source retains its upstream README and license files where supplied.
