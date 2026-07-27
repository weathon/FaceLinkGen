# PerceptFace (vendored source)

Source: <https://huggingface.co/spaces/daizigege/PerceptFace>, the official HuggingFace
Space for **"Make Identity Unextractable yet Perceptible"** (arXiv 2509.11249v1).

Vendored because the authors' GitHub repo `daizigege/PerceptFace` is missing several
files that its own code imports (`insightface_func/utils/face_align_ffhqandnewarc.py`,
`insightface_func/__init__.py`, `util/util.py` are all 404), while the Space is complete
and runnable. The Space names the main class `AIDPro` (in `AIDPro_MSE.py`); the GitHub
repo calls the same class `PerceptFace`. `methods/perceptface/` imports the Space's
`AIDPro_MSE.ID_transform` and `fs_networks_fix.Generator_Adain_Upsample`.

**No LICENSE file is supplied upstream** — the Space's `README.md` is only Gradio
frontmatter. The code is included here unmodified, for research reproduction of the
attack in this repository. `fs_networks_fix.py`, `insightface_func/`, `util/` and
`models/` are themselves derived from SimSwap (<https://github.com/neuralchen/SimSwap>,
CC BY-NC 4.0).

Weights are **not** committed. Fetch them from the same Space into
`checkpoints/perceptface/` — see `methods/perceptface/REPRODUCE.md`.

Files removed from the upstream copy: `pretrained_models/` and
`insightface_func/models/*.onnx` (weights, gitignored elsewhere), `__pycache__/`,
`.gitattributes`.
