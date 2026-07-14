#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-val}"

# CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n neg python -u insight_test.py
# CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n neg python -u insight_test_partial.py
# CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n minus_face python -u insight_test_minus.py

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n py311 python -u eval_arc2face_blackbox.py --embeddings "${MODE}_frac.pkl" &
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n py311 python -u eval_arc2face_blackbox.py --embeddings "${MODE}_partial.pkl" &
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n py311 python -u eval_arc2face_blackbox.py --embeddings "${MODE}_minus.pkl" &
wait
