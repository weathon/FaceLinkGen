#!/bin/bash
# Train: FFHQ512 (RichardErkhov/FFHQ512, webdataset tar, 26.4 GiB)
# Test:  LFW deepfunneled (DerrickUnleashed/LFW, 120 MiB) + its identity csvs
set -e
PY=/home/wg25r/face_deid/.venv/bin/python
F=/raid/wg25r/ffhq512_hf
L=/raid/wg25r/lfw
mkdir -p $F $L

echo "=== FFHQ512.tar $(date)"
$PY - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download('RichardErkhov/FFHQ512', 'FFHQ512.tar', repo_type='dataset',
                    local_dir='/raid/wg25r/ffhq512_hf', resume_download=True)
print('downloaded ->', p)
EOF

echo "=== extract FFHQ512.tar $(date)"
if [ ! -f $F/.extracted ]; then
    mkdir -p $F/images
    tar -xf $F/FFHQ512.tar -C $F/images
    touch $F/.extracted
fi
echo "FFHQ entries on disk: $(ls $F/images | wc -l)"

echo "=== LFW deepfunneled $(date)"
B=https://huggingface.co/datasets/DerrickUnleashed/LFW/resolve/main
# `[ -f ] ||` would short-circuit on a half-written file from an interrupted run and
# accept it as complete, so -C - would never get a chance to resume. Sentinel instead.
for f in lfw-deepfunneled.zip lfw_allnames.csv people.csv pairs.csv matchpairsDevTest.csv mismatchpairsDevTest.csv; do
    if [ ! -f $L/$f.done ]; then
        curl -f -L -C - -o $L/$f $B/$f
        touch $L/$f.done
    fi
done
if [ ! -f $L/.extracted ]; then
    unzip -n -q $L/lfw-deepfunneled.zip -d $L
    touch $L/.extracted
fi
echo "LFW identity dirs: $(ls $L/lfw-deepfunneled | wc -l)"
echo "=== ALL DONE $(date)"
