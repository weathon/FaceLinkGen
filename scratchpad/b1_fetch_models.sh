#!/bin/bash
# AdaFace (attack side) + TIP-IM (protection side) source and weights.
set -e
W=/raid/wg25r/redteam_work/weights
T=/raid/wg25r/redteam_work/third_party
PY=/home/wg25r/face_deid/.venv/bin/python
mkdir -p $W $T

echo "=== clone AdaFace $(date)"
[ -d $T/AdaFace ] || git clone --depth 1 https://github.com/mk-minchul/AdaFace.git $T/AdaFace

echo "=== clone TIP-IM $(date)"
[ -d $T/TIP-IM ] || git clone --depth 1 https://github.com/ShawnXYang/TIP-IM.git $T/TIP-IM

echo "=== AdaFace ir101 webface12m ckpt $(date)"
[ -f $W/adaface_ir101_webface12m.ckpt ] || \
    $PY -m gdown 1dswnavflETcnAuplZj1IOKKP0eM8ITgT -O $W/adaface_ir101_webface12m.ckpt

echo "=== TIP-IM ArcFace IR-SE50 $(date)"
mkdir -p $T/TIP-IM/ckpts
[ -f $T/TIP-IM/ckpts/model_ir_se50.pth ] || \
    curl -f -L -C - -o $T/TIP-IM/ckpts/model_ir_se50.pth \
        http://ml.cs.tsinghua.edu.cn/~xiaoyang/face_models/ArcFace/model_ir_se50.pth

ls -la $W $T/TIP-IM/ckpts
echo "=== ALL DONE $(date)"
