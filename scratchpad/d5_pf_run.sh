#!/bin/bash
# PerceptFace end to end on one GPU: masks -> Stage1 400K -> Stage2 400K.
# Each training script stops itself at MAX_SECONDS (6 h) and writes ckpt.pt every 1000
# steps with optimizer + wandb id, so this just relaunches until TOTAL_STEP is reached.
# A non-zero exit from a training script is a real failure (the normal MAX_SECONDS stop
# exits 0), so set -e aborts the chain instead of relaunching into the same crash.
set -e
export PYTHONPATH=/home/wg25r/face_deid/PerceptFace/pylibs
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
PY=/home/wg25r/face_deid/.venv/bin/python
S=/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/scratchpad
L=/raid/wg25r/redteam_work/logs
TOTAL=400000

step_of () {   # $1 = ckpt path; 0 if it does not exist yet
    $PY -c "
import os, torch, sys
p = sys.argv[1]
print(torch.load(p, map_location='cpu', weights_only=False)['step'] if os.path.exists(p) else 0)
" "$1"
}

echo "=== masks $(date)"
$PY $S/d2_pf_masks.py >> $L/d2_pf_masks.log 2>&1

for STAGE in 1 2; do
    SCRIPT=$S/d3_pf_stage1.py
    [ $STAGE -eq 2 ] && SCRIPT=$S/d4_pf_stage2.py
    CK=/raid/wg25r/redteam_work/ckpt/pf_stage$STAGE/ckpt.pt
    # set -e does not apply to a `while` condition, so a step_of that dies (truncated
    # ckpt -- torch.save is not atomic and this design is kill-and-relaunch) would make
    # the test error out and the loop exit as if the stage were finished. Check explicitly.
    while true; do
        CUR=$(step_of $CK)
        case "$CUR" in ''|*[!0-9]*) echo "step_of $CK returned '$CUR'"; exit 1;; esac
        [ "$CUR" -ge $TOTAL ] && break
        echo "=== stage$STAGE from step $CUR $(date)"
        $PY $SCRIPT >> $L/d_pf_stage$STAGE.log 2>&1
    done
    echo "=== stage$STAGE done at step $CUR $(date)"
done
echo "=== PF ALL DONE $(date)"
