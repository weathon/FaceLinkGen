"""TIP-IM protected images. Per-image MI-FGSM in ArcFace IR-SE50 embedding space.

Upstream has no training phase. The attack loop, the submodular target search, the Gain
functions and every hyperparameter are the upstream defaults from run.py:
  src_model ArcFace (IR-SE50), num_iter 100, target_nums 10, gain3, norm l2, gamma 0.0,
  epsilon 12/255 hardcoded, alpha = 1.5*epsilon/iters, momentum m = 1.0.

batch_size stays 1: upstream's submodular allocates gains with len(target_feas) == 10 but
indexes it with range(len(tmpadv_feas)) == batch*10, so it is only correct at batch 1.
Throughput comes from running several of these processes in parallel instead
(--shard i --nshards n splits the work list).

Two things differ from upstream, both forced:
  - alignment/re_align are skipped. Inputs are already the 112 arcface crops, so there is
    no unaligned original to warp the perturbation back onto; the protected image IS the
    112 crop. That also avoids the TF1 MTCNN in align_methods.
  - .contiguous() after permute, because ArcFace's output_layer Flatten uses .view().
input_diversify.py in the clone was patched: the unused scipy.misc import (removed from
SciPy) is gone and affine_grid/grid_sample now pass align_corners=True explicitly, which
was the default when the paper was written.

Usage: python d6_tipim.py --shard 0 --nshards 8
"""
import os
import sys
import argparse
import random
import numpy as np
import cv2
import torch

TIPIM = '/raid/wg25r/redteam_work/third_party/TIP-IM'
WORK = '/raid/wg25r/redteam_work'
os.chdir(TIPIM)                      # FaceModel loads './ckpts/model_ir_se50.pth'
sys.path.insert(0, TIPIM)
from get_model import getmodel
from input_diversify import input_diversity
from mmd import mmd_loss

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, required=True)
ap.add_argument('--nshards', type=int, required=True)
args = ap.parse_args()

ITERS = 100
TARGET_NUMS = 10
EPSILON = 12.0
ALPHA = 1.5 * EPSILON / ITERS
GAMMA = 0.0
MOMENTUM = 1.0
device = 'cuda'

model, img_shape = getmodel('ArcFace')


def load112(path):
    """0-255 float RGB HWC, the layout run.py feeds the model (Backbone does (x-127.5)/128)."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError('unreadable: ' + path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def L2distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2, dim=1))


def Gain3(adv_fea, init_feas, target_feas):
    d1 = L2distance(adv_fea, init_feas)
    d2 = torch.max(torch.exp(d1 - L2distance(adv_fea, target_feas)))
    return torch.log(1.0 + d2)


targets = sorted(open(WORK + '/splits/tipim_targets.txt').read().split())
tgt = np.stack([load112('%s/crops/ffhq/112/%s' % (WORK, n)) for n in targets])
with torch.no_grad():
    target_feas = model.forward(
        torch.Tensor(tgt).to(device).permute(0, 3, 1, 2).contiguous())

JOBS = [
    ('ffhq', WORK + '/splits/ffhq_attack_2000.txt'),
    ('ffhq', WORK + '/splits/ffhq_gate_val.txt'),
    ('lfw', WORK + '/splits/lfw_query.txt'),
]
work = []
for ds, split in JOBS:
    for rel in open(split).read().split():
        n = os.path.splitext(rel.replace('/', '__'))[0] + '.png'
        work.append((ds, n))
work = sorted(set(work))
mine = work[args.shard::args.nshards]
print('shard %d/%d: %d of %d images' % (args.shard, args.nshards, len(mine), len(work)),
      flush=True)

random.seed(args.shard)
np.random.seed(args.shard)
done = 0
for ds, name in mine:
    dst_dir = '%s/protected/tipim/%s' % (WORK, ds)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        continue

    aligned = load112('%s/crops/%s/112/%s' % (WORK, ds, name))[None]
    inputs = torch.Tensor(aligned).to(device).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        init_feas = model.forward(inputs)

    sum_grad = torch.zeros_like(inputs)
    min_img = torch.clamp(inputs - EPSILON, min=0)
    max_img = torch.clamp(inputs + EPSILON, max=255)
    adv_images = inputs.detach().clone().requires_grad_(True)

    for _ in range(ITERS):
        std_proj = random.uniform(0.01, 0.1)
        std_rotate = random.uniform(0.01, 0.1)
        tmp_advs, tmp_grads = [], []
        model.zero_grad()
        images = input_diversity(adv_images, std_proj, std_rotate)
        adv_feas = model.forward(images)
        loss_mmd = mmd_loss(adv_images.clone().reshape(adv_images.size(0), -1),
                            inputs.clone().reshape(inputs.size(0), -1))

        for idx in range(TARGET_NUMS):
            model.zero_grad()
            loss_i = torch.mean((adv_feas - init_feas) ** 2)
            loss_t = torch.mean((adv_feas - target_feas[idx]) ** 2)
            loss = loss_t - loss_i + GAMMA * loss_mmd
            loss.backward(retain_graph=True)
            grad = adv_images.grad.data.clone()
            grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
            tmp_sum_grad = MOMENTUM * sum_grad.clone() + grad
            adv_images.grad.data.zero_()

            factor = np.sqrt(np.prod(img_shape) * 3)
            grad2d = tmp_sum_grad.reshape((tmp_sum_grad.size(0), -1))
            grad_unit = grad2d / grad2d.norm(p=2, dim=1, keepdim=True)
            delta = -torch.reshape(grad_unit, tmp_sum_grad.size()) * ALPHA * factor
            tmp = torch.min(torch.max(adv_images.data.clone() + delta, min_img), max_img)

            tmp_grads.append(tmp_sum_grad)
            tmp_advs.append(tmp)

        with torch.no_grad():
            feas = model.forward(torch.cat(tmp_advs))
            gains = torch.stack([Gain3(feas[i].unsqueeze(0), init_feas, target_feas)
                                 for i in range(len(feas))])
        best = int(torch.argmax(gains))
        sum_grad = tmp_grads[best]
        adv_images = tmp_advs[best].detach().requires_grad_(True)

    out = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy()[0]
    # Write then rename: cv2.imwrite is not atomic and resume skips by file existence, so
    # a kill mid-write would leave a truncated PNG that is skipped forever and consumed
    # downstream as a valid protected image.
    tmp_path = dst + '.part%d.png' % args.shard   # cv2 picks the writer from the extension
    cv2.imwrite(tmp_path, cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    os.replace(tmp_path, dst)
    done += 1
    if done % 25 == 0:
        print('shard %d: %d/%d' % (args.shard, done, len(mine)), flush=True)

print('shard %d DONE %d written' % (args.shard, done), flush=True)
