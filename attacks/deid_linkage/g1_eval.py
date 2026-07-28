"""Identity-level retrieval on LFW.

gallery = 10723 CLEAN LFW crops -> frozen AdaFace teacher. Same for every method.
query   = 1550 PROTECTED LFW crops (one per identity that has >=2 images in the pool).

Three rows:
  before attack  protected query -> teacher
  after attack   protected query -> the distilled student
  upper bound    clean query     -> teacher

rank_best = the rank (1-based) of the highest-ranked gallery image of the query's identity.
K = ceil(0.005 * |gallery|). Both readings of the top-K number are reported:
  topK_hit    1[rank_best <= K]                        -- did any same-identity image land in top K
  topK_recall |topK cap same_id| / |same_id in gallery| -- what fraction of them did
avg_rank is normalised by |gallery| (perfect ~1/|gallery|, random ~0.5).

Usage: python g1_eval.py {perceptface|canfg|canfg_ano|tipim} {100|200|500|2000} [distill|converge]
  distill   (default) the fixed-5000-step student, ckpt/distill_<m>_<n>/ckpt.pt
  converge  the early-stopped best-val student, ckpt/converge_<m>_<n>/best.pt
"""
import os
import sys
import json
import math
import torch
import numpy as np

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/attacks/deid_linkage')
from adaface_wrap import load_adaface, read_112

WORK = '/raid/wg25r/redteam_work'
METHOD, NPAIRS = sys.argv[1], int(sys.argv[2])
KIND = sys.argv[3] if len(sys.argv) > 3 else 'distill'
CKPT = {'distill': '%s/ckpt/distill_%s_%d/ckpt.pt', 'converge': '%s/ckpt/converge_%s_%d/best.pt'}
TAG = {'distill': '', 'converge': '_converge'}
device = 'cuda'

gal_rel = sorted(open(WORK + '/splits/lfw_gallery.txt').read().split())
qry_rel = sorted(open(WORK + '/splits/lfw_query.txt').read().split())
gal_name = [os.path.splitext(r.replace('/', '__'))[0] + '.png' for r in gal_rel]
qry_name = [os.path.splitext(r.replace('/', '__'))[0] + '.png' for r in qry_rel]
gal_id = np.array([r.split('/')[0] for r in gal_rel])
qry_id = np.array([r.split('/')[0] for r in qry_rel])
K = math.ceil(0.005 * len(gal_rel))
print('gallery %d, query %d, K %d' % (len(gal_rel), len(qry_rel), K), flush=True)

teacher = load_adaface(device)
teacher.requires_grad_(False)
student = load_adaface(device)
ck = torch.load(CKPT[KIND] % (WORK, METHOD, NPAIRS),
                map_location='cpu', weights_only=False)
student.load_state_dict(ck['student'])
student.eval()
student.requires_grad_(False)
print('student ckpt at step %d' % ck['step'], flush=True)


def embed(net, root, names):
    out = []
    for i in range(0, len(names), 256):
        x = torch.stack([read_112(os.path.join(root, n)) for n in names[i:i + 256]])
        with torch.no_grad():
            out.append(net(x.to(device))[0].cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()


G = embed(teacher, WORK + '/crops/lfw/112', gal_name)                       # clean gallery
PROT = '%s/protected/%s/lfw' % (WORK, METHOD)
rows = {
    'before attack': embed(teacher, PROT, qry_name),
    'after attack': embed(student, PROT, qry_name),
    'upper bound': embed(teacher, WORK + '/crops/lfw/112', qry_name),
}

res = {'method': METHOD, 'n_pairs': NPAIRS, 'kind': KIND, 'gallery': len(gal_rel),
       'query': len(qry_rel), 'K': K, 'student_step': ck['step']}
Gt = torch.from_numpy(G).to(device)
for label, Q in rows.items():
    sims = torch.from_numpy(Q).to(device) @ Gt.T                            # [q, g]
    order = sims.argsort(dim=1, descending=True).cpu().numpy()
    same = gal_id[order] == qry_id[:, None]                                 # [q, g] bool

    rank_best = same.argmax(1) + 1                                          # every query has >=1
    n_same = same.sum(1)
    topk_hit = rank_best <= K
    topk_recall = same[:, :K].sum(1) / n_same
    rank_all = np.concatenate([np.nonzero(row)[0] + 1 for row in same])

    res[label] = {
        'top1_hit': float((rank_best == 1).mean()),
        'topK_hit': float(topk_hit.mean()),
        'topK_recall': float(topk_recall.mean()),
        'avg_rank_best': float((rank_best / len(gal_rel)).mean()),
        'avg_rank_all': float((rank_all / len(gal_rel)).mean()),
        'median_rank_best': float(np.median(rank_best) / len(gal_rel)),
        'median_rank_best_raw': float(np.median(rank_best)),
    }
    print('%-14s top1 %.4f  topK_hit %.4f  topK_recall %.4f  avg_rank_best %.6f  avg_rank_all %.6f'
          % (label, res[label]['top1_hit'], res[label]['topK_hit'], res[label]['topK_recall'],
             res[label]['avg_rank_best'], res[label]['avg_rank_all']), flush=True)

os.makedirs(WORK + '/results', exist_ok=True)
out = '%s/results/retrieval_%s_%d%s.json' % (WORK, METHOD, NPAIRS, TAG[KIND])
json.dump(res, open(out, 'w'), indent=2)
print('-> ' + out)
