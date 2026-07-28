"""Average query-to-gallery cosine, to sit alongside the retrieval metrics in g1_eval.

Same embeddings and same three rows as g1_eval.py. For each query q with same-identity
gallery set S_q:
    genuine_q  = mean_{g in S_q} cos(q, g)
    impostor_q = mean_{g not in S_q} cos(q, g)
and both are averaged over queries (macro). The impostor number is the scale reference:
the gallery is fixed and clean, so it is what "no identity signal" looks like.

Usage: python g2_cosine.py {method} {n} [distill|converge]
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/attacks/deid_linkage')
from adaface_wrap import load_adaface, read_112

WORK = '/raid/wg25r/redteam_work'
METHOD, NPAIRS = sys.argv[1], int(sys.argv[2])
KIND = sys.argv[3] if len(sys.argv) > 3 else 'converge'
CKPT = {'distill': '%s/ckpt/distill_%s_%d/ckpt.pt', 'converge': '%s/ckpt/converge_%s_%d/best.pt'}
TAG = {'distill': '', 'converge': '_converge'}
device = 'cuda'

gal_rel = sorted(open(WORK + '/splits/lfw_gallery.txt').read().split())
qry_rel = sorted(open(WORK + '/splits/lfw_query.txt').read().split())
gal_name = [os.path.splitext(r.replace('/', '__'))[0] + '.png' for r in gal_rel]
qry_name = [os.path.splitext(r.replace('/', '__'))[0] + '.png' for r in qry_rel]
gal_id = np.array([r.split('/')[0] for r in gal_rel])
qry_id = np.array([r.split('/')[0] for r in qry_rel])

teacher = load_adaface(device)
teacher.requires_grad_(False)
student = load_adaface(device)
ck = torch.load(CKPT[KIND] % (WORK, METHOD, NPAIRS), map_location='cpu', weights_only=False)
student.load_state_dict(ck['student'])
student.eval()
student.requires_grad_(False)


def embed(net, root, names):
    out = []
    for i in range(0, len(names), 256):
        x = torch.stack([read_112(os.path.join(root, n)) for n in names[i:i + 256]])
        with torch.no_grad():
            out.append(net(x.to(device))[0].cpu())
    return torch.nn.functional.normalize(torch.cat(out), dim=1)


G = embed(teacher, WORK + '/crops/lfw/112', gal_name).to(device)
PROT = '%s/protected/%s/lfw' % (WORK, METHOD)
rows = {
    'before attack': embed(teacher, PROT, qry_name),
    'after attack': embed(student, PROT, qry_name),
    'upper bound': embed(teacher, WORK + '/crops/lfw/112', qry_name),
}

same = torch.from_numpy(gal_id[None, :] == qry_id[:, None]).to(device)
res = {'method': METHOD, 'n_pairs': NPAIRS, 'kind': KIND}
for label, Q in rows.items():
    sims = Q.to(device) @ G.T
    gen = (sims * same).sum(1) / same.sum(1)
    imp = (sims * ~same).sum(1) / (~same).sum(1)
    res[label] = {'avg_cos_genuine': float(gen.mean()), 'avg_cos_impostor': float(imp.mean()),
                  'median_cos_genuine': float(gen.median())}
    print('%-14s avg_cos genuine %.4f  impostor %.4f  median genuine %.4f'
          % (label, gen.mean(), imp.mean(), gen.median()), flush=True)

out = '%s/results/cosine_%s_%d%s.json' % (WORK, METHOD, NPAIRS, TAG[KIND])
json.dump(res, open(out, 'w'), indent=2)
print('-> ' + out)
