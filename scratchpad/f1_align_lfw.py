"""f1: LFW deepfunneled -> arcface-aligned 112x112 flat dir, plus the 6000-pair annotation.

Detector is insightface buffalo_l. LFW is 250x250 and det_size must be a multiple of 32
and must not exceed the image size, so 224 (same rule as c1_crop_224.py). When several
faces are detected the one whose box centre is closest to the image centre wins -- LFW's
subject is centred by construction, the extras are bystanders in group shots.

Images with no detection are skipped whole and recorded; a pair referencing a skipped
image is dropped from lfw_ann.txt and counted. Resumable: an existing crop is not redone.

Output:
  data/lfw_112x112/<Name>_<NNNN>.jpg   flat, matches insight_test.py's os.listdir
  data/lfw_ann.txt                     "<label> <basename1> <basename2>", label 1 = same
"""
import csv
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

SRC = "/raid/wg25r/lfw/lfw-deepfunneled"
PAIRS = "/raid/wg25r/lfw/pairs.csv"
DST = "/raid/wg25r/fracface_rerun/data/lfw_112x112"
ANN = "/raid/wg25r/fracface_rerun/data/lfw_ann.txt"
SKIPFILE = "/raid/wg25r/fracface_rerun/data/lfw_skipped.txt"
DET = 224

os.makedirs(DST, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(DET, DET))

rels = sorted(p + "/" + f for p in os.listdir(SRC) if not p.startswith(".")
              for f in os.listdir(os.path.join(SRC, p)))
known_skip = set(open(SKIPFILE).read().split()) if os.path.exists(SKIPFILE) else set()
todo = [r for r in rels
        if r not in known_skip and not os.path.exists(os.path.join(DST, os.path.basename(r)))]
print("lfw images %d, known-skipped %d, todo %d" % (len(rels), len(known_skip), len(todo)), flush=True)

skipped = []
for i, rel in enumerate(todo):
    img = cv2.imread(os.path.join(SRC, rel))
    if img is None:
        raise RuntimeError("unreadable image: " + rel)
    faces = app.get(img)
    if len(faces) == 0:
        skipped.append(rel)
        print("NO FACE " + rel, flush=True)
        continue
    cy, cx = img.shape[0] / 2, img.shape[1] / 2
    face = min(faces, key=lambda f: (f.bbox[[0, 2]].mean() - cx) ** 2 + (f.bbox[[1, 3]].mean() - cy) ** 2)
    cv2.imwrite(os.path.join(DST, os.path.basename(rel)), face_align.norm_crop(img, face.kps, image_size=112))
    if (i + 1) % 2000 == 0:
        print("%d/%d skipped=%d" % (i + 1, len(todo), len(skipped)), flush=True)

with open(SKIPFILE, "w") as f:
    for r in sorted(known_skip | set(skipped)):
        f.write(r + "\n")

# --- pairs: matched rows are (name, n1, n2, ''), mismatched are (name1, n1, name2, n2)
have = set(os.listdir(DST))
rows = list(csv.reader(open(PAIRS)))[1:]
lines, dropped = [], 0
for r in rows:
    if r[3] == "":
        label, a, b = 1, "%s_%04d.jpg" % (r[0], int(r[1])), "%s_%04d.jpg" % (r[0], int(r[2]))
    else:
        label, a, b = 0, "%s_%04d.jpg" % (r[0], int(r[1])), "%s_%04d.jpg" % (r[2], int(r[3]))
    if a in have and b in have:
        lines.append("%d %s %s" % (label, a, b))
    else:
        dropped += 1

with open(ANN, "w") as f:
    f.write("\n".join(lines) + "\n")

print("DONE crops_on_disk=%d skipped_total=%d pairs_kept=%d pairs_dropped=%d"
      % (len(have), len(known_skip | set(skipped)), len(lines), dropped), flush=True)
