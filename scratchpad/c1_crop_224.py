"""FFHQ512 / LFW -> 224 arcface-aligned crops, using PerceptFace's own Face_detect_crop.

Adapted from methods/perceptface/prep_crops.py. No padding anywhere. det_size never
exceeds the image size (REPRODUCE.md section 5: SCRFD upsamples first when it does and
detection collapses) and must be a multiple of 32 — see the SETS comment. FFHQ 512x512
uses 512 exactly; LFW 250x250 uses 224, so LFW is downscaled before detection.
Images with no detection are skipped whole and recorded. Resumable.

Usage: python c1_crop_224.py {ffhq|lfw}
"""
import os
import sys
import cv2
import multiprocessing as mp

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/third_party/perceptface')

DET_ROOT = '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/checkpoints/perceptface'
WORK = '/raid/wg25r/redteam_work'
# det_size must be a multiple of 32 (SCRFD builds its anchor grid as input_h // stride;
# 250 raises "operands could not be broadcast together with shapes (450,) (512,)"), and
# must not exceed the image size or SCRFD upsamples first and detection collapses.
# FFHQ is 512x512 -> 512. LFW is 250x250 -> 224, the largest multiple of 32 that fits.
SETS = {
    'ffhq': ('/raid/wg25r/ffhq512_hf/images/FFHQ512/FFHQ512', 512),
    'lfw': ('/raid/wg25r/lfw/lfw-deepfunneled', 224),
}
WHICH = sys.argv[1]
SRC, DET = SETS[WHICH]
DST = WORK + '/crops/' + WHICH + '/224'

app = None


def init_worker():
    global app
    from insightface_func.face_detect_crop_single import Face_detect_crop
    app = Face_detect_crop(name='antelope', root=DET_ROOT)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(DET, DET), mode='None')


def crop_name(rel):
    """LFW is <Name>/<Name>_NNNN.jpg, FFHQ is NNNNN.png. Flatten the path and always
    write PNG so the train and test crops go through the same (lossless) codec."""
    return os.path.splitext(rel.replace('/', '__'))[0] + '.png'


def run_one(rel):
    img = cv2.imread(os.path.join(SRC, rel))
    if img is None:
        raise RuntimeError('unreadable image: ' + rel)
    out = app.get(img, 224)
    if out is None:
        return rel, False
    cv2.imwrite(os.path.join(DST, crop_name(rel)), out[0][0])
    return rel, True


if __name__ == '__main__':
    os.makedirs(DST, exist_ok=True)
    if WHICH == 'ffhq':
        rels = sorted(n for n in os.listdir(SRC) if n.endswith('.png'))
    else:
        rels = sorted(p + '/' + f for p in os.listdir(SRC) if not p.startswith('.')
                      for f in os.listdir(os.path.join(SRC, p)))

    # A skipped image produces no crop file, so without this it would be re-detected on
    # every resume and appended to the log again. The detector is deterministic.
    SKIPFILE = WORK + '/crops/' + WHICH + '/skipped_224.txt'
    known = set()
    if os.path.exists(SKIPFILE):          # resume state; absent on the first run
        known = set(open(SKIPFILE).read().split())
    todo = [r for r in rels
            if r not in known and not os.path.exists(os.path.join(DST, crop_name(r)))]
    print('%s: candidates %d, known-skipped %d, todo %d'
          % (WHICH, len(rels), len(known), len(todo)), flush=True)

    skipped = []
    done = 0
    with mp.Pool(64, initializer=init_worker) as pool:
        for rel, ok in pool.imap_unordered(run_one, todo, chunksize=8):
            done += 1
            if not ok:
                skipped.append(rel)
                print('NO FACE ' + rel, flush=True)
            if done % 2000 == 0:
                print('%d/%d skipped=%d' % (done, len(todo), len(skipped)), flush=True)

    with open(SKIPFILE, 'w') as f:
        for r in sorted(known | set(skipped)):
            f.write(r + '\n')
    print('DONE %s cropped=%d skipped_this_run=%d skipped_total=%d on_disk=%d'
          % (WHICH, len(todo) - len(skipped), len(skipped),
             len(known | set(skipped)), len(os.listdir(DST))), flush=True)
