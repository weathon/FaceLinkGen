"""FFHQ512 / LFW -> 224 arcface-aligned crops, using PerceptFace's own Face_detect_crop.

Adapted from methods/perceptface/prep_crops.py. det_size = the image size, no padding
anywhere (REPRODUCE.md section 5: SCRFD upsamples first when det_size exceeds the image
and detection collapses). Images with no detection are skipped whole and recorded.
Resumable: an existing crop is not redone.

Usage: python c1_crop_224.py {ffhq|lfw}
"""
import os
import sys
import cv2
import multiprocessing as mp

sys.path.insert(0, '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/third_party/perceptface')

DET_ROOT = '/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/checkpoints/perceptface'
WORK = '/raid/wg25r/redteam_work'
SETS = {
    'ffhq': ('/raid/wg25r/ffhq512_hf/images/FFHQ512/FFHQ512', 512),
    'lfw': ('/raid/wg25r/lfw/lfw-deepfunneled', 250),
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


def run_one(rel):
    img = cv2.imread(os.path.join(SRC, rel))
    if img is None:
        raise RuntimeError('unreadable image: ' + rel)
    out = app.get(img, 224)
    if out is None:
        return rel, False
    dst = os.path.join(DST, rel.replace('/', '__'))
    cv2.imwrite(dst, out[0][0])
    return rel, True


if __name__ == '__main__':
    os.makedirs(DST, exist_ok=True)
    if WHICH == 'ffhq':
        rels = sorted(n for n in os.listdir(SRC) if n.endswith('.png'))
    else:
        rels = sorted(p + '/' + f for p in os.listdir(SRC) if not p.startswith('.')
                      for f in os.listdir(os.path.join(SRC, p)))
    todo = [r for r in rels if not os.path.exists(os.path.join(DST, r.replace('/', '__')))]
    print('%s: candidates %d, todo %d' % (WHICH, len(rels), len(todo)), flush=True)

    skipped = []
    done = 0
    with mp.Pool(64, initializer=init_worker) as pool:
        for rel, ok in pool.imap_unordered(run_one, todo, chunksize=8):
            done += 1
            if not ok:
                skipped.append(rel)
            if done % 2000 == 0:
                print('%d/%d skipped=%d' % (done, len(todo), len(skipped)), flush=True)

    with open(WORK + '/crops/' + WHICH + '/skipped_224.txt', 'a') as f:
        for r in sorted(skipped):
            f.write(r + '\n')
    print('DONE %s cropped=%d skipped=%d on_disk=%d'
          % (WHICH, len(todo) - len(skipped), len(skipped), len(os.listdir(DST))), flush=True)
