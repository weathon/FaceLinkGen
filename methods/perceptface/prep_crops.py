"""FFHQ512 -> 224 arcface-aligned crops, using PerceptFace's own Face_detect_crop.

Run from this directory. SRC is the only path you have to fill in: it is the raw FFHQ512
image directory, which is not shipped with this repo. Everything else resolves inside the
repo (third_party/perceptface for the code, checkpoints/perceptface for SCRFD,
data/perceptface for the output; both of the latter are gitignored).

det_size = (512,512) = image size. No padding anywhere. Images with no SCRFD detection
are skipped and recorded in skipped.txt. Resumable: an existing crop is not redone.
"""
import os
import sys
import cv2
import multiprocessing as mp

sys.path.insert(0, '../../third_party/perceptface')

SRC = '/path/to/FFHQ512/images'
DST = '../../data/perceptface/crops224'
DET_ROOT = '../../checkpoints/perceptface'  # holds antelope/scrfd_10g_bnkps.onnx
CORRUPT = {'08828.png'}  # PNG IDAT checksum error in the FFHQ512 release used here

app = None


def init_worker():
    global app
    from insightface_func.face_detect_crop_single import Face_detect_crop
    app = Face_detect_crop(name='antelope', root=DET_ROOT)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(512, 512), mode='None')


def run_one(name):
    img = cv2.imread(os.path.join(SRC, name))
    if img is None:
        raise RuntimeError('unreadable image: ' + name)
    out = app.get(img, 224)
    if out is None:
        return name, False
    cv2.imwrite(os.path.join(DST, name), out[0][0])
    return name, True


if __name__ == '__main__':
    os.makedirs(DST, exist_ok=True)
    names = sorted(n for n in os.listdir(SRC) if n.endswith('.png') and n not in CORRUPT)
    todo = [n for n in names if not os.path.exists(os.path.join(DST, n))]
    print('candidates %d, todo %d' % (len(names), len(todo)), flush=True)

    skipped = []
    done = 0
    with mp.Pool(64, initializer=init_worker) as pool:
        for name, ok in pool.imap_unordered(run_one, todo, chunksize=8):
            done += 1
            if not ok:
                skipped.append(name)
                print('NO FACE ' + name, flush=True)
            if done % 1000 == 0:
                print('%d/%d skipped=%d' % (done, len(todo), len(skipped)), flush=True)

    with open(os.path.dirname(DST) + '/skipped.txt', 'a') as f:
        for n in sorted(skipped):
            f.write(n + '\n')
    print('DONE cropped=%d skipped=%d on_disk=%d'
          % (len(todo) - len(skipped), len(skipped), len(os.listdir(DST))), flush=True)
