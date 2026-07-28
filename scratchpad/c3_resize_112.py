"""224 arcface crops -> 112, the working resolution for TIP-IM.

TIP-IM's white-box FR (IR-SE50) is native 112, and attacking at 112 keeps the perturbation
from being resampled. The 224 crops are already arcface-aligned, so this is a plain
downscale, no second detection.

Usage: python c3_resize_112.py {ffhq|lfw}
"""
import os
import sys
import cv2
import multiprocessing as mp

WORK = '/raid/wg25r/redteam_work'
WHICH = sys.argv[1]
SRC = WORK + '/crops/' + WHICH + '/224'
DST = WORK + '/crops/' + WHICH + '/112'


def run_one(name):
    img = cv2.imread(os.path.join(SRC, name))
    if img is None:
        raise RuntimeError('unreadable crop: ' + name)
    cv2.imwrite(os.path.join(DST, name),
                cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR))


if __name__ == '__main__':
    os.makedirs(DST, exist_ok=True)
    names = sorted(os.listdir(SRC))
    todo = [n for n in names if not os.path.exists(os.path.join(DST, n))]
    print('%s: candidates %d, todo %d' % (WHICH, len(names), len(todo)), flush=True)
    with mp.Pool(64) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_one, todo, chunksize=64)):
            if (i + 1) % 10000 == 0:
                print('%d/%d' % (i + 1, len(todo)), flush=True)
    print('DONE %s on_disk=%d' % (WHICH, len(os.listdir(DST))), flush=True)
