import os, sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
sys.path.insert(0,"../../methods/fracface")
import data2npy, torch, numpy as np
from PIL import Image
img = Image.open("/path/to/casia-webface/"+open("../../data_splits/index.txt").readline().split()[0]).convert("RGB").resize((112,112))
# try different 2nd-arg values (black-box analysis only)
for a in [0,1,2,3,5]:
    try:
        r = data2npy.preprocess_and_return(img, a)
        o = r[0]
        print(f"arg={a}: type={type(o).__name__} shape={tuple(o.shape)} range=[{float(o.min()):.3f},{float(o.max()):.3f}] mean={float(o.mean()):.3f} std={float(o.std()):.3f} len(ret)={len(r)}")
    except Exception as e:
        print(f"arg={a}: ERR {type(e).__name__}: {str(e)[:80]}")
# also inspect the 81-channel structure for arg=1
o = data2npy.preprocess_and_return(img,1)[0]
print("per-channel std (first 10):", [round(float(o[i].std()),3) for i in range(min(10,o.shape[0]))])
print("mean-over-ch std:", float(o.mean(0).std()), " vs single ch0 std:", float(o[0].std()))
