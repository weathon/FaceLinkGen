import pickle, os, torch
d = pickle.load(open("blackbox/val_minus_lfw.pkl", "rb"))
print("KEYS", list(d.keys()))
se = d["student_embeddings"]
print("SE_TYPE", type(se).__name__, "LEN", len(se))
e0 = se[0]
print("SE0_TYPE", type(e0).__name__, "SHAPE", tuple(e0.shape))
fn = d["filenames"]
flat = sum(fn, []) if isinstance(fn[0], (list, tuple)) else list(fn)
print("NFILES", len(flat))
print("EX0", flat[0])
print("EXISTS0", os.path.exists(flat[0]))
# total embeddings
tot = sum(x.shape[0] for x in se)
print("TOTAL_EMB", tot)
