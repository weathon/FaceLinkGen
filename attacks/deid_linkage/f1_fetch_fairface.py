"""f1: fetch FairFace config 0.25 via the datasets library (parquet shards)."""
from datasets import load_dataset

ds = load_dataset("HuggingFaceM4/FairFace", "0.25")
print(ds)
print("features:", ds["train"].features)
print("train[0] keys:", {k: (v if k != "image" else type(v).__name__) for k, v in ds["train"][0].items()})
print("DONE")
