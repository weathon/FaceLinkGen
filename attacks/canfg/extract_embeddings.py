"""Pre-extract InsightFace teacher embeddings for a random 10k subset on CUDA."""
import os
import cv2
import pickle
import random
import numpy as np
import tqdm
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='antelopev2', root='../../checkpoints/insightface', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

def get_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image", image_path)
        return None
    image = cv2.resize(image, (160, 160))
    faces = app.get(image)
    if len(faces) == 0:
        print("No face found in", image_path)
        return None
    return faces[0].embedding.astype(np.float32)

root = '../../data/canfg/CelebA/protected_A/'
available_images = set(os.listdir(root))


all_paths = []
with open("/path/to/list_eval_partition.csv", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        filename, split = line.split(",")
        celeba_name = "img_align_celeba_" + filename.replace(".jpg", ".png")
        if celeba_name not in available_images:
            continue
        all_paths.append(os.path.join(root, celeba_name))



import pandas as pd
map_ids = pd.read_csv('/path/to/Identity_CelebA.txt', sep=' ')

all_ids = map_ids["Label"]
train_ids = random.sample(list(all_ids), min(10000, len(all_ids)))
val_ids = [x for x in all_ids if x not in train_ids]


# Randomly select 10k
random.seed(42)
selected_paths = all_paths# random.sample(all_paths, min(20000, len(all_paths)))

train_path = []
val_path = []
for path in selected_paths:
    img_name = os.path.basename(path)
    row = map_ids[map_ids["Image"] == img_name]
    if row.empty:
        continue
    id_val = row["Label"].iloc[0]
    if id_val in train_ids:
        train_path.append(path)
    elif id_val in val_ids:
        val_path.append(path)



os.makedirs("log", exist_ok=True)
with open("log/train_paths.pkl", "wb") as f:
    pickle.dump(train_path, f)
with open("log/val_paths.pkl", "wb") as f:
    pickle.dump(val_path, f)
print(f"Train: {len(train_path)}, Val: {len(val_path)}")

embeddings = {}
failed = 0
for path in tqdm.tqdm(selected_paths, desc="Extracting embeddings"):
    emb = get_embedding(path)
    if emb is None:
        failed += 1
        embeddings[path] = np.zeros(512, dtype=np.float32)
    else:
        embeddings[path] = emb

print(f"Done. {len(embeddings)} embeddings extracted, {failed} failed.")

with open("log/teacher_embeddings_insight.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Saved to log/teacher_embeddings_insight.pkl")
