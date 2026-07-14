
import wandb
import os
wandb.init(project="minusface")
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(128, 128))
def is_same_person(img1_path, img2_path, threshold=0.3, model="buffalo_l"):
    def get_embedding(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128))
        for det_thresh in [0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.00]:
            app.det_model.det_thresh = det_thresh
            faces = app.get(img, max_num=8)
            max_area = 0
            max_area_idx = 0
            for idx, face in enumerate(faces):
                box = face.bbox 
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    max_area_idx = idx
            if len(faces) == 0:
                continue
            else:
                break
        return faces[max_area_idx].normed_embedding
    emb1, emb2 = get_embedding(img1_path), get_embedding(img2_path)

    sim = float(np.dot(emb1, emb2))
    return {
       "distance": 1 - sim,
       "emb1": emb1,
       "emb2": emb2
    }
# sim, emb1, emb2 = is_same_person("image1.png", "image2.png")
# sim

ranks = []
dist = []
db_embeddings = []
residue_embeddings = []
# for test_id in range(1, len(os.listdir("/path/to/FaceLinkGen/original_pair_face/"))-1):
#   dis = []
#   result = is_same_person(
#         img1_path = f"/path/to/FaceLinkGen/original_pair_face/{test_id:04d}.png",
#         img2_path = f"./results/{test_id:04d}.png",
#   )
#   dis.append(result["distance"])
#   print(dis[-1]) 
#   import os
#   import tqdm
#   import numpy as np
#   # for image_name in tqdm.tqdm(sorted(os.listdir("/path/to/FaceLinkGen/original_face/"))):
#   for image_name in tqdm.tqdm(sorted(os.listdir("/path/to/FaceLinkGen/original_pair_face/"))):
#     if image_name == f"{test_id:04d}.png":
#       continue
#     img2_path = f"/path/to/FaceLinkGen/original_pair_face/{image_name}"
#     img1_path = f"./results/{test_id:04d}.png" 
#     print(img1_path, img2_path)
#     result = is_same_person(
#       img1_path = img1_path,  
#       img2_path = img2_path,
#     )
#     dis.append(result["distance"])  
#     print(f"{image_name}: {result['distance']}") 
#     dist.append(dis[-1])
#     wandb.log({"distance_rank": np.where(np.argsort(dis)==0)[0]/len(dis), "dis_histgram": wandb.Histogram(np.array(dist))})
#   ranks.append(np.where(np.argsort(dis)==0)[0][0] / len(dis))
#   wandb.log({"final_rank": ranks[-1], "rank_histgram": wandb.Histogram(np.array(ranks))})   
#   print(f"final rank: {ranks[-1]}") 


import os
import tqdm
import numpy as np
# for image_name in tqdm.tqdm(sorted(os.listdir("/path/to/FaceLinkGen/original_face/"))):
for image_name in tqdm.tqdm(sorted(os.listdir("/path/to/FaceLinkGen/original_pair_face/"))):
    img2_path = f"/path/to/FaceLinkGen/original_pair_face/{image_name}"
    img1_path = f"./results/{image_name}" 
    print(img1_path, img2_path)
    result = is_same_person(
        img1_path = img1_path,  
        img2_path = img2_path,
    )
    print(f"{image_name}: {result['distance']}") 
    db_embeddings.append(result["emb2"])
    residue_embeddings.append(result["emb1"])

import pickle   
with open("results/embeddings.pkl", "wb") as f:
    pickle.dump({
        "db": db_embeddings,
        "residue": residue_embeddings
    }, f)