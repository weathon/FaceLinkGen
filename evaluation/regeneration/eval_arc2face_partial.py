# %%
# from huggingface_hub import hf_hub_download

# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="../../checkpoints/arc2face")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="../../checkpoints/arc2face")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="../../checkpoints/arc2face")
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="../../checkpoints/arc2face")

# %%
# %pip install -U onnxruntime-gpu

# %%
# url = "https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view"
# import gdown
# gdown.download(url, "antelopev2.zip", quiet=False, fuzzy=True)

# %%
import numpy as np
from PIL import Image

def vstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = max([img.size[0] for img in images])
    height = sum([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    y_pos = 0
    for img in images:
        stacked.paste(img, (0, y_pos))
        y_pos += img.size[1]
    return stacked


def hstack(images):
    if len(images) == 0:
        raise ValueError("Need 0 or more images")

    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    width = sum([img.size[0] for img in images])
    height = max([img.size[1] for img in images])
    stacked = Image.new(images[0].mode, (width, height))

    x_pos = 0
    for img in images:
        stacked.paste(img, (x_pos, 0))
        x_pos += img.size[0]
    return stacked

# %%
# !unzip -d ./models/antelopev2 antelopev2.zip

# %%
# hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="../../checkpoints/arc2face/antelopev2")

# %%
import sys
sys.path.append("../../third_party/arc2face")
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

encoder = CLIPTextModelWrapper.from_pretrained(
    '../../checkpoints/arc2face', subfolder="encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    '../../checkpoints/arc2face', subfolder="arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

# %%
import requests
from io import BytesIO

import cv2
from insightface.app import FaceAnalysis
import torch

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

import time
def compare_face_amazon(img1, img2):
    while True:
        try:
            import requests
            from PIL import Image
            from io import BytesIO

            headers = {"Authorization": "Bearer YOUR_EDENAI_TOKEN"}

            url = "https://api.edenai.run/v2/image/face_compare"
            data = {
                "providers": "amazon",
            }

            buf1 = BytesIO()
            buf2 = BytesIO()
            img1.save(buf1, format="PNG")
            img2.save(buf2, format="PNG")
            buf1.seek(0)
            buf2.seek(0)

            files = {
                "file1": ("image1.png", buf1, "image/png"),
                "file2": ("image2.png", buf2, "image/png"),
            }

            response = requests.post(url, data=data, files=files, headers=headers)
            result = response.json()
            print(result)
            # print(max(i["confidence"] for i in result["amazon"]["items"]))
            if len(result["amazon"]["items"]) == 0:
                return False
            else:
                return max(i["confidence"] for i in result["amazon"]["items"]) > 0.8
        except:
            print("Retrying...")
            time.sleep(1)

def compare_face_insight(img1, img2):
    try:
        image = np.array(img1.convert("RGB"))[:,:,::-1]
        faces = app.get(image)
        faceid_embeds_1 = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        image = np.array(img2.convert("RGB"))[:,:,::-1]
        faces = app.get(image)
        faceid_embeds_2 = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(faceid_embeds_1, faceid_embeds_2).item()
        return cos_sim
    except Exception as e:
        print("Error in compare_face_insight:", e)
        return None

def compare_faces(img1, img2, api_key, api_secret):
    buf1 = BytesIO()
    img1.save(buf1, format="JPEG")
    buf1.seek(0)

    buf2 = BytesIO()
    img2.save(buf2, format="JPEG")
    buf2.seek(0)

    files = {
        "image_file1": ("img1.jpg", buf1, "image/jpeg"),
        "image_file2": ("img2.jpg", buf2, "image/jpeg")
    }

    data = {
        "api_key": api_key,
        "api_secret": api_secret
    }

    r = requests.post(
        "https://api-us.faceplusplus.com/facepp/v3/compare",
        data=data,
        files=files
    )
    return r.json()

# %%
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')

# %%
# import onnxruntime
# app = FaceAnalysis(name='antelopev2', root='./', providers=onnxruntime.get_available_providers())
# app.prepare(ctx_id=0, det_size=(640, 640))
import pickle
with open("../../attacks/partialface/test.pkl", "rb") as f:
    id_embs = pickle.load(f)

# %%
fn = id_embs["filenames"][0]
fn = sum(id_embs["filenames"], [])
fn = [i.replace("./", "./") for i in fn] 
# %%

# %%
id_embs["student_embeddings"] = torch.cat(id_embs["student_embeddings"], dim=0)
templates = id_embs["templates"]

templates = torch.cat(templates, dim=0) 
# %%
img = np.array(Image.open('alex.png').convert("RGB"))[:,:,::-1]


# %% 
import wandb
wandb.init(project="arc2face_insight_eval")

# %%
levels = []
at_least_one_passed = []
sims = []
amazon_pass = []
import shutil
for idx in range(1000):
    id_emb = id_embs["student_embeddings"][idx][None,:].to(torch.float16)
    id_emb = (id_emb/torch.norm(id_emb, dim=1, keepdim=True)).to(torch.float16).cuda()
    id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

    num_images = 1 
    images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=5).images
    images[0].save(f"eval_arc2face/partial_{idx:04}_regen.png")
    shutil.copy(fn[idx], f"eval_arc2face/partial_{idx:04}_real.png")
    continue
    cache = []
    face_1_boxes = []
    face_2_boxes = []
    for i in range(5):
        res = compare_faces(images[i], Image.open(fn[idx]), "YOUR_FACEPP_API_KEY", "YOUR_FACEPP_API_SECRET")
        if len(res['faces2']) == 0:
            print("No face detected") 
            continue
        if len(res['faces1']) == 0:
            print("Failed to generate face")
            levels.append("Failed")
            cache.append("Failed")
            continue
        if res["confidence"] > res['thresholds']['1e-5']:
            level = "1e-5"
        elif res["confidence"] > res['thresholds']['1e-4']:
            level = "1e-4"
        elif res["confidence"] > res['thresholds']['1e-3']:
            level = "1e-3"
        else:
            level = "Failed"
        cache.append(level)
        levels.append(level)
        face_1_boxes.append(res['faces1'][0]['face_rectangle'])
        face_2_boxes.append(res['faces2'][0]['face_rectangle'])
        sim = compare_face_insight(images[i], Image.open(fn[idx]))
        if sim is not None:
            sims.append(sim)
        amazon_pass.append(compare_face_amazon(Image.open(fn[idx]), images[i]))

    success_rate = len([l for l in levels if l != "Failed"])/len(levels)
    if len(face_1_boxes) == 0 or len(face_2_boxes) == 0:
        wandb.log({
            "at_least_one_passed_rate": np.mean(at_least_one_passed), 
                    "per_image_success_rate": success_rate,
            })
        continue
    if any(l != "Failed" for l in cache):
        at_least_one_passed.append(1)
    else:
        at_least_one_passed.append(0)
    print("At least one passed score:" , np.mean(at_least_one_passed))

    print(level) 
    success_rate = len([l for l in levels if l != "Failed"])/len(levels)
    print(f"Per-image success rate: {success_rate}") 
    template = templates[idx].permute(1,2,0).numpy().mean(-1)
    template = (template - template.min()) / (template.max() - template.min() + 1e-8)
    print(template.shape)
    template = template
    template = Image.fromarray((template*255).astype(np.uint8)) 
    l, c = np.unique(levels, return_counts=True)
    c = [i/len(levels) for i in c] 
    print(dict(zip(l, c)))

    x, y, w, h = face_1_boxes[0]['left'], face_1_boxes[0]['top'], face_1_boxes[0]['width'], face_1_boxes[0]['height']
    generated_face = images[0]
    generated_face = generated_face.crop((x, y, x+w, y+h)).resize((160, 160))
    x, y, w, h = face_2_boxes[0]['left'], face_2_boxes[0]['top'], face_2_boxes[0]['width'], face_2_boxes[0]['height']
    real_face = Image.open(fn[idx])
    real_face = real_face.crop((x, y, x+w, y+h)).resize((160, 160))
    

    template = template.resize((160, 160))
    wandb.log({
        "at_least_one_passed_rate": np.mean(at_least_one_passed), 
                "per_image_success_rate": success_rate,
                "image": wandb.Image(hstack([real_face.resize((160, 160)), template.resize((160, 160)), generated_face.resize((160, 160))])),
                "sim": np.mean(sims) if len(sims) > 0 else 0,
                "sim_high_only": np.mean([s for s in sims if s > 0.3]) if len([s for s in sims if s > 0.3]) > 0 else 0,
                "amazon_pass": np.mean(amazon_pass)
        })

# %%
np.unique(levels, return_counts=True)
with open("res_partialface.pkl", "wb") as f:
    pickle.dump({ 
        "levels": levels,
        "at_least_one_passed": at_least_one_passed,
        "sims": sims
    }, f)

