# %%
import pickle
with open("embeddings_another.pkl", "rb") as f:
    embeddings_another = pickle.load(f)
    
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f) 

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
student_embeddings = embeddings["student"]
teacher_embeddings = embeddings["teacher"]
filenames = embeddings["filenames"]
student_images = embeddings["student_images"]

# %%
embeddings.keys()

# %%
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image

base_model_path = "SG161222/RealVisXL_V3.0"
ip_ckpt = "ip-adapter-faceid_sdxl.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
)

# load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

# %%
torch.nn.functional.cosine_similarity(student_embeddings, teacher_embeddings[:,0], dim=1).argsort()

# %%
# from insightface.app import FaceAnalysis
# app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
import json
import requests
from io import BytesIO


def crop_face(img, ret_json): 
    image = np.array(img)
    box = ret_json['faces2'][0]['face_rectangle']
    x, y, w, h = box['left'], box['top'], box['width'], box['height']
    cropped = image[y:y+h, x:x+w]
    return Image.fromarray(cropped).resize((512, 512))

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

for idx in range(512):
    faceid = student_embeddings[idx:idx+1] / student_embeddings[idx:idx+1].norm()

    # %%
    faceid.shape

    # %%
    import cv2
    import random
    # generate image
    prompt = "photo of a person in white clothes in a kitchen facing the camera"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, face wear, overexposed"


    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid, num_samples=1, width=1024, height=1024, num_inference_steps=50, seed=random.randint(0, 1000000), scale=0.8, guidance_scale=3.5
    )
    from PIL import ImageDraw, ImageFont
    import PIL

    original_face = Image.open("val/"+filenames[idx]).resize((512, 512))
    protected_face = Image.fromarray((student_images[idx].permute(1,2,0).numpy()*128 + 128).astype('uint8')).resize((512, 512))
    generated_image = images[0].resize((1024, 1024))

    # %%

    # %%
    with open(f"visualize/res_{idx}.json", "w") as f:
        res = compare_faces(original_face, generated_image, "5W5RrSIa4Vk_AC7uLfXGjL2Vu8WGf9Qo", "X19PnBAW-reeXBt5I57uCxuH5uoQFMEi")
        json.dump(res, f, indent=4)

    # %%
    hstack([original_face, protected_face, crop_face(generated_image, res)]).save(f"visualize/{idx}_gen.png")

    # %%


    # %%



