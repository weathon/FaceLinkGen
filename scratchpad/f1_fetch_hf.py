"""f1: fetch TPDNE zip, Arc2Face weights, SD1.5 base. Resumable via hf_hub cache."""
import os
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download

DATA = "/raid/wg25r/fracface_rerun/data"
CKPT = "/home/wg25r/face_deid/PerceptFace/upstream/FaceLinkGen/checkpoints/arc2face"
os.makedirs(DATA, exist_ok=True)

# --- TPDNE: a single zip file in the repo, not HF shards -> fetch as a file
zpath = hf_hub_download(
    repo_id="TLeonidas/this-person-does-not-exist",
    filename="thisPersonDoesNotExist.zip",
    repo_type="dataset",
)
print("tpdne zip:", zpath, os.path.getsize(zpath))

tpdne_dir = os.path.join(DATA, "tpdne")
if not os.path.exists(tpdne_dir):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print("tpdne zip entries:", len(names), names[:5])
        z.extractall(tpdne_dir + ".part")
    os.rename(tpdne_dir + ".part", tpdne_dir)
print("tpdne dir:", tpdne_dir, len(os.listdir(tpdne_dir)))

# --- Arc2Face: encoder + unet subfolders (+ its own arcface.onnx)
for f in [
    "arc2face/config.json",
    "arc2face/diffusion_pytorch_model.safetensors",
    "encoder/config.json",
    "encoder/pytorch_model.bin",
    "arcface.onnx",
]:
    p = hf_hub_download(repo_id="FoivosPar/Arc2Face", filename=f, local_dir=CKPT)
    print("arc2face:", p)

# --- SD1.5 base, pulled by name at pipeline build time; prefetch the snapshot now
p = snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    allow_patterns=["*.json", "*.txt", "vae/*.safetensors", "text_encoder/*.safetensors",
                    "unet/*.safetensors", "tokenizer/*"],
)
print("sd15 snapshot:", p)
print("DONE")
