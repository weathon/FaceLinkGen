import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from deepface import DeepFace
from torchjpeg import dct


device = "cuda" if torch.cuda.is_available() else "cpu"

_conversion_model = None

tf_conv = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
])


def highpass(img):
    return img - TF.gaussian_blur(img, (5, 5))


def normalize_batch(imgs, eps=1e-6):
    minv = imgs.amin(dim=(1, 2, 3), keepdim=True)
    maxv = imgs.amax(dim=(1, 2, 3), keepdim=True)
    return (imgs - minv) / (maxv - minv + eps)


def save_tensor_png(img, path):
    if img.dim() == 4:
        img = img[0]
    img = img.detach().cpu()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = img.permute(1, 2, 0).numpy()
    Image.fromarray((img * 255).astype("uint8")).save(path)


def ensure_path(path):
    if path not in sys.path:
        sys.path.append(path)


def get_conversion_model():
    global _conversion_model
    if _conversion_model is None:
        ensure_path("../../methods/minusface")
        from minusface import MinusBackbone

        model = MinusBackbone(mode="stage1")
        model.load_state_dict(
            torch.load("../../checkpoints/minusface_stage1.pth", map_location="cpu")
        )
        _conversion_model = model.eval().to(device)
    return _conversion_model


def dct_transform(x, chs_remove=None, chs_pad=False):
    assert x.shape[1] == 3
    size = 8
    stride = size
    pad = 0
    dilation = 1
    ratio = size

    x = x * 0.5 + 0.5
    x = F.interpolate(x, scale_factor=ratio, mode="bilinear", align_corners=True)
    x = x * 255
    x = dct.to_ycbcr(x)
    x = x - 128

    b, c, h, w = x.shape
    n_block = h // stride
    x = x.view(b * c, 1, h, w)
    x = torch.nn.functional.unfold(
        x,
        kernel_size=(size, size),
        dilation=dilation,
        padding=pad,
        stride=(stride, stride),
    )
    x = x.transpose(1, 2)
    x = x.view(b, c, -1, size, size)
    x_freq = dct.block_dct(x)
    x_freq = x_freq.view(b, c, n_block, n_block, size * size).permute(0, 1, 4, 2, 3)
    chs_remove = list(range(15))
    channels = list(set([i for i in range(x_freq.shape[2])]) - set(chs_remove))
    x_freq = x_freq[:, :, channels, :, :]

    x_freq = x_freq.reshape(b, -1, n_block, n_block)
    x_freq = x_freq.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return x_freq


def load_face(path):
    try:
        face = DeepFace.extract_faces(
            path, detector_backend="opencv", enforce_detection=False
        )[0]["face"]
        face = cv2.resize(face, (112, 112))
        img_pil = Image.fromarray((face * 255).astype("uint8"))
        img_tensor = torch.from_numpy(face).permute(2, 0, 1).float()
        return img_pil, img_tensor
    except Exception:
        img_pil = Image.open(path).convert("RGB").resize((112, 112))
        img_tensor = tf_conv(img_pil)
        return img_pil, img_tensor


def fracface_special(img_pil):
    ensure_path("../../methods/fracface")
    import data2npy

    c_img = data2npy.preprocess_and_return(img_pil, 1)[0]
    if isinstance(c_img, torch.Tensor):
        c_img = c_img.float()
    else:
        c_img = torch.from_numpy(c_img).float()
    c_img = c_img.reshape(-1, 112, 112).mean(0, keepdim=True).repeat(3, 1, 1)
    c_img = (c_img - c_img.min()) / (c_img.max() - c_img.min() + 1e-6)
    c_img = highpass(c_img)
    return c_img


def partialface_special(img_tensor):
    ensure_path("../partialface")
    import processing_utils as util

    c_img = util.form_training_batch(img_tensor.unsqueeze(0), [1])[0][0]
    c_img = c_img.reshape(-1, 112, 112).mean(0, keepdim=True).repeat(3, 1, 1)
    c_img = (c_img - c_img.min()) / (c_img.max() - c_img.min() + 1e-6)
    c_img = highpass(c_img)
    return c_img


def minusface_special(img_tensor):
    with torch.no_grad():
        out = get_conversion_model()(img_tensor.unsqueeze(0).to(device))[5]
    imgs = normalize_batch(out.float())
    imgs = dct_transform(imgs)
    imgs = normalize_batch(imgs)
    imgs = highpass(imgs)
    return imgs[0].cpu()


def pick_default_image():
    if os.path.exists("0001.png"):
        return "0001.png"
    fallback_dir = "../training-tpdne"
    if os.path.isdir(fallback_dir):
        files = sorted(os.listdir(fallback_dir))
        if files:
            return os.path.join(fallback_dir, files[0])
    raise FileNotFoundError("No image found: 0001.png or ../training-tpdne")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_path",
        nargs="?",
        default=pick_default_image(),
        help="Path to input image.",
    )
    parser.add_argument(
        "--method",
        choices=["fracface", "partialface", "minusface"],
        default="fracface",
        help="Which method to visualize.",
    )
    args = parser.parse_args()

    image_path = args.image_path
    img_pil, img_tensor = load_face(image_path)

    if args.method == "fracface":
        frac = fracface_special(img_pil)
        save_tensor_png(frac, "demo_fracface_highpass.png")
    elif args.method == "partialface":
        part = partialface_special(img_tensor)
        save_tensor_png(part, "demo_partialface_highpass.png")
    else:
        minus = minusface_special(img_tensor)
        save_tensor_png(minus, "demo_minusface_highpass.png")
