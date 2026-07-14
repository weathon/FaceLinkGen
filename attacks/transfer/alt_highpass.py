"""
Compare alternative highpass operators visually and via PSD:
- Gaussian highpass (current baseline)
- Sobel edges
- Laplacian
- FFT-based bandpass (own implementation, NOT DCT)
- Self-DCT (own naive block DCT, not from minusface/fracface)
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
sys.path.insert(0, "../../methods/minusface")
from minusface import MinusBackbone
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
tf_conv = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

mface = MinusBackbone(mode='stage1')
mface.load_state_dict(torch.load("../../checkpoints/minusface_stage1.pth", map_location='cpu'))
mface = mface.eval().to(device)


def to_grayscale_minmax(t):
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True)
    mx = t.amax(dim=(1, 2, 3), keepdim=True)
    t = (t - mn) / (mx - mn + 1e-6)
    return (t - 0.5) / 0.5


def gauss_hp(img, k=5):
    blurred = TF.gaussian_blur(img, (k, k))
    return img - blurred


def sobel_mag(img):
    """L2 mag of Sobel x/y. img: (B, 3, H, W)."""
    sx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=img.device).expand(3, 1, 3, 3)
    sy = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=img.device).expand(3, 1, 3, 3)
    pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
    gx = F.conv2d(pad, sx, groups=3)
    gy = F.conv2d(pad, sy, groups=3)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def laplacian(img):
    kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float32, device=img.device).expand(3, 1, 3, 3)
    pad = F.pad(img, [1, 1, 1, 1], mode='reflect')
    return F.conv2d(pad, kernel, groups=3)


def fft_highpass(img, cutoff=8):
    """Zero out low-frequency disc within `cutoff` pixels from center, then iFFT.
    No DCT, just standard FFT.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    F_img = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device) - cy,
                             torch.arange(W, device=img.device) - cx, indexing='ij')
    r = torch.sqrt((yy ** 2 + xx ** 2).float())
    mask = (r > cutoff).float().unsqueeze(0).unsqueeze(0)
    F_img = F_img * mask
    F_img = torch.fft.ifftshift(F_img, dim=(-2, -1))
    out = torch.fft.ifft2(F_img).real
    return out


def own_dct_hp(img, block=8, keep_high_from=2):
    """Own implementation of block DCT highpass.
    1. Block-DCT each 8x8 patch
    2. Zero out the top-left `keep_high_from x keep_high_from` low-frequency coeffs
    3. iDCT.
    Naive implementation using FFT-based DCT, not torchjpeg.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    assert H % block == 0 and W % block == 0
    # Block reshape
    x = img.view(B, C, H // block, block, W // block, block).permute(0, 1, 2, 4, 3, 5).contiguous()
    # x is now (B, C, nh, nw, block, block)
    nh, nw = H // block, W // block

    # 2D DCT via separable 1D DCT (use FFT trick)
    def dct_1d(x, dim):
        # naive DCT-II using FFT
        N = x.shape[dim]
        v = torch.cat([x, x.flip(dim)], dim=dim)
        V = torch.fft.fft(v, dim=dim)
        k = torch.arange(N, dtype=torch.float32, device=x.device)
        if dim == -1:
            phase = torch.exp(-1j * torch.pi * k / (2 * N))
            return (V[..., :N] * phase).real
        else:
            phase = torch.exp(-1j * torch.pi * k / (2 * N))
            shape = [1] * x.dim()
            shape[dim] = -1
            return (V.narrow(dim, 0, N) * phase.view(shape)).real

    # x: (B, C, nh, nw, block, block)
    x = dct_1d(x, -1)
    x = dct_1d(x, -2)
    # Now x is in DCT domain per block

    # Zero out low frequencies (top-left keep_high_from x keep_high_from)
    x[..., :keep_high_from, :keep_high_from] = 0

    # iDCT
    def idct_1d(X, dim):
        N = X.shape[dim]
        # iDCT-II
        k = torch.arange(N, dtype=torch.float32, device=X.device)
        if dim == -1:
            phase = torch.exp(1j * torch.pi * k / (2 * N))
            V = torch.zeros(*X.shape[:-1], 2 * N, dtype=torch.complex64, device=X.device)
            V[..., :N] = X * phase
            v = torch.fft.ifft(V, dim=-1) * 2 * N
            return v[..., :N].real
        else:
            phase = torch.exp(1j * torch.pi * k / (2 * N))
            shape = [1] * X.dim()
            shape[dim] = -1
            V = torch.zeros(*X.shape[:dim], 2 * N, *X.shape[dim+1:], dtype=torch.complex64, device=X.device) if dim != -2 else \
                torch.zeros(*X.shape[:-2], 2 * N, X.shape[-1], dtype=torch.complex64, device=X.device)
            V.narrow(dim, 0, N).copy_(X * phase.view(shape))
            v = torch.fft.ifft(V, dim=dim) * 2 * N
            return v.narrow(dim, 0, N).real

    # Use FFT iDCT via cosine basis (simpler: just iFFT of DCT result is approximate)
    # Approximation: zero low and run inverse with same dct_1d twice (since DCT is approximately involutive in inverse with normalization)
    # Use scipy-style: easier to just use the torch idct via fft2.real approach below
    # For simplicity, let's apply dct_1d again (DCT-II is approximate inverse of DCT-III)
    # Better: use torch.fft.irfft trick. Easiest: zero out and reverse the unfold using a precomputed iDCT matrix.

    # Just use a precomputed iDCT matrix
    M = make_idct_matrix(block, device=img.device)  # (block, block) - iDCT-II rows
    # Apply: y_kl = sum_ij M[k,i] M[l,j] X_ij
    # via matmul along last 2 dims
    x = torch.einsum('bcnmij,ki,lj->bcnmkl', x, M, M)
    # x is now back in spatial domain (per block)

    # Reshape back to image
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return x


def make_idct_matrix(N, device):
    """Return iDCT-II matrix: spatial = M @ dct(spatial) along that dim.
    Actually we want X_spat = sum_k M[i,k] X_dct[k]
    """
    n = torch.arange(N, dtype=torch.float32, device=device)
    k = torch.arange(N, dtype=torch.float32, device=device)
    # idct-II: x[n] = (1/N) X[0] + (2/N) sum_{k=1}^{N-1} X[k] cos(pi (2n+1) k / (2N))
    M = torch.cos(torch.pi * (2 * n.unsqueeze(1) + 1) * k.unsqueeze(0) / (2 * N))
    # scale
    M[:, 0] *= 1 / N
    M[:, 1:] *= 2 / N
    return M


# Run experiments
root = '/path/to/casia-webface'
paths = []
for d in sorted(os.listdir(root))[:16]:
    if os.path.isdir(os.path.join(root, d)):
        for f in sorted(os.listdir(os.path.join(root, d)))[:1]:
            paths.append(os.path.join(root, d, f))
            break

imgs = torch.stack([tf_conv(Image.open(p).convert("RGB").resize((112, 112))) for p in paths])
imgs = imgs.to(device) * 2 - 1  # to [-1, 1]
print(f"imgs: {imgs.shape}, [{imgs.min():.2f}, {imgs.max():.2f}]")

# Reference: minusface + hp(k=21) (more visible than k=5)
with torch.no_grad():
    minus_out = mface(imgs)[5]
minus_norm = to_grayscale_minmax(minus_out)
minus_hp = gauss_hp(minus_norm, k=5)

# Candidates -- all operate on raw face
imgs_norm = to_grayscale_minmax(imgs)  # to (B, 3, 112, 112) [-1, 1]
cands = {
    "minus_hp_k5_ref": minus_hp,
    "gauss_hp_k5": gauss_hp(imgs_norm, k=5),
    "gauss_hp_k21": gauss_hp(imgs_norm, k=21),
    "sobel": sobel_mag(imgs_norm),
    "laplacian": laplacian(imgs_norm),
    "fft_hp_c8": fft_highpass(imgs_norm, cutoff=8),
    "fft_hp_c16": fft_highpass(imgs_norm, cutoff=16),
    "fft_hp_c4": fft_highpass(imgs_norm, cutoff=4),
}

# Save samples (normalized to [0, 1] for display)
def to_disp(t):
    t = t.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    mn = t.amin(dim=(1, 2, 3), keepdim=True)
    mx = t.amax(dim=(1, 2, 3), keepdim=True)
    return (t - mn) / (mx - mn + 1e-6)


for name, t in cands.items():
    save_image(to_disp(t[:8]), f"alt_{name}.png", nrow=4)


# Radial PSD
def radial_psd(t):
    gray = t.mean(dim=1)
    F_img = torch.fft.fftshift(torch.fft.fft2(gray.double()), dim=(-2, -1))
    psd = (F_img.abs() ** 2)
    H, W = gray.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing='ij')
    r = torch.sqrt((yy.float() ** 2 + xx.float() ** 2)).int()
    rmax = r.max().item()
    res = []
    for p in psd:
        w = p.flatten().cpu().double().numpy()
        i = r.flatten().cpu().numpy()
        s = np.bincount(i, weights=w, minlength=rmax + 1)
        c = np.bincount(i, minlength=rmax + 1)
        res.append(s / np.maximum(c, 1))
    return np.stack(res).mean(axis=0)


plt.figure(figsize=(10, 6))
for name, t in cands.items():
    psd = radial_psd(t)
    plt.plot(np.log10(psd + 1e-12), label=name)
plt.xlabel("radial freq")
plt.ylabel("log10 power")
plt.title("Alternative highpass operators PSD")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("alt_psd.png", dpi=100)
print("Saved alt_*.png and alt_psd.png")

# Print L2 distance from minus_hp_k5 reference PSD
ref_psd = radial_psd(cands["minus_hp_k5_ref"])
print("\n=== Distance from minus_hp_k5_ref PSD ===")
for name, t in cands.items():
    if name == "minus_hp_k5_ref":
        continue
    p = radial_psd(t)
    # normalize both to peak=1 for shape comparison
    rp = ref_psd / (ref_psd.max() + 1e-12)
    cp = p / (p.max() + 1e-12)
    dist = np.linalg.norm(np.log10(rp + 1e-12) - np.log10(cp + 1e-12))
    print(f"{name:20s}  l2_log_psd={dist:.3f}  std={t.std().item():.4f}")
