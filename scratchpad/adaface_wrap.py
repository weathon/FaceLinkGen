"""AdaFace IR-101 WebFace12M — the attack side (teacher AND student).

Different family from every protection method here (all ArcFace: PerceptFace = SimSwap
ArcFace R50, CanFG/CanFG-Ano = IR-SE50, TIP-IM = IR-SE50).

net.Backbone.output_layer is BatchNorm2d -> Dropout(0.4) -> Flatten -> Linear(512*7*7,512)
-> BatchNorm1d, so the sweep recipe's Dropout(0.4)-before-the-final-Linear is already
native here and must NOT be spliced in a second time. forward() returns (feature, norm)
with feature already L2-normalised.

Preprocessing is shared by teacher and student so the two are consistent: BGR uint8,
112x112, (x/255 - 0.5)/0.5, CHW. This matches AdaFace's own inference.to_input, which
converts a PIL RGB image to BGR before normalising.
"""
import sys
import cv2
import torch

ADAFACE = '/raid/wg25r/redteam_work/third_party/AdaFace'
CKPT = '/raid/wg25r/redteam_work/weights/adaface_ir101_webface12m.ckpt'
sys.path.insert(0, ADAFACE)
import net


def load_adaface(device):
    model = net.build_model('ir_101')
    sd = torch.load(CKPT, map_location='cpu')['state_dict']
    model.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('model.')})
    return model.to(device)


def read_112(path):
    """Disk PNG -> the tensor AdaFace expects. Raises on an unreadable file."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError('unreadable image: ' + path)
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(((img / 255.0) - 0.5) / 0.5).permute(2, 0, 1).float()
