from pathlib import Path
import torch
from mtcnn import MTCNN
import cv2
import numpy as np


import PIL.Image as Image
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from torchvision import transforms as trans
import os
# import libnvjpeg
# import pickle
#todo 按照原始顺序存储

# img_root_dir = r'D:\BaiduNetdiskDownload\vggface2\vggface2_test\vggface2_test\test'
# save_path = r'D:\BaiduNetdiskDownload\vggface2\vggface2_test\vggface2_test\test1'
# img_root_dir = '/home/user/fastdata/marshall/img_align_celeba/img_align_celeba/'
save_path = '/path/to/fairface'



# embed_path = '/home/taotao/Downloads/celeb-aligned-256/embed.pkl'

device = torch.device('cuda:0')
# device = torch.device('cpu')
mtcnn = MTCNN()

model = Backbone(50, 0.6, 'ir_se').to(device)
model.eval()
model.load_state_dict(torch.load('../../checkpoints/canfg/model_ir_se50.pth'))

# threshold = 1.54
test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# decoder = libnvjpeg.py_NVJpegDecoder()

embed_map = {}
# selected = os.listdir("/path/to/casia-webface/insight_embeddings")
# selected = set([x.split('.')[0] for x in selected])
# print(selected)


from datasets import load_dataset

ds = load_dataset("HuggingFaceM4/FairFace", "1.25")["validation"]

import tqdm
for i, sample in enumerate(tqdm.tqdm(ds)):
    try:
       
        # print(p)
        new_path = f"image_{i:06d}.jpg"
        # if Path(new_path).stem not in selected:
        #     print(f"skip {name}")
        #     continue
        img = np.array(sample['image'])[:,:,::-1]
        # if img.shape[0]>256 and img.shape[1]
        faces = mtcnn.align_multi(Image.fromarray(img[:, :, ::-1]), min_face_size=64, crop_size=(128, 128))
        if len(faces) == 0:
            continue
        for face in faces: 
            # scaled_img = face.resize((112, 112), Image.ANTIALIAS)
            # with torch.no_grad():
            #     embed = model(test_transform(scaled_img).unsqueeze(0).cuda()).squeeze().cpu().numpy()
            print(new_path)
            face.save(os.path.join(save_path, new_path.replace(".png", ".jpg")))
        # embed_map[new_path] = embed.detach().cpu()
    except Exception as e:
        print(e)
        continue

# with open(embed_path, 'wb') as f:
#     pickle.dump(embed_map, f)
#
# img = cv2.imread('/home/taotao/Pictures/47d947b4d9cf3e2f62c0c8023a1c0dea.jpg')[:,:,::-1]
# # bboxes, faces = mtcnn.align_multi(Image.fromarray(img), limit=10, min_face_size=30)
# bboxes, faces = mtcnn.align(Image.fromarray(img))
# input = test_transform(faces[0]).unsqueeze(0)
# embed = model(input.cuda())
# print(embed.shape)
# print(bboxes)
# face = np.array(faces[0])[:,:,::-1]
# cv2.imshow('', face)
# cv2.waitKey(0)
