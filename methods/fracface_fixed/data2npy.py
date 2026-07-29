import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from utils.dct_utils import dct_transform
from utils.fractal_utils import generate_fsm, apply_fractal_transform, generate_snake_indices, form_training_batch_with_fractal, create_square_subsets


# Function to preprocess and save a single image or .npy file
def preprocess_and_save(image_path, label, save_path, image_size=112, transform=None):
    try: 
        # 1. Read image or .npy file
        if image_path.endswith('.npy') or image_path.endswith('.npz'):
            img_array = np.load(image_path)
            if img_array.ndim == 3 and img_array.shape[0] == 3:
                img_tensor = torch.from_numpy(img_array).float()
            elif img_array.ndim == 3 and img_array.shape[2] == 3:
                img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported npy shape: {img_array.shape}")   
            img_tensor = img_tensor / 255.0
        else:
            # img_s = DeepFace.extract_faces(image_path, detector_backend = "opencv", enforce_detection=False)[0]["face"] 
            # img = Image.fromarray((img_s * 255).astype("uint8"))
            # img = Image.open(image_path).convert('RGB')
            img = Image.open(image_path).convert('RGB')
            img = img.resize((image_size, image_size))
            img_array = np.asarray(img).copy()
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0  # [3,112,112]

        if transform:
            img_tensor = transform(img_tensor)

        part1, _ = create_square_subsets(img_tensor.unsqueeze(0))  # (1, 81, H, W)
        inputs, _ = form_training_batch_with_fractal(part1, label)  # (1, 81, 112, 112)

        assert not torch.isnan(inputs).any(), f"NaN in FSM output"
        assert not torch.isinf(inputs).any(), f"Inf in FSM output"
        assert inputs.shape[1] == 81, f"Expected 81 channels, got {inputs.shape[1]}"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, inputs.squeeze(0).cpu().numpy())  # Saved as shape (81,112,112)

    except Exception as e:
        # print(f"[ERROR] Failed to preprocess {image_path}: {e}")
        pass


import random
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def preprocess_and_return(pil_image, label, image_size=112, transform=None, fixed_channel=True):
    """Return one FracFace template, optionally fixing the random FSM with seed 42."""
    if fixed_channel:
        seed_everything()
    img = pil_image.convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.asarray(img).copy()
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0  # [3,112,112]

    if transform:
        img_tensor = transform(img_tensor)

    # 2. Apply create_square_subsets + form_training_batch_with_fractal
    part1, _ = create_square_subsets(img_tensor.unsqueeze(0))  # (1, 81, H, W)
    inputs, _ = form_training_batch_with_fractal(part1, 0)  # (1, 81, 112, 112)

    return inputs 



# Function to iterate and preprocess the entire dataset
def preprocess_dataset(index_file, image_root, save_root, image_size=112, transform=None):
    with open(index_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Preprocessing dataset"):
        rel_path, label = line.strip().split()
        label = int(label)
        img_path = os.path.join(image_root, rel_path)

        # Generate target .npy save path
        save_path = os.path.join(save_root, rel_path.replace('.jpg', '.npy'))

        # Skip if .npy already exists
        if not os.path.exists(save_path): 
            preprocess_and_save(img_path, label, save_path, image_size, transform)


# Function to count and process missing .npy files
def count_missing_and_generate_npy(image_root, index_file, save_root, image_size=112, transform=None):
    missing_count = 0
    with open(index_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing missing .npy files"):
        rel_path, label = line.strip().split()
        label = 0
        img_path = os.path.join(image_root, rel_path)
        save_path = os.path.join(save_root, rel_path.replace('.jpg', '.npy'))

        # Generate only if .npy does not exist
        if (not os.path.exists(save_path)) and (not os.path.exists(save_path + ".npz")):
            preprocess_and_save(img_path, label, save_path, image_size, transform)
            missing_count += 1

    return missing_count
# from multiprocessing import Pool
# import os
# from tqdm import tqdm

# def _worker(args):
#     image_root, save_root, image_size, transform, line = args
#     rel_path, _ = line.strip().split()
#     label = 0
#     img_path = os.path.join(image_root, rel_path)
#     save_path = os.path.join(save_root, rel_path.replace(".jpg", ".npy"))
#     if not os.path.exists(save_path):
#         preprocess_and_save(img_path, label, save_path, image_size, transform)
#         return 1
#     return 0

# def count_missing_and_generate_npy(image_root, index_file, save_root, image_size=112, transform=None):
#     with open(index_file, "r") as f:
#         lines = f.readlines()

#     args = [(image_root, save_root, image_size, transform, line) for line in lines]

#     missing_count = 0
#     with Pool(processes=8) as pool:
#         for r in tqdm(pool.imap_unordered(_worker, args), total=len(args), desc="Processing missing .npy files"):
#             missing_count += r

#     return missing_count

if __name__ == "__main__":
    # Define paths
    index_file = "../../data_splits/index.txt"
    # index_file = "../full_index.txt"
    # index_file = "../tpdne_index.txt"
    image_root = "/path/to/casia-webface"
    # image_root = "/path/to/FaceLinkGen/fracface/training-tpdne"
    # image_root = "/path/to/celeba_aligned"
    # save_root = "/path/to/fracface_templates_fracface_tpdne" 
    save_root = "/path/to/fracface_templates" 

    # Count and generate missing .npy files
    missing_count = count_missing_and_generate_npy(image_root, index_file, save_root, image_size=112, transform=None)

    # Output the number of files generated
    print(f"Total missing .npy files generated: {missing_count}")
