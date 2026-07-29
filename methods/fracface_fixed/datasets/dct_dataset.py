import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from methods.fracface_fixed.utils.dct_utils import dct_transform
from methods.fracface_fixed.utils.fractal_utils import generate_snake_indices

def create_square_subsets(x, chs_prune_per_layer=None):
    """
    Create square subsets of frequency domain features using DCT and snake-like indexing.

    Args:
        x (Tensor): Input feature tensor of shape [B, C, H, W].
        chs_prune_per_layer (list of list of int, optional): 
            Lists of frequency channels to be removed at each layer.

    Returns:
        Tensor: A subset of the frequency-transformed input of shape [B, 81, H, W].
    """
    if chs_prune_per_layer is None:
        chs_prune_per_layer = [
            [0, 1, 2, 3, 8, 9, 10, 16, 17, 24],
            [64, 65, 66, 67, 71, 72, 73, 80, 81, 88],
            [128, 129, 130, 131, 136, 137, 138, 144, 145, 152]
        ]

    # Flatten all pruning indices into a single list
    all_pruned = [ch for layer in chs_prune_per_layer for ch in layer]

    # Apply DCT transform and remove the specified frequency channels
    x_freq = dct_transform(x, chs_remove=all_pruned, chs_pad=False)  # Output shape: [B, 162, H, W]

    # Get snake-sorted indices and select the first part (81 channels)
    part1_indices, part2_indices = generate_snake_indices()
    part1 = x_freq[:, part1_indices]  # Shape: [B, 81, H, W]

    return part1  # Return only part1 as the frequency-domain subset

class DCTDataset(Dataset):
    def __init__(self, img_dir, index_file, img_size=112):
        """
        A PyTorch Dataset class that loads images, applies DCT transform,
        and returns frequency-domain features.

        Args:
            img_dir (str): Root directory containing all image files.
            index_file (str): Path to index file, with each line in "rel_path label" format.
            img_size (int): Final image size (height and width after resizing).
        """
        self.img_dir = img_dir

        # Read the index file and parse it into (relative path, label) pairs
        with open(index_file, 'r') as f:
            self.samples = [line.strip().split() for line in f.readlines()]

        # Extract labels and compute number of unique classes
        self.labels = [int(sample[1]) for sample in self.samples]
        self.num_classes = len(set(self.labels))

        self.img_size = img_size

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve and preprocess a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (Tensor of shape [81, H, W], int label)
        """
        # Extract image filename and label from the sample list
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image and convert to RGB format
        image = Image.open(img_path).convert('RGB')

        # Convert image to tensor (shape: [3, H, W], range: [0.0, 1.0])
        image = transforms.ToTensor()(image)

        # Ensure the input image has 3 channels
        if image.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, but got {image.shape[0]} channels.")

        # Add batch dimension: shape becomes [1, 3, H, W]
        image = image.unsqueeze(0)

        # Apply DCT transform without channel pruning
        image = dct_transform(image, chs_remove=None, chs_pad=False)

        # Get 81 selected frequency channels using snake-like indexing
        part1_indices, _ = generate_snake_indices()
        image = image[:, part1_indices]  # Output shape: [1, 81, H, W]

        # Resize the feature map to the target image size
        image = transforms.Resize((self.img_size, self.img_size))(image)

        # Normalize the frequency features: zero mean, unit variance
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        else:
            print(f"[WARNING] std=0 at idx={idx}, path={img_path}")
            image = image - mean  # Avoid division by zero

        # Remove batch dimension and return image and label
        return image.squeeze(0), int(label)
