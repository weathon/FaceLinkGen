import torch
from torch.utils.data import Dataset
import numpy as np
import os

from methods.fracface_fixed.utils.dct_utils import dct_transform
from methods.fracface_fixed.utils.fractal_utils import (
    generate_fsm, apply_fractal_transform,
    generate_snake_indices, form_training_batch_with_fractal,
    create_square_subsets
)


class FractalDataset(Dataset):
    def __init__(self, index_file, dataset_root, transform=None, remap_labels=True, label_map=None):
        """
        Custom PyTorch Dataset class for loading preprocessed .npy data with optional label remapping.

        Args:
            index_file (str): Path to a text file where each line contains the relative image path and label.
            dataset_root (str): Root directory containing the .npy files.
            transform (callable, optional): Optional transform to be applied on the sample.
            remap_labels (bool): Whether to remap labels to a contiguous range starting from 0.
            label_map (dict, optional): An externally provided label mapping (used when remap_labels is False).
        """
        self.transform = transform
        self.dataset_root = dataset_root

        self.samples = []      # List of (relative_npy_path, label) pairs
        raw_labels = []        # Original label list before any remapping

        # Read index file and load paths and labels
        with open(index_file, 'r') as f:
            for line in f:
                if line.strip():
                    rel_path, label = line.strip().split()
                    npy_path = rel_path.replace('.jpg', '.npy')  # Ensure it corresponds to preprocessed .npy file
                    self.samples.append((npy_path, int(label)))
                    raw_labels.append(int(label))

        # Label remapping (to start from 0)
        if remap_labels:
            unique_labels = sorted(set(raw_labels))
            self.label_map = {old: new for new, old in enumerate(unique_labels)}
            self.samples = [(path, self.label_map[label]) for path, label in self.samples]
            print(f"[DEBUG] Labels remapped. Number of classes: {len(self.label_map)}")

        # Use external label_map (for validation set, etc.)
        elif label_map is not None:
            self.label_map = label_map
            try:
                self.samples = [(path, self.label_map[label]) for path, label in self.samples]
                print(f"[DEBUG] Using external label_map. Number of classes: {len(self.label_map)}")
            except KeyError as e:
                raise ValueError(f"Label {e} not found in provided label_map. Please check train/val consistency.")
        
        # No label remapping
        else:
            self.label_map = None
            print(f"[DEBUG] No label remapping. Max label value: {max(raw_labels)}")

        self.num_classes = len(self.label_map) if self.label_map else max(raw_labels) + 1

        print(f"[DEBUG] Total samples: {len(self.samples)} | Total classes: {self.num_classes}")
        print(f"[DEBUG] Label distribution: min={min(raw_labels)}, max={max(raw_labels)}, unique={len(set(raw_labels))}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads a single sample (input tensor and label) given an index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (input_tensor, label) where:
                - input_tensor (Tensor): Shape (81, 112, 112), normalized
                - label (Tensor): Long type tensor label
        """
        try:
            rel_path, label = self.samples[idx]
            npy_path = os.path.join(self.dataset_root, rel_path)

            # Load .npy file and convert to torch tensor
            input_tensor = torch.from_numpy(np.load(npy_path)).float()  # Shape: (81, 112, 112)

            # Normalize to zero mean and unit variance
            mean = input_tensor.mean()
            std = input_tensor.std()
            if std > 0:
                input_tensor = (input_tensor - mean) / std
            else:
                print(f"[WARNING] std=0 at idx={idx}, path={rel_path}")
                input_tensor = input_tensor - mean  # Avoid division by zero

            # Apply optional transform (e.g., data augmentation or additional normalization)
            if self.transform:
                input_tensor = self.transform(input_tensor)

            return input_tensor, torch.tensor(label).long()

        except Exception as e:
            print(f"[ERROR] Failed to load sample at idx={idx}: {e}")
            # Return a dummy tensor and default label on error
            dummy_input = torch.zeros((81, 112, 112))
            return dummy_input, torch.tensor(0).long()
