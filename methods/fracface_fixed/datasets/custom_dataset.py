import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, index_file, transform=None):
        """
        Args:
            index_file (str): Path to the index file containing image paths and corresponding labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.index_file = index_file
        self.transform = transform
        self.data = self.load_data(index_file)  # Load data entries (image path and label)

        # Parse image paths and labels from the loaded data
        self.imgs = [item[0] for item in self.data]  # List of image file paths
        self.labels = [item[1] for item in self.data]  # Corresponding labels for the images

    def load_data(self, index_file):
        """
        Reads the index file and parses it into a list of (image_path, label) pairs.

        Args:
            index_file (str): Path to the index file.

        Returns:
            list of tuples: A list where each item is a tuple (image_path, label).
        """
        with open(index_file, 'r') as f:
            data = f.readlines()

        # Remove whitespace and split each line into image path and label
        data = [line.strip().split() for line in data]
        print(f"Loaded data from {index_file}: {data[:5]}")  # Print the first 5 entries for verification

        return data

    def load_image(self, image_path):
        """
        Loads an image from a given path and converts it to RGB format.

        Args:
            image_path (str): Path to the image file.

        Returns:
            PIL.Image.Image: The loaded image in RGB format.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")

        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure the image has 3 channels (RGB)

        return img

    def __getitem__(self, index):
        """
        Retrieves the image and label at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: (transformed image, label)
        """
        image_path = self.imgs[index]
        label = int(self.labels[index])  # Convert label to integer if necessary

        img = self.load_image(image_path)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.data)
