import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcMarginProduct(nn.Module):
    """
    Implement of the ArcFace (Additive Angular Margin Loss)
    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(self, in_features, out_features, s=16.0, m=0.30, easy_margin=False):
        """
        Args:
            in_features (int): Size of each input sample (embedding dimension).
            out_features (int): Number of classes.
            s (float): Norm scaling factor (usually 32~128).
            m (float): Additive angular margin (usually 0.3~0.8).
            easy_margin (bool): Whether to use the easy margin strategy.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weight matrix (class centers), initialized with Xavier uniform distribution
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin

        # Precompute constants for margin computation
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.eps = 1e-5  # small value to prevent sqrt(0)

    def forward(self, input, label):
        """
        Args:
            input (Tensor): Feature matrix with shape (batch_size, in_features).
            label (Tensor): Ground truth labels with shape (batch_size,).

        Returns:
            Tensor: Scaled logits with angular margin.
        """

        # Check for NaNs or Infs in the input
        if torch.isnan(input).any() or torch.isinf(input).any():
            print("Found NaN or Inf in input")
            print(f"Input: {input}")
            raise ValueError("Input contains NaN or Inf")

        if torch.isnan(label).any() or torch.isinf(label).any():
            print("Found NaN or Inf in label")
            print(f"Label: {label}")
            raise ValueError("Label contains NaN or Inf")

        # Normalize the input features and weights, compute cosine similarity
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # Compute sine from cosine using trigonometric identity
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=self.eps))

        # Compute cos(θ + m) using cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply margin strategy
        if self.easy_margin:
            # Use phi only when cosine > 0; otherwise fallback to cosine
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Use phi when cosine > threshold; otherwise use cosine - margin offset
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # Combine phi for target class and cosine for non-target classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Apply scaling
        output *= self.s

        # Final numerical check
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Found NaN or Inf in output")
            print(f"Cosine: {cosine}")
            print(f"Sine: {sine}")
            print(f"Phi: {phi}")
            print(f"Output: {output}")
            raise ValueError("ArcMargin output contains NaN or Inf")

        return output
