import torch
import torch.nn as nn

class FSMSelectAttention(nn.Module):
    def __init__(self, channels=81, reduction=9, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        w = self.fc(self.pool(x))  # Shape: (B, 81, 1, 1)
        x = x * w                  # Channel-wise attention
        x = self.dropout(x)       # Dropout after attention
        return x
