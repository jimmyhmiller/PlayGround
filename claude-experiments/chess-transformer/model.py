"""
Residual CNN for board encoding + MLP for scalar features.
Maia-style architecture for move-time prediction.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class MoveTimeModel(nn.Module):
    def __init__(self, in_channels=12, cnn_channels=64, n_blocks=6, n_scalars=6):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.n_blocks = n_blocks

        # CNN board encoder
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(cnn_channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Scalar feature encoder
        self.scalar_net = nn.Sequential(
            nn.Linear(n_scalars, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(cnn_channels + 128, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, board, scalars):
        x = self.input_conv(board)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)

        s = self.scalar_net(scalars)

        combined = torch.cat([x, s], dim=1)
        return self.head(combined).squeeze(-1)
