'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Assignment 2
10/19/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    """A ResNet block with two convolutional layers."""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.conv_block(x)  # Residual connection

class Generator(nn.Module):
    """Generator that processes 32x32 patches."""
    def __init__(self, input_nc, output_nc, n_blocks=6):
        super(Generator, self).__init__()
        
        # Initial convolution layer to process patches
        layers = [
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=3),  # Keep size 32x32
            nn.ReLU(inplace=True)
        ]
        
        # Add ResNet blocks for deeper feature extraction
        for _ in range(n_blocks):
            layers.append(ResnetBlock(64))

        # Final convolution layer to generate output
        layers.append(nn.Conv2d(64, output_nc, kernel_size=7, padding=3))  # Output shape remains 32x32
        
        # Build the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass for a batch of patches."""
        return self.model(x)

class Discriminator(nn.Module):
    """Discriminator that classifies 32x32 patches as real or fake."""
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        
        # Series of convolutional layers to downsample the patch
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),  # 4x4 -> 1x1
            nn.Sigmoid()  # Output probability that the patch is real or fake
        )

    def forward(self, x):
        """Forward pass for a batch of patches."""
        return self.model(x)
