'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Assignment 2
10/19/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''


import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import scipy.io as sio
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.optim as optim

class ColorConstancyCNN(nn.Module):
    def __init__(self, filter_size='3x3', activation_function='ReLU', dropout_rate=0.5):
        super(ColorConstancyCNN, self).__init__()

        # Define filters for the convolutional layer (only 1x1 and 3x3 filters allowed now)
        if filter_size == '1x1':
            self.filter_size = '1x1'
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=240, kernel_size=1, stride=1)
        elif filter_size == '3x3':
            self.filter_size = '3x3'
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=240, kernel_size=3, stride=1)
        else:
            raise ValueError("Unsupported filter size. Use '1x1' or '3x3'.")

        # Add dropout 
        self.dropout = nn.Dropout(dropout_rate)

        # Max pooling after the convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)

        # Fully connected layers after the convolutional layers
        # Now assuming the feature map size is 4x4 after pooling
        # 240 channels * 4 * 4 = 3,840 features
        if filter_size == '3x3':
            self.fc1 = nn.Linear(240 * 3 * 3, 40)
        else:
            self.fc1 = nn.Linear(240 * 4 * 4, 40)  # Fully connected layer with 40 nodes
        
        self.fc2 = nn.Linear(40, 3)  # Output layer with 3 nodes (RGB illuminant prediction)

        # Activation function setup (used in fully connected layers)
        if activation_function == 'PReLU':
            self.activation_fc = nn.PReLU()  # Initialize PReLU as a learnable layer
        elif activation_function == 'LeakyReLU':
            self.activation_fc = nn.LeakyReLU(0.01)  # Leaky ReLU
        elif activation_function == 'ReLU':
            self.activation_function = 'ReLU'  # Store ReLU to apply using torch.relu
        else:
            raise ValueError("Unsupported activation function. Use 'PReLU', 'LeakyReLU', or 'ReLU'.")

        # Store the activation function string for non-learnable activations (ReLU/LeakyReLU)
        self.activation_function = activation_function

    def forward(self, x):
        # Apply the selected convolutional layer
        x = self.conv1(x)

        # Apply max pooling
        x = self.pool(x)


        # Flatten the feature map to feed into the fully connected layers
        if self.filter_size == '1x1':
            x = x.reshape(-1, 240 * 4 * 4)
        else:
            x = x.reshape(-1, 240 * 3 * 3)
        

        
        # Fully connected layer with specified activation function
        x = self.fc1(x)

        if self.activation_function == 'ReLU':
            x = torch.relu(x)  # Apply ReLU activation
        elif self.activation_function == 'LeakyReLU':
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)  # Apply Leaky ReLU
        elif self.activation_function == 'PReLU':
            x = self.activation_fc(x)  # Apply PReLU (learnable)
        else:
            raise ValueError("Unsupported activation function. Use 'PReLU', 'LeakyReLU', or 'ReLU'.")

        # Apply dropout
        x = self.dropout(x)

        # Output layer (no activation function on the final output)
        x = self.fc2(x)  # Output layer (RGB illuminant prediction)

        return x
