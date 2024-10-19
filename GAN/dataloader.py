'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Assignment 2
10/19/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import os
import scipy.io as sio
from torchvision import transforms
from PIL import Image
import pandas as pd

# Define a transformation to resize and convert images to tensors
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

def load_images(folder):
    """Load .tiff images from the specified folder."""
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(transform(img))
    return images

def load_illuminants_from_excel(file_path):
    """Load illuminant values from an Excel file."""
    df = pd.read_excel(file_path, header=None)  # Read without headers
    return df.values  # Convert to NumPy array
