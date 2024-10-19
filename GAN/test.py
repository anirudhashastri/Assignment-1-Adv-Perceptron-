'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Assignment 2
10/19/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import torch
from Network import Generator
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# Set the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the saved model and input image
model_path = 'generator.pth'  # Adjust if saved elsewhere
input_image_path = 'C:/Users/kodur/OneDrive - Northeastern University/Desktop/GAN2/test_image_cnn_unmasked.tiff'
output_image_path = 'output_image.png'  # Output path for generated image

# Initialize the generator model and load weights
G = Generator(3, 3).to(device)
G.load_state_dict(torch.load(model_path))
G.eval()  # Set the model to evaluation mode

def generate_white_balanced_image(input_path, output_path):
    """Generate a white-balanced image from an input image, preserving the original size."""
    # Open the input image and get its original size
    img = Image.open(input_path).convert('RGB')
    original_size = img.size  # (width, height)

    print(f"Original image size: {original_size}")

    # Transform the input image to a tensor (without resizing)
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Generate the white-balanced image
    with torch.no_grad():
        output_tensor = G(img_tensor)  # Output tensor

    # Remove batch dimension and move to CPU
    output_tensor = output_tensor.squeeze(0).cpu()

    # Convert the output tensor to a PIL image
    output_img = transforms.ToPILImage()(output_tensor)

    # Resize the output image to the original size
    output_img = output_img.resize(original_size, Image.BILINEAR)

    # Save the resized output image
    output_img.save(output_path)
    print(f"White-balanced image saved at: {output_path}")

def show_images(input_path, output_path):
    """Display the input and generated images side by side."""
    input_img = Image.open(input_path).convert('RGB')
    output_img = Image.open(output_path).convert('RGB')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(output_img)
    axes[1].set_title('White-Balanced Image')
    axes[1].axis('off')

    plt.show()

# Generate and display the white-balanced image
generate_white_balanced_image(input_image_path, output_image_path)
show_images(input_image_path, output_image_path)
