'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Assignment 2
10/19/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import torch
import torch.optim as optim
from dataloader import load_images, load_illuminants_from_excel
from Network import Generator, Discriminator
from preprocessing import resize_image_to_multiple_of_32, get_image_patches, histogram_stretching
import numpy as np

# Initialize models and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to dataset
images_folder = 'C:/Users/kodur/OneDrive - Northeastern University/Desktop/GAN2/all_masked/AllMasked'
mat_file = 'C:/Users/kodur/OneDrive - Northeastern University/Desktop/true_illuminant_1.xlsx'

# Load data
images = load_images(images_folder)
illuminants = load_illuminants_from_excel(mat_file)

# Initialize models
G = Generator(3, 3).to(device)
D = Discriminator(3).to(device)

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = torch.nn.MSELoss()

#train loop 
for epoch in range(120):  # Train for 1 epoch (you can increase to 100 later)
    for img in images:
    # Preprocess: Ensure the image is valid
      if img is None:
        print("No image found, skipping...")
        continue  # Skip if the image is invalid

    print(f"Loaded image of shape {img.shape}")  # Confirm image shape

    # Convert the PyTorch tensor to a NumPy array before resizing
    img_numpy = img.permute(1, 2, 0).cpu().numpy()  # Convert from [C, H, W] to [H, W, C]

    # Resize the image to ensure both dimensions are multiples of 32
    resized_img = resize_image_to_multiple_of_32(img_numpy)

    # Extract 32x32 patches from the resized image
    patches = get_image_patches(resized_img)  # [num_patches, patch_height, patch_width, channels]

    # Convert patches to PyTorch tensors with the correct shape
    patches = torch.tensor(patches).permute(0, 3, 1, 2).float()  # [num_patches, C, H, W]

    # Debugging: Print the correct shape of patches
    print(f"Patches shape (before stacking): {patches.shape}")  # Should be [num_patches, 3, 32, 32]

    # Move patches to GPU
    patches = patches.to(device)

    # Generate fake patches using the generator
    fake_patches = G(patches)  # Now the shape should match

    # Train the discriminator
    real_loss = criterion(D(patches), torch.ones_like(D(patches)))  # Real patches
    fake_loss = criterion(D(fake_patches.detach()), torch.zeros_like(D(fake_patches)))  # Fake patches
    d_loss = (real_loss + fake_loss) / 2  # Average loss

    # Backpropagation and optimizer step for the discriminator
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train the generator
    g_loss = criterion(D(fake_patches), torch.ones_like(D(fake_patches)))  # Train to fool discriminator

    # Backpropagation and optimizer step for the generator
    optimizer_G.zero_grad()
    g_loss.backward()
    optimizer_G.step()

# Print loss values after the epoch
print(f"Epoch {epoch+1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save the trained generator model
print("Training complete. Saving the generator model...")
torch.save(G.state_dict(), 'generator.pth')
print("Model saved.")
