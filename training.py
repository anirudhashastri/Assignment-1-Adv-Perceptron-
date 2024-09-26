import os
import torch
import torch.nn as nn
#from torch.utils.data import DataLoader
import scipy.io as sio
from ColorConstancyDataset import ColorConstancyDataset
from torch.utils.data import Dataset
from NetworkNew import ColorConstancyCNN
from NetworkNew import train_model_kfold
from preprocessing import load_image, resize_image_to_multiple_of_32, get_image_patches, histogram_stretching
from torch.utils.data import Dataset

    
# Define the paths to the dataset and ground truth illuminants
canon_1d_path = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\Canon1D'  # Folder containing your dataset of images
canon_5d_path = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\Canon5D'  # Folder containing your dataset of images
output_folder = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\ProcessedDataset'  # Folder where processed patches will be saved
groundtruth_path = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\real_illum_568.mat'

canon_1d_images = sorted([os.path.join(canon_1d_path, f) for f in os.listdir(canon_1d_path) if f.endswith('.png')])
canon_5d_images = sorted([os.path.join(canon_5d_path, f) for f in os.listdir(canon_5d_path) if f.endswith('.png')])
image_paths = canon_1d_images + canon_5d_images  # Combined image paths in order

# Load ground truth illuminants (already in order for Canon 1D followed by Canon 5D)
groundtruth_data = sio.loadmat(groundtruth_path)
groundtruth_illuminants = groundtruth_data['real_rgb']

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("PyTorch is running on the GPU!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is running on the CPU.")
# print the length of the image paths
print("Length of Image Paths: ", len(image_paths))
# Prepare the dataset and dataloader

dataset = ColorConstancyDataset(image_paths, groundtruth_illuminants,output_folder)
# dataset = ColorConstancyDataset(
#     image_paths=image_paths,
#     groundtruth_illuminants=groundtruth_illuminants,
#     output_folder=output_folder,
#     patch_size=32,  # Default patch size
#     max_size=1200,  # Default max image size
#     transform=None  # You can apply any transformations if needed
# )
#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loop through the dataloader to retrieve batches
#for batch_idx, (patches, groundtruth) in enumerate(dataloader):
#    print(f"Batch {batch_idx} - Patches shape: {patches[0].shape} - Ground Truth Illuminant: {groundtruth}")
#    break  # Stop after the first batch




# Define the model, optimizer, and loss function
model = ColorConstancyCNN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Instantiate the model, define the loss function and the optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Train the model using 3-fold cross-validation
train_loss_history, test_loss_history = train_model_kfold(
     dataset, 
     criterion, 
     optimizer=None,  # We define the optimizer inside the loop for each fold
     num_epochs=10, 
     n_splits=3
 )

print ("Training Loss History: ", train_loss_history)