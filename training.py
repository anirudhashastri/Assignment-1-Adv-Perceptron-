'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Assignment 2
10/19/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import scipy.io as sio
from preprocessing import load_image, resize_image_to_multiple_of_32, get_image_patches, histogram_stretching
from ColorConstancyDataset import ColorConstancyDataset
from Network import train_model_kfold,AngularLoss



# Load the Dataset and Ground Truth
def load_data():
    canon_1d_path = r'MaskedDataset\Canon1D'
    canon_5d_path = r'MaskedDataset\Canon5D'
    output_folder = r'ProcessedDataset'
    groundtruth_path = r'Dataset\real_illum_568.mat'

    canon_1d_images = sorted([os.path.join(canon_1d_path, f) for f in os.listdir(canon_1d_path) if f.endswith('.tiff')])
    canon_5d_images = sorted([os.path.join(canon_5d_path, f) for f in os.listdir(canon_5d_path) if f.endswith('.tiff')])
    image_paths = canon_1d_images + canon_5d_images

    # Load ground truth illuminants
    groundtruth_data = sio.loadmat(groundtruth_path)
    groundtruth_illuminants = groundtruth_data['real_rgb']

    # Prepare the dataset
    dataset = ColorConstancyDataset(image_paths, groundtruth_illuminants, output_folder)
    print("Length of dataset")
    print(len(dataset))
    return dataset




# Step 4: Main Pipeline Execution
def main():
    dataset = load_data()

    # Instantiate the model, define the loss function
    criterion = AngularLoss()

    # Train the model using 3-fold cross-validation
    train_loss_history, test_loss_history = train_model_kfold(
        dataset,
        criterion,
        num_epochs=5,
        n_splits=3
    )

    print("Training Loss History: ", train_loss_history)
    print("Test Loss History: ", test_loss_history)


if __name__ == "__main__":
    main()
