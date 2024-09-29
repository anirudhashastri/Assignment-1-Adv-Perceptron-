'''
Anirudha Shastri, Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
9/27/2024
CS 7180 Advanced Perception
Travel Days Used: 1
'''

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2
import scipy.io as sio
from preprocessing import load_image, resize_image_to_multiple_of_32, get_image_patches, histogram_stretching
from torch.utils.data import DataLoader
class ColorConstancyDataset(Dataset):
    def __init__(self, image_paths, groundtruth_illuminants, output_folder, patch_size=32, max_size=1200, transform=None):
        """
        Args:
            image_paths (list): List of image file paths.
            groundtruth_illuminants (ndarray): Corresponding ground truth illuminants (in the same order as image_paths).
            output_folder (str): Path where patches are saved.
            patch_size (int): Size of each patch.
            max_size (int): Maximum size for resizing.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = image_paths
        self.groundtruth_illuminants = groundtruth_illuminants  # Ground truth RGB values for each image
        self.output_folder = output_folder
        self.patch_size = patch_size
        self.max_size = max_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path for the given index
        #print("I am here")
        image_path = self.image_paths[idx]

        supported_extensions = (".jpg", ".jpeg", ".png", ".dng","tiff")  # Case-insensitive check for supported extensions
        if image_path.lower().endswith(supported_extensions):
            #print("i passed the lower case check")
            # Step 1: Load the image
            image = load_image(image_path)
            if image is None:
                return None  # Return None if image could not be loaded
            
            #print(f"Shape before permute in prepoces: {image.shape}") 
            # Step 2: Resize the image to have dimensions that are multiples of 32
            resized_image = resize_image_to_multiple_of_32(image, max_size=self.max_size)

            # Step 3: Split the image into non-overlapping 32x32 patches
            patches = get_image_patches(resized_image, patch_size=self.patch_size)

            # Step 4: Apply histogram stretching to each patch
            processed_patches = [histogram_stretching(patch) for patch in patches]
            
            # Extract patch name
            #path_parts = image_path[:-4]
            #path_parts = path_parts.split('\\')
            #patch_name = path_parts[-1]

            # Combine all patches for this image
            #all_patches = np.array(processed_patches)
            
            # Step 5: Save each patch
            #for idx, patch in enumerate(processed_patches):
            #    patch_filename = f"{patch_name}_patch_{idx}.png"
            #    patch_output_path = os.path.join(self.output_folder, patch_filename)
            #    cv2.imwrite(patch_output_path, patch)

            # Save all patches to a pickle file
            #with open('all_patches.pickle', 'wb') as file:
            #    pickle.dump(all_patches, file)
            
            # Convert the patches to tensors
            processed_patches = [torch.tensor(patch, dtype=torch.float32) for patch in patches]

            # Stack the patches into a single tensor (batch of patches)
            processed_patches_tensor = torch.stack(processed_patches)
            
            # Get the ground truth illuminant for this image
            groundtruth_illuminant = self.groundtruth_illuminants[idx]  # Assuming the groundtruth is already aligned with image_paths
            
            # Optionally apply a transform if provided
            if self.transform:
                processed_patches = self.transform(processed_patches)

            # Return the processed patches and corresponding ground truth illuminant
            return processed_patches_tensor, torch.tensor(groundtruth_illuminant, dtype=torch.float32)

        else:
            print(f"Skipping unsupported file format: {image_path}")
            return None





