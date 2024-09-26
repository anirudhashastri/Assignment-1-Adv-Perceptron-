import rawpy
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
import os
import cv2
import scipy.io as sio

def resize_image_to_multiple_of_32(image, max_size=1200):
    """Resize the image so that both dimensions are multiples of 32, while keeping aspect ratio."""
    #print("I am in the resize function")
    h, w = image.shape[:2]
    #print(f"Original image size: {w}x{h}")

    # Rescale the image so that the largest dimension is max_size
    if max(h, w) > max_size:
        scaling_factor = max_size / float(max(h, w))
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        #print(f"Resized image size (before adjusting to multiple of 32): {image.shape[1]}x{image.shape[0]}")

    # Adjust the dimensions to be multiples of 32
    new_h = (image.shape[0] // 32) * 32
    new_w = (image.shape[1] // 32) * 32

    # Resize the image again to ensure both dimensions are multiples of 32
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #print(f"Final image size (multiple of 32): {new_w}x{new_h}")
    
    return image

def histogram_stretching(image_patch):
    """Apply histogram stretching (contrast normalization) to a 32x32 image patch."""
    #print("I am in the histogram function")
    image_patch = image_patch.astype(np.float32)
    
    # For each channel (R, G, B), apply histogram stretching
    for i in range(3):  # Assuming the patch is in RGB format
        min_val = np.min(image_patch[:, :, i])
        max_val = np.max(image_patch[:, :, i])
        if max_val - min_val != 0:
            # Stretch the pixel values to the range [0, 255]
            image_patch[:, :, i] = (image_patch[:, :, i] - min_val) / (max_val - min_val) * 255
            
    return image_patch.astype(np.uint8)

def get_image_patches(image, patch_size=32):
    """Split the image into non-overlapping 32x32 patches."""
    #print("I am in the patch function")
    patches = []
    h, w, _ = image.shape
    #print(f"Extracting patches from image of size: {w}x{h}")

    # Iterate over the image to extract non-overlapping patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape == (patch_size, patch_size, 3):  # Ensure patch is of the correct size
                patches.append(patch)
    #print(f"Extracted {len(patches)} patches from the image.")
    return np.array(patches)

def load_dng_image(dng_path):
    """Loads a DNG image and converts it to an RGB image."""
    #print("I am in the load function")
    try:
        raw = rawpy.imread(dng_path)
        rgb_image = raw.postprocess()  # Converts the raw DNG image to an RGB image
        #print(f"Successfully loaded and processed DNG file: {dng_path}")
        return rgb_image
    except Exception as e:
        print(f"Failed to load DNG image: {dng_path}. Error: {e}")
        return None

def load_image(image_path):
    """Loads an image based on its file extension."""
    #print("I am in the load function")
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".dng":
        return load_dng_image(image_path)
    else:
        # Load other image formats (.jpg, .png, etc.) using OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            pass
            #print(f"Successfully loaded image: {image_path}")
        else:
            print(f"Failed to load image: {image_path}")
        return image

# def process_images_in_folder(folder_path_1D,folder_path_5D, output_folder, patch_size=32, max_size=1200):
#     """Process all supported images in a folder, resize them, and apply histogram stretching to patches."""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Check if the folder path is correct
#     if not os.path.exists(folder_path_5D):
#         print(f"Folder not found: {folder_path_5D}")
#         return
    
#     # Check if the folder path is correct
#     if not os.path.exists(folder_path_1D):
#         print(f"Folder not found: {folder_path_1D}")
#         return

#     # Supported image extensions
#     supported_extensions = (".jpg", ".jpeg", ".png", ".dng")

#     # Print files found in the folder for debugging
#     #files = os.listdir(folder_path)
#     #print(f"Files found in folder: {files}")

#     #-----------------------------------------------------------------------------#
#     # Get list of all image file paths (from both Canon 1D and Canon 5D)
#     canon_1d_images = sorted([os.path.join(folder_path_1D, f) for f in os.listdir(folder_path_1D) if f.endswith('.png')])
#     canon_5d_images = sorted([os.path.join(folder_path_5D, f) for f in os.listdir(folder_path_5D) if f.endswith('.png')])

#     # Combine the image file paths into one list
#     image_paths = canon_1d_images + canon_5d_images

#     all_patches= np.array([])



#     #-----------------------------------------------------------------------------#

#     for image_path in tqdm(image_paths):
#         if image_path.lower().endswith(supported_extensions):  # Case-insensitive check for supported extensions
#             #print(f"Processing file: {filename}")
#             #image_path = os.path.join(folder_path, filename)
            
#             # Step 1: Load the image
#             image = load_image(image_path)
#             if image is None:
#                 continue  # Skip if image could not be loaded

#             # Step 2: Resize the image to have dimensions that are multiples of 32
#             resized_image = resize_image_to_multiple_of_32(image, max_size=max_size)

#             # Step 3: Split the image into non-overlapping 32x32 patches
#             patches = get_image_patches(resized_image, patch_size=patch_size)

#             # Step 4: Apply histogram stretching to each patch
#             processed_patches = [histogram_stretching(patch) for patch in patches]
            
#             path_parts= image_path[:-4]
#             path_parts = path_parts.split('\\')
#             patch_name = path_parts[-1]

#             all_patches = np.append(all_patches,np.array(processed_patches))
            
#             # Step 5: Save each patch
#             for idx, patch in enumerate(processed_patches):
#                 patch_filename = f"{patch_name}_patch_{idx}.png"
#                 patch_output_path = os.path.join(output_folder, patch_filename)
#                 cv2.imwrite(patch_output_path, patch)
#                 #print(f"Saved patch: {patch_output_path}")
#         else:
#             print(f"Skipping unsupported file format: {image_path}")
#     # Open a file for binary write
#     with open('all_patches.pickle', 'wb') as file:
#         # Use pickle.dump to write the data to the file
#         pickle.dump(all_patches, file)

folder_path_1D = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\Canon1D'  # Folder containing your dataset of images
folder_path_5D = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\Dataset\Canon5D'  # Folder containing your dataset of images
output_folder = r'D:\My repos\Adv-perception-Assignment-1\Assignment-1-Adv-Perceptron-\ProcessedDataset'  # Folder where processed patches will be saved
#process_images_in_folder(folder_path_1D,folder_path_5D, output_folder)
