import cv2
import numpy as np
import os
import re

# Path to the folder containing images and masks
#image_folder = "E:/Adv perception/Assignment-1-Adv-Perceptron-/Dataset/1D"
#mask_folder = "E:/Adv perception/Assignment-1-Adv-Perceptron-/Dataset/1D-Masks"
#output_folder = "E:/Adv perception/Assignment-1-Adv-Perceptron-/MaskedDataset/Canon1D"

image_folder = "E:/Adv perception/Assignment-1-Adv-Perceptron-/Dataset/5D"
mask_folder = "E:/Adv perception/Assignment-1-Adv-Perceptron-/Dataset/5D-Masks"
output_folder = "E:/Adv perception/Assignment-1-Adv-Perceptron-/MaskedDataset/Canon5D"

# Function to apply masks to a color image

def apply_black_regions_from_masks(image, masks):
    """
    Apply the black regions from masks to the image by darkening those regions.
    
    Parameters:
    image (numpy array): The original image.
    masks (list of numpy arrays): A list of mask images.

    Returns:
    numpy array: The image with the black regions applied from the masks.
    """
    # Convert the original image to a NumPy array if not already
    image_np = np.array(image)
    
    for mask in masks:
        # Convert the mask to grayscale
        #mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_gray= mask
        # Create a boolean mask where black parts are represented (i.e., pixel value is 0)
        black_only_mask = mask_gray == 0
        
        # Apply the black parts of the mask to the original image by setting those regions to black
        image_np[black_only_mask] = [0, 0, 0]  # Darken the region in the image
        
    # Return the modified image
    return image_np

# Function to clean the mask filename and remove the number between "maskX_" and the actual filename
def clean_mask_filename(mask_path):
    # Use a regular expression to remove the number between "maskX_" and the actual filename
    cleaned_path = re.sub(r"(mask\d+)_\d+_", r"\1_", mask_path)
    return cleaned_path

# Function to display each mask
def display_masks(masks):
    for mask in masks:    
        # Display the mask
        cv2.imshow('Mask', mask)
        
        # Wait for the user to press 'q' to move to the next mask
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Loop through the images
for image_name in os.listdir(image_folder):
    if image_name.endswith(".tiff"):
        image_path = image_folder + '/' + image_name
        # Load the color image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Check if image was loaded successfully
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Extract the base name to find corresponding masks
        base_name = os.path.splitext(image_name)[0]

        # Find the corresponding masks
        mask1_path = mask_folder + '/mask1_' + base_name + '.tiff'
        mask2_path = mask_folder + '/mask2_' + base_name + '.tiff'
        mask3_path = mask_folder + '/mask3_' + base_name + '.tiff'

        # Clean the mask paths to remove extra numbers
        mask1_path = clean_mask_filename(mask1_path)
        mask2_path = clean_mask_filename(mask2_path)
        mask3_path = clean_mask_filename(mask3_path)

        # Print the paths to help debug
        print(f"Trying to load masks for image: {image_name}")
        
        # Load the masks as grayscale
        mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
        mask3 = cv2.imread(mask3_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if masks were loaded successfully
        if mask1 is None or mask2 is None or mask3 is None:
            print(f"Could not load one or more masks for image: {image_name}")
            continue
        
        # Apply the masks to the image
        masks = [mask1, mask2, mask3]

        # Call the function with the list of masks
        #display_masks(masks)
        result_image = apply_black_regions_from_masks(image, masks)
        #cv2.imshow('Mask', result_image)
        
        # Wait for the user to press 'q' to move to the next mask
        #while True:
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
        # Close all OpenCV windows
        #cv2.destroyAllWindows()

        
        # Save the resulting image
        output_path = os.path.join(output_folder, f'{image_name}')
        
        cv2.imwrite(output_path, result_image)
print("Pipeline completed.")
