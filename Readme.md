# Advanced Perception Project - Color Constancy with CNN & GAN

---

## Team Members

**Name:** Anirudha Shastri  
**Teammates' names:** Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru  
**Time Travel Days Used:** 1

---

## System Info

- **Operating System:** Windows 11
- **IDE:** Visual Studio Code

---

## Project Description

A basic CNN color constancy architecture was implemented in Assignment 1. In Assignment 2, the architecture was further refined through hyperparameter tuning and experimental testing. The network was trained and tested on the color constancy dataset using 17 different parameter combinations (activation functions, dropout rates, filter sizes, etc.). Additionally, a standalone GAN was trained and tested on the dataset for comparison, and finally, a hybrid approach was used where the CNN output was passed through the GAN for further refinement.

---

## Instructions to Run the Files

### CNN:

1. **mask_application_script.py**  
   The first step is to apply masks over the Macbeth ColorChecker present in all images.

   - To run this file, specify the paths to the image dataset, the folder containing the masks, and the output folder where the masked images will be saved.
   - Running this script will prepare your dataset by applying the necessary masks.

2. **preprocessing.py**  
   After applying the masks, divide the images into 32x32 patches.

   - Specify the path to the dataset (masked images) in the `preprocessing.py` file, and run the script.
   - This processes the images, preparing them for training by dividing them into patches suitable for input into the neural network.

3. **ColorConstancyDataset.py**  
   Defines the dataset class and preprocessing steps for the CNN pipeline.

   - It loads the images, applies transformations, and organizes them into batches for training and testing.
   - Preprocessing includes normalizing images, resizing, and creating data loaders. Ensure paths to the dataset and required ground truth files are set before using this script.

4. **Network.py**  
   Defines the CNN architecture.

   - It contains the structure of the network used to perform the color constancy task, including the network layers and forward pass logic.

5. **training.py**  
   Responsible for training the CNN.

   - Specify the path to the dataset (preprocessed 32x32 patches) and the ground truth illuminant `.mat` file.
   - Anirudha trained the network and shared the trained model’s `.pth` file after completion.

6. **Testing.py**  
   Used for testing the trained model.

   - Specify the path to the saved `.pth` file (trained model), the test image dataset, and the ground truth illuminant `.mat` file.
   - The script will load the trained model, apply it to the test dataset, and compare the predicted illuminants to the ground truth.

7. **TuningNetwork.py**  
   Contains the modified CNN architecture for hyperparameter tuning.

   - Defines the structure of the network, including options for different filter sizes (1x1 and 3x3), activation functions (ReLU, PReLU, LeakyReLU), and dropout rates.

8. **ModelTuning.py**  
   Responsible for hyperparameter tuning on the CNN architecture.
   - Uses combinations of hyperparameters (learning rate, filter sizes, activation functions, dropout rates, batch size) and trains models using K-Fold Cross-Validation.
   - Saves results and trained models for later use and comparison.

### GAN:

1. **dataloader.py**  
   Handles data loading for the GAN.

   - Functions to load `.tiff` or `.tif` images from a specified folder and illuminant values from an Excel file.
   - Resizes images to 256x256 pixels and converts them to tensors.

2. **network.py**  
   Defines the GAN architecture (generator and discriminator models).

   - The generator uses ResNet blocks to generate color-corrected images, while the discriminator distinguishes between real and fake images.

3. **preprocessing.py**  
   Contains preprocessing functions for the GAN.

   - This includes resizing images to multiples of 32, extracting 32x32 patches, and applying histogram stretching for contrast normalization.

4. **test.py**  
   Used to test the trained GAN model.

   - Takes an input image, applies the generator, and produces a white-balanced output.
   - Displays the input and generated images side-by-side for easy comparison.

5. **train.py**  
   Responsible for training the GAN model.
   - Uses images and illuminant data to iteratively train both the generator and discriminator models, and optimizes using Mean Squared Error (MSE) loss.

---

## Required Libraries:

- PyTorch
- NumPy
- OpenCV
- SciPy
- Tqdm
- Matplotlib

---

## File Structure

```plaintext
ASSIGNMENT-1-ADV-PERCEPTRON/
├── __pycache__/
├── Dataset/
│   ├── 1D/
│   ├── 1D-Masks/
│   ├── 5D/
│   ├── 5D-Masks/
│   ├── Canon1D/
│   ├── Canon5D/
│   ├── colourchecker_gamma1_bit12.mat
│   ├── real_illum_568.mat
├── MaskedDataset/
│   ├── Canon1D/
│   ├── Canon5D/
├── ProcessedDataset/
├── .gitignore
├── color_constancy_angular_cnn_2_fold_1.pth
├── color_constancy_angular_cnn_2_fold_2.pth
├── color_constancy_angular_cnn_2_fold_3.pth
├── ColorConstancyDataset.py
├── mask_application_script.py
├── Network.py
├── preprocessing.py
├── Readme.md
├── Testing.py
├── training.py
├── TuningNetwork.py
├── ModelTuning.py
├── GAN/
│   ├── dataloader.py
│   ├── Network.py
│   ├── preprocessing.py
│   ├── test.py
│   ├── train.py
```
