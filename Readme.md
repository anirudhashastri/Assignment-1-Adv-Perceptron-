# Advanced Perception Project - Color Constancy with CNN & GAN

**Team Members**

- **Anirudha Shastri**
- **Elliot Khouri**
- **Venkata Satya Naga Sai Karthik Koduru**
- **Time Travel Days Used**: 1

---

## System Information

- **Operating System**: Windows 11
- **IDE**: Visual Studio Code

---

## Project Overview

A basic CNN color constancy architecture was implemented in Assignment 1, and in Assignment 2, the architecture was further refined through hyperparameter tuning and experimental testing. The network was trained and tested on the color constancy dataset using 17 different sets of parameter combinations (activation functions, dropout rates, filter sizes, etc.). Additionally, a standalone GAN was trained on the dataset for comparison, and finally, a hybrid approach was used where the CNN output was run through the GAN for further refinement.

---

## Instructions to Run the Files

### CNN Files

1. **`mask_application_script.py`**

   - 📸 _Description_: Applies masks over the Macbeth ColorChecker present in all images.
   - 🔧 _How to Run_: Specify the paths to the image dataset, the folder containing the masks, and the output folder where masked images will be saved.

2. **`preprocessing.py`**

   - 🖼️ _Description_: Divides masked images into 32x32 patches.
   - 🔧 _How to Run_: Specify the path to the dataset in the script and execute to process images for training.

3. **`ColorConstancyDataset.py`**

   - 🗂️ _Description_: Defines the dataset class and preprocessing steps for the CNN. It handles image transformations, resizing, and data loader creation.
   - 🔧 _How to Run_: Make sure the dataset paths and ground truth files are correctly set before running the script.

4. **`Network.py`**

   - 🤖 _Description_: Defines the CNN architecture used for the color constancy task. Includes the model’s layers, forward pass logic, and activation functions.

5. **`training.py`**

   - 🎓 _Description_: Responsible for training the CNN using the preprocessed patches and ground truth illuminant `.mat` files.
   - 🔧 _How to Run_: Set the paths for both dataset and illuminant files, then run the script.

6. **`Testing.py`**

   - 🔍 _Description_: Tests the trained model on a test dataset and evaluates performance against ground truth.
   - 🔧 _How to Run_: Provide the path to the trained model and test dataset to perform testing.

7. **`TuningNetwork.py`**

   - ⚙️ _Description_: Defines a modified CNN architecture to allow for hyperparameter tuning. Supports different filter sizes (`1x1`, `3x3`), activation functions (`ReLU`, `PReLU`, `LeakyReLU`), and dropout rates.

8. **`ModelTuning.py`**
   - 📊 _Description_: Performs hyperparameter tuning using combinations of hyperparameters and K-Fold Cross-Validation. Saves results for further analysis.

### GAN Files

1. **`dataloader.py`**

   - 🗃️ _Description_: Loads `.tiff` images and illuminant values for GAN training. Converts images to tensors after resizing to 256x256.

2. **`network.py`**

   - 🔄 _Description_: Defines both the generator and discriminator of the GAN. The generator uses ResNet blocks to create color-corrected images, and the discriminator learns to classify real vs. generated images.

3. **`preprocessing.py`**

   - ⚗️ _Description_: Provides preprocessing functions including resizing images, extracting 32x32 patches, and applying histogram stretching for contrast normalization.

4. **`test.py`**

   - 🧪 _Description_: Tests the trained GAN model on an input image and saves the generated white-balanced image. Also, displays input vs. output images side by side.

5. **`train.py`**
   - 📈 _Description_: Responsible for training the GAN. Loads images and illuminant data, and iteratively trains both generator and discriminator using Mean Squared Error (MSE) loss.

---

## Required Libraries

- **PyTorch**
- **NumPy**
- **OpenCV**
- **SciPy**
- **Tqdm**
- **Matplotlib**

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
