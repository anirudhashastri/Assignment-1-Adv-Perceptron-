Name: Anirudha Shastri
Teammates' names: Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Time Travel Days Used:1

=======

System Info:
Operating System: Windows 11 
IDE: Visual Studio code

Instructions to run the files:

The project consists of five files:

1.mask_application_script.py: The first step is to apply masks over the Macbeth ColorChecker present in all the images. To run this file , you need to specify the paths to the image dataset, the folder containing the masks, and the output folder where the masked images will be saved. Running this script will prepare your dataset by applying the necessary masks.

2.preprocessing.py: After applying the masks, the next step is to divide the images into 32x32 patches. To do this, specify the path to the dataset (the masked images) in the `preprocessing.py` file, and run the script. This script processes the images and prepares them for training by dividing them into patches suitable for input into the neural network.

3.ColorConstancyDataset.py:
This file defines the dataset class and the preprocessing steps needed for the CNN pipeline. It loads the images, applies transformations, and organizes them into batches for training and testing. The preprocessing steps include normalizing the images, resizing, and creating data loaders, which are then used by the training and testing scripts. Make sure the paths to the dataset and the required ground truth files are set before using this script. It's essential to call this dataset file before running the CNN pipeline, as it prepares the data for the network.

4.Network.py: This file contains the Convolutional Neural Network (CNN) architecture. It defines the structure of the network that will be used to perform the color constancy task. The network’s architecture, layers, and forward pass logic are written here.

5.training.py: This file is responsible for training the CNN. You need to specify the path to the dataset (the preprocessed 32x32 patches) and also the path to the ground truth illuminant `.mat` file, which contains the correct illuminants for each image. Our teammate, Anirudha, was responsible for training the network, and after completing the training, he shared the trained model’s `.pth` file.

6.Testing.py: This script is used for testing the trained model. You will need to specify the path to the saved `.pth` file (the trained model), the test image dataset, and the ground truth illuminant `.mat` file. This script will load the trained model, apply it to the test dataset, and compare the predicted illuminants to the ground truth to evaluate the model's performance.


Required Libraries:

PyTorch
NumPy
OpenCV
SciPy
Tqdm 
Matplotlib 


File Structure 

ASSIGNMENT-1-ADV-PERCEPTRON
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
├── Readme.txt
├── Testing.py
├── training.py






