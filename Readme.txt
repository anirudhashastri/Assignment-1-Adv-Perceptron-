Name: Anirudha Shastri
Teammates' names: Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Time Travel Days Used:1

=======

System Info:
Operating System: Windows 11 
IDE: Visual Studio code

Instructions to run the files:

A basic CNN color constancy architecture was implemented in assignment 1, in assignment 2 the architecture was further refined through hyperparameter tuning and experimental testing. The network is trained and run on the color constancy dataset with 17 different sets of parameters (activation functions, dropout rates, filter sizes etc.), additionally a standalone GAN is trained and run on the dataset for comparison, and finally a hybrid approach is used where the CNN output is run through the GAN for further refinement.  

The project consists of eight files:

1.mask_application_script.py: The first step is to apply masks over the Macbeth ColorChecker present in all the images. To run this file , you need to specify the paths to the image dataset, the folder containing the masks, and the output folder where the masked images will be saved. Running this script will prepare your dataset by applying the necessary masks.

2.preprocessing.py: After applying the masks, the next step is to divide the images into 32x32 patches. To do this, specify the path to the dataset (the masked images) in the `preprocessing.py` file, and run the script. This script processes the images and prepares them for training by dividing them into patches suitable for input into the neural network.

3.ColorConstancyDataset.py:
This file defines the dataset class and the preprocessing steps needed for the CNN pipeline. It loads the images, applies transformations, and organizes them into batches for training and testing. The preprocessing steps include normalizing the images, resizing, and creating data loaders, which are then used by the training and testing scripts. Make sure the paths to the dataset and the required ground truth files are set before using this script. It's essential to call this dataset file before running the CNN pipeline, as it prepares the data for the network.

4.Network.py: This file contains the Convolutional Neural Network (CNN) architecture. It defines the structure of the network that will be used to perform the color constancy task. The network’s architecture, layers, and forward pass logic are written here.

5.training.py: This file is responsible for training the CNN. You need to specify the path to the dataset (the preprocessed 32x32 patches) and also the path to the ground truth illuminant `.mat` file, which contains the correct illuminants for each image. Our teammate, Anirudha, was responsible for training the network, and after completing the training, he shared the trained model’s `.pth` file.

6.Testing.py: This script is used for testing the trained model. You will need to specify the path to the saved `.pth` file (the trained model), the test image dataset, and the ground truth illuminant `.mat` file. This script will load the trained model, apply it to the test dataset, and compare the predicted illuminants to the ground truth to evaluate the model's performance.

7. TuningNetwork.py: This file contains the modified CNN architecture used for hyperparameter tuning. It defines the structure of the network, which includes options for different filter sizes (1x1 and 3x3), activation functions (ReLU, PReLU, LeakyReLU), and dropout rates. This flexibility allows us to systematically explore which configurations yield the best results for the color constancy task. This file is essential for parameter tuning as it forms the foundation for the network variations.

8. ModelTuning.py: This script is responsible for performing hyperparameter tuning on the CNN architecture. It uses combinations of hyperparameters such as learning rate, filter sizes, activation functions, dropout rates, and batch size, and trains models using K-Fold Cross-Validation. The script saves the results of each run (training/testing loss, training time, model parameters) into .pkl files for further analysis. Additionally, the trained models are saved for later use and comparison.

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
├── TuningNetwork.py
├── ModelTuning.py






