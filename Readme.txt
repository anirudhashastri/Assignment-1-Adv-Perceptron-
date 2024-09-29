Name: Anirudha Shastri
Teammates' names: Elliot Khouri, Venkata Satya Naga Sai Karthik Koduru
Time Travel Days Used:1

System Info:
Operating System: Windows 11 
IDE: Visual Studio code

Instructions to run the files:

The project consists of five files:

1.mask_application_script.py: The first step is to apply masks over the Macbeth ColorChecker present in all the images. To run this file , you need to specify the paths to the image dataset, the folder containing the masks, and the output folder where the masked images will be saved. Running this script will prepare your dataset by applying the necessary masks.

2.preprocessing.py: After applying the masks, the next step is to divide the images into 32x32 patches. To do this, specify the path to the dataset (the masked images) in the `preprocessing.py` file, and run the script. This script processes the images and prepares them for training by dividing them into patches suitable for input into the neural network.

3.Network.py: This file contains the Convolutional Neural Network (CNN) architecture. It defines the structure of the network that will be used to perform the color constancy task. The network’s architecture, layers, and forward pass logic are written here.

4.training.py: This file is responsible for training the CNN. You need to specify the path to the dataset (the preprocessed 32x32 patches) and also the path to the ground truth illuminant `.mat` file, which contains the correct illuminants for each image. Our teammate, Anirudha, was responsible for training the network, and after completing the training, he shared the trained model’s `.pth` file.

5.Testing.py: This script is used for testing the trained model. You will need to specify the path to the saved `.pth` file (the trained model), the test image dataset, and the ground truth illuminant `.mat` file. This script will load the trained model, apply it to the test dataset, and compare the predicted illuminants to the ground truth to evaluate the model's performance.


Required Libraries:

PyTorch
NumPy
OpenCV
SciPy
Tqdm 
Matplotlib 
