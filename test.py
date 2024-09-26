import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import scipy.io as sio
from preprocessing import load_image, resize_image_to_multiple_of_32, get_image_patches, histogram_stretching
from ColorConstancyDataset import ColorConstancyDataset


class AngularLoss(torch.nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # Normalize the predicted and ground truth vectors
        y_pred_norm = torch.nn.functional.normalize(y_pred, dim=1)
        y_true_norm = torch.nn.functional.normalize(y_true, dim=1)
        
        # Compute the dot product between the normalized predictions and ground truth
        dot_product = torch.sum(y_pred_norm * y_true_norm, dim=1)
        dot_product = torch.clamp(dot_product, min=-1.0, max=1.0)  # Clamp to avoid numerical issues
        
        # Compute the angular error in radians and convert to degrees
        angular_error = torch.acos(dot_product) * (180.0 / 3.141592653589793)
        
        # Return the mean angular error over the batch
        return torch.mean(angular_error)





# Step 1: Define the CNN Architecture
class ColorConstancyCNN(nn.Module):
    def __init__(self):
        super(ColorConstancyCNN, self).__init__()
        
        # Convolutional layer with 240 kernels of size 1x1x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=240, kernel_size=1, stride=1)
        
        # Max pooling layer with 8x8 kernel and stride of 8
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        
        # Fully connected layer with 40 nodes
        self.fc1 = nn.Linear(240 * 4 * 4, 40)
        
        # Output layer with 3 nodes (to predict the illuminant color in RGB)
        self.fc2 = nn.Linear(40, 3)
        
    def forward(self, x):
        # Apply convolutional layer
        x = self.conv1(x)
        
        # Apply max pooling
        x = self.pool(x)
        
        # Flatten the feature maps using reshape instead of view
        x = x.reshape(-1, 240 * 4 * 4)
        
        # Fully connected layer
        x = torch.relu(self.fc1(x))
        
        # Output layer (Illuminant prediction)
        x = self.fc2(x)
        
        return x

# Step 2: Load the Dataset and Ground Truth
def load_data():
    canon_1d_path = r'E:\Adv perception\Assignment-1-Adv-Perceptron-\Dataset\Canon1D'
    canon_5d_path = r'E:\Adv perception\Assignment-1-Adv-Perceptron-\Dataset\Canon5D'
    output_folder = r'E:\Adv perception\Assignment-1-Adv-Perceptron-\ProcessedDataset'
    groundtruth_path = r'E:\Adv perception\Assignment-1-Adv-Perceptron-\Dataset\real_illum_568.mat'

    canon_1d_images = sorted([os.path.join(canon_1d_path, f) for f in os.listdir(canon_1d_path) if f.endswith('.png')])
    canon_5d_images = sorted([os.path.join(canon_5d_path, f) for f in os.listdir(canon_5d_path) if f.endswith('.png')])
    image_paths = canon_1d_images + canon_5d_images

    # Load ground truth illuminants
    groundtruth_data = sio.loadmat(groundtruth_path)
    groundtruth_illuminants = groundtruth_data['real_rgb']

    # Prepare the dataset
    dataset = ColorConstancyDataset(image_paths, groundtruth_illuminants, output_folder)
    
    return dataset

# Step 3: K-Fold Cross-Validation Training Function
def train_model_kfold(dataset, criterion, num_epochs=10, n_splits=3):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_train_loss_history = []
    fold_test_loss_history = []

    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{n_splits}")
        
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)

        model = ColorConstancyCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        train_loss_history = []
        test_loss_history = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, groundtruths in train_loader:
                print(f"Shape after flattening: {images.shape}")
                if images.dim() == 5:
                    # Flatten the num_patches dimension into the batch dimension
                    images = images.view(-1, *images.shape[2:])  # Combine batch_size and num_patches into one dimension
                
                # Now images should be 4D (batch_size * num_patches, height, width, channels)
                print(f"Shape after flattening: {images.shape}")
                images = images.permute(0, 3, 1, 2).float().to(device)
                groundtruths = groundtruths.float().to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Reshape outputs back to (batch_size, num_patches, 3) and average the predictions across all patches
                num_patches_per_image = outputs.shape[0] // groundtruths.shape[0]  # Calculate how many patches per image
                outputs = outputs.view(-1, num_patches_per_image, 3)  # Reshape to (batch_size, num_patches, 3)
                outputs = outputs.mean(dim=1)  # Average over the patches for each image


                loss = criterion(outputs, groundtruths)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            epoch_train_loss = running_loss / len(train_loader)
            train_loss_history.append(epoch_train_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Fold {fold+1}, Training Loss: {epoch_train_loss:.4f}')
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, groundtruths in test_loader:
                    if images.dim() == 5:
                    # Flatten the num_patches dimension into the batch dimension
                        images = images.view(-1, *images.shape[2:])  # Combine batch_size and num_patches into one dimension

                    images = images.permute(0, 3, 1, 2).float().to(device)
                    groundtruths = groundtruths.float().to(device)
                    outputs = model(images)

                    # Reshape outputs back to (batch_size, num_patches, 3) and average the predictions across all patches
                    num_patches_per_image = outputs.shape[0] // groundtruths.shape[0]  # Calculate how many patches per image
                    outputs = outputs.view(-1, num_patches_per_image, 3)  # Reshape to (batch_size, num_patches, 3)
                    outputs = outputs.mean(dim=1)  # Average over the patches for each image
                    
                    loss = criterion(outputs, groundtruths)
                    test_loss += loss.item()

            epoch_test_loss = test_loss / len(test_loader)
            test_loss_history.append(epoch_test_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Fold {fold+1}, Test Loss: {epoch_test_loss:.4f}')

        # Save the model for each fold
        model_save_path = f"color_constancy_angular_cnn_fold_{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for fold {fold+1} saved to {model_save_path}")
        
        fold_train_loss_history.append(train_loss_history)
        fold_test_loss_history.append(test_loss_history)
    
    return fold_train_loss_history, fold_test_loss_history

# Step 4: Main Pipeline Execution
def main():
    dataset = load_data()

    # Instantiate the model, define the loss function
    #criterion = # Instantiate the custom Angular Loss function
    criterion = AngularLoss()

    # Train the model using 3-fold cross-validation
    train_loss_history, test_loss_history = train_model_kfold(
        dataset,
        criterion,
        num_epochs=10,
        n_splits=3
    )

    print("Training Loss History: ", train_loss_history)
    print("Test Loss History: ", test_loss_history)

if __name__ == "__main__":
    main()
