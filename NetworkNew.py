import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import scipy.io as sio
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.optim as optim

class ColorConstancyCNN(nn.Module):
    def __init__(self):
        super(ColorConstancyCNN, self).__init__()
        
        # Convolutional layer with 240 kernels of size 1x1x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=240, kernel_size=1, stride=1)
        
        # Max pooling layer with 8x8 kernel and stride of 8
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8)
        
        # Fully connected layer with 40 nodes
        self.fc1 = nn.Linear(240 * 4 * 4, 40)  # 240 feature maps of size 4x4 after pooling
        
        # Output layer with 3 nodes (to predict the illuminant color in RGB)
        self.fc2 = nn.Linear(40, 3)
        
    def forward(self, x):
        # Apply convolutional layer
        x = self.conv1(x)
        
        # Apply max pooling
        x = self.pool(x)
        
        # Flatten the feature maps
        #x = x.view(-1, 240 * 4 * 4)
        x = x.reshape(-1, 240 * 4 * 4)
        
        # Fully connected layer
        x = torch.relu(self.fc1(x))
        
        # Output layer (Illuminant prediction)
        x = self.fc2(x)
        
        return x






# Function to train the model and track losses for cross-validation
def train_model_kfold(dataset, criterion, optimizer, num_epochs=10, n_splits=3):
    # KFold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # To store the loss history for all folds
    fold_train_loss_history = []
    fold_test_loss_history = []

    # Iterate through each fold
    for fold, (train_indices, test_indices) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Prepare data loaders for the current fold
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

        # Initialize a new model for each fold
        model = ColorConstancyCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # To store the history of losses for this fold
        train_loss_history = []
        test_loss_history = []
        
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0

            # Training Loop
            for images, groundtruths in train_loader:
                if isinstance(images, list):
                    print("in tensor conversion")
                    images = torch.stack([torch.tensor(img) for img in images])  # Convert list of images to tensor
                print(f"Shape before permute: {images.shape}")  
                
                # If it's a 5D tensor, flatten the first two dimensions
                if images.dim() == 5:
                    # Flatten the first two dimensions (batch and patches) into one
                    images = images.view(-1, *images.shape[2:])  # Combine first two dimensions
                    #print(f"Shape after flattening: {images.shape}")        
                    
                
                images = images.permute(0, 3, 1, 2).float()  # Adjust shape for PyTorch (batch, C, H, W)
                groundtruths = groundtruths.float()

                optimizer.zero_grad()  # Zero the gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, groundtruths)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()

            epoch_train_loss = running_loss / len(train_loader)
            train_loss_history.append(epoch_train_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Fold {fold+1}, Training Loss: {epoch_train_loss:.4f}')
            
            # Evaluation loop (for test/validation set)
            model.eval()  # Set the model to evaluation mode
            test_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation for evaluation
                for images, groundtruths in test_loader:
                    images = images.permute(0, 3, 1, 2).float()  # Adjust shape for PyTorch (batch, C, H, W)
                    groundtruths = groundtruths.float()
                    outputs = model(images)
                    loss = criterion(outputs, groundtruths)
                    test_loss += loss.item()

            epoch_test_loss = test_loss / len(test_loader)
            test_loss_history.append(epoch_test_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Fold {fold+1}, Test Loss: {epoch_test_loss:.4f}')
        
        # Save the model for each fold
        model_save_path = f"color_constancy_cnn_fold_{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model for fold {fold+1} saved to {model_save_path}")
        
        # Store the loss history for this fold
        fold_train_loss_history.append(train_loss_history)
        fold_test_loss_history.append(test_loss_history)
    
    return fold_train_loss_history, fold_test_loss_history

