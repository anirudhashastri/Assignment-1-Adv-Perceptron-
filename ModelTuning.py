import torch
import torch.optim as optim
import os
import gc
import pickle
import time
from training import load_data  
from Network import AngularLoss  
from TuningNetwork import ColorConstancyCNN
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm  # Importing tqdm for progress bar

# Function to save results to pickle
def save_to_pickle(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

# Function to clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Function to automatically generate model names
def generate_model_name(params):
    return f"model_lr_{params['learning_rate']}_bs_{params['batch_size']}_dropout_{params['dropout']}_filters_{params['filter_size']}_activation_{params['activation']}"

# Main training pipeline
def train_pipeline():
    # Hyperparameters to search through
    learning_rates = [0.001, 0.01]
    batch_sizes = [9]
    dropout_rates = [0,0.15, 0.3]
    filter_sizes = ['1x1','3x3']  # Multi-scale convolution filter sizes
    activations = ['Relu','PReLU', 'LeakyReLU']  # Activation functions
    num_epochs = 2

    # Load the dataset
    dataset = load_data()

    # Placeholder for storing results
    results = []
    
    # Iterate over each hyperparameter combination
    for filter_size in filter_sizes:
        for activation_function in activations:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    for dropout_rate in dropout_rates:
                        
                        # Setup parameters for this model run
                        params = {
                            'filter_size': filter_size,
                            'activation': activation_function,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'dropout': dropout_rate
                        }

                        # Generate model name
                        model_name = generate_model_name(params)

                        print(f"Training model: {model_name}")
                        
                        # Start timing the model training process
                        model_start_time = time.time()
                        
                        # Create model, optimizer, and criterion
                        model = ColorConstancyCNN(filter_size, activation_function, dropout_rate).to(device='cuda')
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = AngularLoss()
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = model.to(device)

                        # KFold cross-validation setup
                        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
                        fold_train_loss_history = []
                        fold_test_loss_history = []
                        
                        # Run K-Fold Cross-Validation
                        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
                            print(f"Fold {fold+1}/3")
                            train_subset = torch.utils.data.Subset(dataset, train_idx)
                            test_subset = torch.utils.data.Subset(dataset, test_idx)
                            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

                            train_loss_history = []
                            test_loss_history = []
                            model.train()

                            # Use tqdm for a progress bar in each epoch
                            for epoch in tqdm(range(num_epochs), desc=f'Epochs for model {model_name} - Fold {fold+1}'):  
                                running_loss = 0.0
                                for images, groundtruths in train_loader:
                                    if images.dim() == 5:
                                        images = images.view(-1, *images.shape[2:])  # Combine batch_size and num_patches into one dimension

                                    images = images.permute(0, 3, 1, 2).float().to(device)
                                    groundtruths = groundtruths.float().to(device)

                                    optimizer.zero_grad()
                                    outputs = model(images)

                                    num_patches_per_image = outputs.shape[0] // groundtruths.shape[0]
                                    outputs = outputs.view(-1, num_patches_per_image, 3)
                                    outputs = outputs.mean(dim=1)

                                    loss = criterion(outputs, groundtruths)
                                    loss.backward()
                                    optimizer.step()
                                    running_loss += loss.item()

                                train_loss_history.append(running_loss / len(train_loader))

                                # Evaluate on test set
                                model.eval()
                                test_loss = 0.0
                                with torch.no_grad():
                                    for images, groundtruths in test_loader:
                                        if images.dim() == 5:
                                            images = images.view(-1, *images.shape[2:])

                                        images = images.permute(0, 3, 1, 2).float().to(device)
                                        groundtruths = groundtruths.float().to(device)
                                        outputs = model(images)

                                        num_patches_per_image = outputs.shape[0] // groundtruths.shape[0]
                                        outputs = outputs.view(-1, num_patches_per_image, 3)
                                        outputs = outputs.mean(dim=1)

                                        loss = criterion(outputs, groundtruths)
                                        test_loss += loss.item()

                                test_loss_history.append(test_loss / len(test_loader))
                                print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {running_loss / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}")

                            fold_train_loss_history.append(train_loss_history)
                            fold_test_loss_history.append(test_loss_history)

                        # Save model weights
                        model_path = f"saved_models/{model_name}.pth"
                        torch.save(model.state_dict(), model_path)

                        # End timing the model training process and calculate the elapsed time
                        model_end_time = time.time()
                        training_time = model_end_time - model_start_time
                        print(f"Training time for model {model_name}: {training_time:.2f} seconds")

                        # Store the results for comparison study, including training time
                        result = {
                            'model_name': model_name,
                            'filter_size': filter_size,
                            'activation': activation_function,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'dropout': dropout_rate,
                            'train_loss': fold_train_loss_history,
                            'test_loss': fold_test_loss_history,
                            'training_time': training_time,  # Save the training time
                            'model_path': model_path
                        }

                        results.append(result)

                        # Save the result to a pickle file
                        save_to_pickle(result, f"results/{model_name}_result.pkl")
                        
                        # Clear GPU memory before next model
                        clear_gpu_memory()

    # Save all results after training
    save_to_pickle(results, 'results/all_results.pkl')

if __name__ == "__main__":
    start_time = time.time()
    os.makedirs('saved_models', exist_ok=True)  # Create directory for saving models
    os.makedirs('results', exist_ok=True)  # Create directory for saving results
    train_pipeline()
    print(f"Training pipeline completed in {time.time() - start_time:.2f} seconds")
