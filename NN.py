"""
NN module to train 
"""
import torch
import uproot
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

sns.set_style("darkgrid")

# === DATA PREPARATION  === #
class ROOTDataset(Dataset):
    def __init__(self, signal_file, background_file, variables, max_events=None):
        self.variables = variables

        self.signal_tree = uproot.open(signal_file)["Tsignal"]
        self.background_tree = uproot.open(background_file)["Tback"]

        s_arrs = self.signal_tree.arrays(variables, library="np")
        b_arrs = self.background_tree.arrays(variables, library="np")

        s_X = np.column_stack([s_arrs[v] for v in variables]).astype(np.float32)
        b_X = np.column_stack([b_arrs[v] for v in variables]).astype(np.float32)

        if max_events is not None:
            s_X = s_X[:max_events // 2]
            b_X = b_X[:max_events // 2]

        s_y = np.ones(len(s_X), dtype=np.float32)
        b_y = np.zeros(len(b_X), dtype=np.float32)

        self.X = np.concatenate([s_X, b_X], axis=0)
        self.y = np.concatenate([s_y, b_y], axis=0)

        self.signal_entries = len(s_X)
        self.background_entries = len(b_X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])
    
# OR 
def prepdata(dir_path, root_mc, root_data, variables):
    """
    Load ROOT file, extract features and labels from two trees (signal and background),
    and prepare numpy arrays for training.
    """

    # Open ROOT file and tree
    file_signal = uproot.open(f"{dir_path}/{root_mc}")
    file_back = uproot.open(f"{dir_path}/{root_data}")


    # Acess signal and background trees
    tree_signal = file_signal["Tsignal"]
    tree_background = file_back["Tback"]
 
    # Load features as numpy arrays from ROOT trees
    signal_arrays = tree_signal.arrays(variables, library="np")
    background_arrays = tree_background.arrays(variables, library="np")

    # Stack features column-wise (shape: [n_events, n_features])
    X_signal = np.column_stack([signal_arrays[var] for var in variables])
    X_background = np.column_stack([background_arrays[var] for var in variables])

    # Create y (labels): 1 for signal, 0 for background
    y_signal = np.ones(X_signal.shape[0])
    y_background = np.zeros(X_background.shape[0])

    # Combine for machine learning
    X = np.concatenate([X_signal, X_background])
    y = np.concatenate([y_signal, y_background])

    return X.astype(np.float32), y.astype(np.float32)


class ClassificationDataset(Dataset):
    """
    PyTorch dataset wrapping features and labels.
    """
    def __init__(self, X, y):
        # Convert to torch tensors
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


# === MODEL === #
class ClassificationModel(nn.Module):
    """
    A simple feedforward neural network for binary classification.

    Architecture (based on previous data optuna):
    - Input layer of size `input_size`
    - First hidden layer with 45 neurons and ReLU activation
    - Second hidden layer with 45 neurons and ReLU activation
    - Third hidden layer with 56 neurons and ReLU activation
    - Output layer with 1 neuron and sigmoid activation to produce a probability in [0, 1]
    - Dropout applied (p=0.1) to reduce overfitting
    """
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        #Baseline
        # 1st hidden layer
        #self.fc1 = nn.Linear(input_size, 45) 
        #self.relu1 = nn.ReLU()
        #self.dropout1 = nn.Dropout(0.1)
        # 2nd hidden layer
        #self.fc2 = nn.Linear(45, 45)
        #self.relu2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(0.1)
        # 3rd hidden layer
        #self.fc3 = nn.Linear(45, 56)
        #self.relu3 = nn.ReLU()
        #self.dropout3 = nn.Dropout(0.1)
        # Output layer
        #self.out = nn.Linear(56, 1)      
        #self.sigmoid = nn.Sigmoid()

        # pp Bs Optuna Model 
        # 1st hidden layer
        self.fc1 = nn.Linear(input_size, 248)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.105)
        # 2nd hidden layer
        self.fc2 = nn.Linear(248, 97)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.105)
        # 3rd hidden layer
        self.fc3 = nn.Linear(97, 21)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.105)
        # Output layer
        self.out = nn.Linear(21, 1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.sigmoid(self.out(x))
        return x


# === LOSS FUNCTION === #
class BalancedLoss(nn.Module):
    """
    Implements balanced binary cross-entropy loss using class weights.
    """
    def __init__(self, alpha=None):
        super(BalancedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Calculate the standard binary cross-entropy loss without reduction
        CE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

        if self.alpha is not None:
            #Selects the weight based on the target value (1 or 0)
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            # apply per-sample weighting.
            CE_loss *= alpha_t
        
        # return the mean of the balanced cross-entropy loss
        return torch.mean(CE_loss)


# === EARLY STOPPING === #
class EarlyStopping:
    """
    Stops training if validation loss does not improve for a set number of epochs.
    Saves the best model's weights.
    """
    def __init__(self, patience, delta):
        self.patience = patience  # Number of epochs to wait for improvement in validation loss, before stopping the training
        self.delta = delta  # Minimum change in validation loss that qualifies as an improvement
        self.best_score = None  # Best validation score encountered during training
        self.early_stop = False  # Boolean flag that indicates if training should be stopped early
        self.counter = 0  # Counts the number of epochs since the last improvement in validation loss
        self.best_model_state = None  # Stores the state of the model when the best validation loss was observed

    def __call__(self, val_loss, model):
        """Allows the instance to be called like a function during training"""
        # convert validation loss into a score (lower loss = higher score)
        score = -val_loss

        # on first call, set the initial best score and save model weight
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        
        # if current score not significantly better than best, increment the counter
        elif score < self.best_score - self.delta:
            self.counter += 1
            # if counter exceeds patience stop training
            if self.counter >= self.patience:
                self.early_stop = True

        # if there is a significant improvement, save new best model state and reset counter
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        """Call after training ends to restore model to its best version (based on validation loss)."""
        model.load_state_dict(self.best_model_state)

# === TRAIN STRUCTURE === #
def regul(val_loader, model, criterion, epoch, num_epochs, early_stopping):
    """
    Evaluates model on the validation set at each epoch.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): Model to be evaluated.
        criterion (Loss): Loss function (e.g., BCE, FocalLoss).
        epoch (int): Current epoch index.
        num_epochs (int): Total number of training epochs.
        early_stopping (EarlyStopping): Early stopping handler.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    val_loss = 0.0

    # Compute validation loss for 1 epoch
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs).squeeze()
            loss = criterion(val_outputs, val_targets)
            val_loss += loss.item() * val_inputs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Check if validation loss has reached its minimum
    early_stopping(val_loss, model)

    return val_loss


def train_model(model, early_stopping, train_loader, val_loader, criterion, optimizer, 
                num_epochs=1000, return_losses=False):
    """
    Trains the model and plots training vs validation loss with early stopping.

    Args:
        model (torch.nn.Module): Neural network to train.
        early_stopping (EarlyStopping): Early stopping handler.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer for model weights.
        num_epochs (int, optional): Max number of epochs. Defaults to 1000.
        flag (int, optional): Plot type flag for output naming. Defaults to 0.

    Returns:
        None
    """
    plt.switch_backend("Agg")
    
    tl_vector, vl_vector = [], []
    idx = num_epochs - 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze() # Adjust outputs to match the shape of targets
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss = regul(val_loader, model, criterion, epoch, num_epochs, early_stopping)

        # Save training and validation loss in vectors
        tl_vector.append(train_loss)
        vl_vector.append(val_loss)

        # Save best epoch number
        if early_stopping.early_stop:
            idx = epoch - early_stopping.patience
            print(f"Early stopping at epoch {idx}\n Lowest loss: {-early_stopping.best_score}")
            break

    # Load the best model
    early_stopping.load_best_model(model)

    # If Optuna is running, skip plotting and return loss curves
    if return_losses:
        return tl_vector, vl_vector, idx

    # Otherwise, plot immediately
    last_epoch = epoch + 1
    indices = range(1, last_epoch + 1) 

    plt.figure()
    plt.plot(indices, tl_vector[:last_epoch], label="Training", color="navy", markersize=1)
    plt.plot(indices, vl_vector[:last_epoch], label="Validation", color="orange", markersize=1)
    plt.scatter(idx + 1, vl_vector[idx], color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # Dynamic y-axis focus around the plateau
    min_loss = min(min(tl_vector[:last_epoch]), min(vl_vector[:last_epoch]))
    max_loss = max(min(tl_vector[:last_epoch]), min(vl_vector[:last_epoch]))
    plt.ylim(min_loss * 0.95, max_loss * 1.2)  

    plt.savefig("B_loss.pdf")
    plt.close() 


# === MAIN === #

def main():
    try:
        #largest set 
        #variables = ["Bchi2cl", "Bcos_dtheta", "Bdtheta", "Bnorm_svpvDistance_2D",
        #"Bpt", "Btktkpt", "Btrk1Pt","Btrk2Pt", "Btrk1dR", "Btrk2dR", "BtrkPtimb"] #Two Tracks Particles
        #variables = [] # B+ Single track

        #shap chosen set
        #variables =['Bdtheta', 'Bnorm_svpvDistance_2D', 'Btrk2dR', 'Bpt', 'Btktkpt', 'Bchi2cl'] #pp Bs
        variables =['Bnorm_svpvDistance_2D', 'Bpt', 'Bdtheta', 'Btrk1dR', 'Btrk1Pt'] #pp Bu 
        

        dataset = ROOTDataset("ROOT_files/MC_pp_Bu_signal.root",
                            "ROOT_files/Data_pp_Bu_sidebands.root",
                            variables,
                            max_events=None)   

        # Train/val/test split
        N = len(dataset)
        train_size = int(0.5 * N)
        val_size   = int(0.25 * N)
        test_size  = N - train_size - val_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        train_indices = train_set.indices
        val_indices   = val_set.indices
        test_indices  = test_set.indices

        # DataLoaders
        train_loader = DataLoader(train_set, batch_size=16384, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_set,   batch_size=16384, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_set,  batch_size=16384, shuffle=False, num_workers=4)

        input_size = len(variables)

        # Calculate class weights
        n_signal = dataset.signal_entries
        n_background = dataset.background_entries

        class_weights = torch.tensor([1 / n_background, 1 / n_signal], dtype=torch.float32)
        class_weights /= class_weights.sum()

        # Directory to save models
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialise model
        B_model = ClassificationModel(input_size)

        # Define loss function and optimizer
        B_criterion = BalancedLoss(alpha=class_weights)
        B_optimizer = optim.Adam(B_model.parameters(), lr=0.0017672687518874427, weight_decay=1.8265594494791445e-06)

        # Early stopping
        B_early_stopping = EarlyStopping(patience=85, delta=1e-6)

        # Train model
        print("\nTraining model with Balanced Loss...")
        tl_vector, vl_vector, best_epoch = train_model(B_model, B_early_stopping, train_loader, val_loader, 
                                                    B_criterion, B_optimizer, num_epochs=500, return_losses=True)

        # Save model
        torch.save({
            "model_state_dict": B_model.state_dict(),
            "optimizer_state_dict": B_optimizer.state_dict(),
            "train_loss_curve": tl_vector,
            "val_loss_curve": vl_vector,
            "best_epoch": best_epoch,
            "variables": variables,
            "signal_file": "ROOT_files/MC_pp_Bu_signal.root",
            "background_file": "ROOT_files/Data_pp_Bu_sidebands.root",
            "split_sizes": [train_size, val_size, test_size],
            "split_indices": {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices}
            }, os.path.join(checkpoint_dir, "optuna_pp_Bu_model_checkpoint_alt.pth"))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
