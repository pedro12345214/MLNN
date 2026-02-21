"""
Optuna hyperparameter optimization for your NN (Sigmoid + BCE).
Keeps your ROOTDataset, BalancedLoss, EarlyStopping logic.

Run:
  python -u NN_optuna.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from NN import StandardizedSubset
import uproot

# ----------------------------
# Your dataset (same as yours)
# ----------------------------
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
        # labels as float32 for BCE
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)

# ----------------------------
# Your loss (same as yours)
# ----------------------------
class BalancedLoss(nn.Module):
    def __init__(self, alpha=None):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            ce = ce * alpha_t
        return torch.mean(ce)

# ----------------------------
# Early stopping (same as yours)
# ----------------------------
class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

# ----------------------------
# Dynamic model (Sigmoid output)
# ----------------------------
class DynamicClassificationModel(nn.Module):
    """
    Tunable MLP: variable number of layers/neurons with dropout.
    Output: sigmoid probability (matches your BCE usage).
    """
    def __init__(self, input_size, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

# ----------------------------
# Training/eval
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        p = model(x)
        loss = criterion(p, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

@torch.no_grad()
def eval_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        p = model(x)
        loss = criterion(p, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

def run_training_trial(
    trial,
    dataset,
    train_idx,
    val_idx,
    input_size,
    class_weights,
    device,
    mean,
    std,
    max_epochs=200
):
    # ---- Suggest hyperparameters ----
    n_layers = trial.suggest_int("n_layers", 2, 3)
    hidden_sizes = [
        trial.suggest_int(f"neurons_l{i}", 16, 256, log=True)
        for i in range(n_layers)
    ]
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.005)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [ 64,128, 256, 512])

    patience = trial.suggest_int("patience", 10, 100, step=5)
    delta = 1e-6

    # ---- Loaders (same split for every trial) ----
    pin = torch.cuda.is_available()
    num_workers = 0  # safe default on shared FS

    train_ds = StandardizedSubset(dataset, train_idx, mean.to(device="cpu"), std.to(device="cpu"))
    val_ds   = StandardizedSubset(dataset, val_idx,   mean.to(device="cpu"), std.to(device="cpu"))

    train_loader = DataLoader(
      train_ds,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=pin
    )
    val_loader = DataLoader(
      val_ds,
      batch_size=max(16384, batch_size),
      shuffle=False,
      num_workers=num_workers,
      pin_memory=pin
    )

    # ---- Model/loss/opt ----
    model = DynamicClassificationModel(input_size, hidden_sizes, dropout).to(device)
    criterion = BalancedLoss(alpha=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early = EarlyStopping(patience=patience, delta=delta)

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(max_epochs):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vloss = eval_loss(model, val_loader, criterion, device)

        # Report to Optuna (for pruning)
        trial.report(vloss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        early(vloss, model)
        if vloss < best_val:
            best_val = vloss
            best_epoch = epoch

        if early.early_stop:
            break

    early.load_best_model(model)

    # Save useful attrs
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("hidden_sizes", hidden_sizes)

    return best_val, model

# ----------------------------
# Main Optuna driver
# ----------------------------
def main():
    # ---- Files/vars (edit if needed) ----
    #variables = ['Bdtheta', 'Bnorm_svpvDistance_2D', 'Btrk1dR', 'Btktkmass', 'Bpt', 'Btrk2Pt', 'Bchi2cl'] #pp Bs after standardization
    variables = ['Bnorm_svpvDistance_2D', 'Bnorm_trk1Dxy', 'Bdtheta', 'Btktkmass', 'Bpt', 'Btktkpt', 'Btrk2dR', 'Bchi2cl'] #pp Bs after standardization NEW 
    #variables = ['Bpt', 'Btrk1dR', 'Bnorm_svpvDistance_2D', 'Bchi2cl', 'Btrk1Pt', 'Bcos_dtheta'] #pp B+ after standardization
    signal_file = "ROOT_files/MC_pp_Bs_signal.root"
    background_file = "ROOT_files/Data_pp_Bs_sidebands.root"

    dataset = ROOTDataset(signal_file, background_file, variables, max_events=None)
    input_size = len(variables)

    # ---- Class weights (same as your code) ----
    n_signal = dataset.signal_entries
    n_background = dataset.background_entries
    class_weights = torch.tensor([1.0 / n_background, 1.0 / n_signal], dtype=torch.float32)
    class_weights /= class_weights.sum()

    # ---- Fixed split for fairness across trials ----
    idx_all = np.arange(len(dataset))
    y_all = dataset.y  # numpy array

    train_idx, temp_idx = train_test_split(
        idx_all, test_size=0.50, random_state=42, stratify=y_all
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=y_all[temp_idx]
    )

    # ---- Compute scaler on TRAIN split only ----
    X_train = dataset.X[train_idx]  # numpy [Ntrain, n_features]
    mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
    std  = torch.tensor(X_train.std(axis=0), dtype=torch.float32)

    # Protect against zero variance features
    std[std == 0] = 1.0


    # train=50%, val=25%, test=25%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Where to save best model ----
    checkpoint_dir = "checkpoints_optuna"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "sd_new_pp_Bs_model_checkpoint.pth")

    # ---- Objective wrapper ----
    def objective(trial):
        best_val, model = run_training_trial(
            trial=trial,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            input_size=input_size,
            class_weights=class_weights,
            device=device,
            mean=mean,
            std=std,
            max_epochs=200
        )

        # Save best model checkpoint across all trials
        # Optuna will call objective many times; we only overwrite if this trial is best.
        trial.set_user_attr("val_loss", float(best_val))

        return best_val  # minimize validation loss

    # ---- Optuna study ----
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print("\nBest trial:")
    best = study.best_trial
    print("  best_val_loss:", best.value)
    print("  params:", best.params)
    print("  attrs:", best.user_attrs)

    # ---- Retrain best model once and save checkpoint ----
    # (so you have the weights, not just hyperparams)
    best_params = best.params

    # Create a dummy trial-like object? Easier: run_training_trial with fixed params by reusing the same code.
    # We'll just manually build the best model here:
    hidden_sizes = [best_params[f"neurons_l{i}"] for i in range(best_params["n_layers"])]
    dropout = best_params["dropout"]
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    batch_size = best_params["batch_size"]
    patience = best_params["patience"]

    pin = torch.cuda.is_available()

    train_ds = StandardizedSubset(dataset, train_idx, mean, std)
    val_ds   = StandardizedSubset(dataset, val_idx,   mean, std)
    test_ds  = StandardizedSubset(dataset, test_idx,  mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=max(16384, batch_size), shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=max(16384, batch_size), shuffle=False, num_workers=0, pin_memory=pin)

    model = DynamicClassificationModel(input_size, hidden_sizes, dropout).to(device)
    criterion = BalancedLoss(alpha=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early = EarlyStopping(patience=patience, delta=1e-6)

    best_epoch = -1
    best_val = float("inf")

    for epoch in range(400):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vloss = eval_loss(model, val_loader, criterion, device)
        early(vloss, model)
        if vloss < best_val:
            best_val = vloss
            best_epoch = epoch
        if early.early_stop:
            break

    early.load_best_model(model)

    # Save checkpoint like your style
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "variables": variables,
        "signal_file": signal_file,
        "background_file": background_file,
        "split_indices": {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()},
        "trial_params": best_params
    }, best_ckpt_path)

    # Save Optuna summary
    with open(os.path.join(checkpoint_dir, "optuna_study_best_sd_new.json"), "w") as f:
        json.dump({
            "best_value": best.value,
            "best_params": best.params,
            "best_user_attrs": best.user_attrs
        }, f, indent=4)

    print(f"\nSaved best model checkpoint to: {best_ckpt_path}")

if __name__ == "__main__":
    main()
