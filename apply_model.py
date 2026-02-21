"""
Module to apply trained model to full dataset.
Teresa 25/07/2025
"""
import sys
import os
import uproot
import numpy as np
import torch
import json
from torch.utils.data import Dataset

from NN import ClassificationModel

def standardize_X(X, mean, std):
    mean = torch.tensor(mean, dtype=torch.float32)
    std  = torch.tensor(std,  dtype=torch.float32)
    return (X - mean) / std


def prepdata_for_application(file_path, variables, tree_name="Tdata"):
    """
    Load features from a ROOT tree for applying trained model.

    Parameters:
            Path to the ROOT file with full dataset.
        tree_name : str
            Name of the TTree inside the ROOT file.
        variables:
            variables used during training.

    Returns:
        X : torch.Tensor
            Feature matrix ready for model inference.
    """
    file = uproot.open(file_path)
    tree = file[tree_name]

    arrays = tree.arrays(variables, library="np")
    X = np.column_stack([arrays[var] for var in variables])
    X = torch.from_numpy(X.astype(np.float32))

    return X


def model_application(file_path, checkpoint_path, variables):
    """
    Apply trained model on full dataset.

    Parameters:
        file_path : str
            Path to ROOT file with data.
        checkpoint_path : str
            Path to trained model checkpoint

    Returns:
        np.ndarray: Predicted labels (0 or 1).
        np.ndarray: Model output probabilities.
    """

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    X = prepdata_for_application(file_path, variables)

    # ---- standardize if scaler exists in checkpoint ----
    if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
        X = standardize_X(X, checkpoint["scaler_mean"], checkpoint["scaler_std"])
    else:
        print("[WARN] No scaler_mean/scaler_std found in checkpoint. "
              "Assuming model was trained on raw (non-standardized) variables.")

    input_size = X.shape[1]
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        outputs = model(X)
        probabilities = outputs.squeeze().numpy()
    
    return probabilities


def save_outputs(input_file, checkpoint_path, variables, tree_name='Tdata'):
    """
    Applies trained model to dataset and saves the ML output scores and thresholds
    as new branches to the existing ROOT file.

    Parameters:
        input_file : str
            Path to input ROOT file.
        tree_name : str
            Name of the TTree in the ROOT file.
    """
    
    # Open original ROOT file and read all branches into numpy arrays
    file = uproot.open(input_file)
    tree = file[tree_name]
    arrays = tree.arrays(library="np")
    num_entries = len(arrays["Bmass"])
    
    if "eventN" in arrays:
        arrays["eventN"] = arrays["eventN"].astype(np.int64)

    # Load data and apply model
    probabilities = model_application(input_file, checkpoint_path, variables)

    # Only add the MLscore branch
    arrays["MLscore"] = probabilities.astype(np.float32)

    # Create new file
    output_file = input_file.replace(".root", f"_ml_output_expQ.root")
    with uproot.recreate(output_file) as new_file:
        # Write tree with all original dtypes preserved
        branch_types = {key: val.dtype for key, val in arrays.items()}
        new_file.mktree(tree_name, branch_types)
        new_file[tree_name].extend(arrays)


def main():
 
    checkpoint_path = f"checkpoints/shap_pp_Bs_model_checkpoint_expQ.pth"
    input_data = f"ROOT_files/Data_pp_Bs_selected.root"
    input_mc = f"ROOT_files/MC_pp_Bs_selected.root"

    variables = ['BQvalue', 'Btktkpt', 'Bdtheta', 'Bnorm_svpvDistance_2D', 'Btktkmass'] #B0s expQ
    #variables = ['Bnorm_svpvDistance_2D', 'Bnorm_trk1Dxy', 'Bdtheta', 'Btktkmass', 'Bpt', 'Btktkpt', 'Btrk2dR', 'Bchi2cl'] #B0s sd v2   
    #variables = ['Bnorm_svpvDistance_2D', 'Bpt', 'Bdtheta', 'Btrk1dR', 'Btrk1Pt'] #pp B+
    #variables = ['Bchi2cl','Bnorm_svpvDistance_2D', 'Bpt', 'Bcos_dtheta', 'Btrk1dR', 'Btrk1Pt'] #pp B+ 
    #variables = ['Bdtheta', 'Bnorm_svpvDistance_2D', 'Btrk1dR', 'Btktkmass', 'Bpt', 'Btrk2Pt', 'Bchi2cl'] #pp Bs
    #variables = ["Bchi2cl", "Bcos_dtheta", "Bdtheta", "Bnorm_svpvDistance_2D",
    #    "Bpt", "Btktkpt", "Btrk1Pt","Btrk2Pt", "Btrk1dR", "Btrk2dR", "BtrkPtimb"]    
    #variables = ['Bpt', 'Btrk1dR', 'Bnorm_svpvDistance_2D', 'Bchi2cl', 'Btrk1Pt', 'Bcos_dtheta']
    save_outputs(input_data, checkpoint_path, variables)
    save_outputs(input_mc, checkpoint_path, variables)

if __name__ == '__main__':
    main()
