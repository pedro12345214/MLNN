import sys
import os
import torch
import uproot
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from fpdf import FPDF
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sns.set_style("darkgrid")

from NN import ROOTDataset, ClassificationModel

#def load_model_save_params(checkpoint_path):
#
#    if not os.path.exists(checkpoint_path):
#        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
#    # load checkpoint
#    #checkpoint_path = f"checkpoints/shap_pp_Bs_model_checkpoint.pth"
#    checkpoint = torch.load(checkpoint_path, weights_only=False)
#
#    full_dataset = checkpoint["dataset"]
#    test_dataset = checkpoint["test_set"]
#    test_dataset.y = test_dataset.y.long()
#
#    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
#
#    hyperparams = checkpoint["trial_params"]
#    neurons = [hyperparams[f"neurons_l{i}"] for i in range(hyperparams["n_layers"])]
#    input_size = full_dataset.X.shape[1]
#
#    model = DynamicClassificationModel(
#        input_size,
#        hyperparams["n_layers"],
#        neurons,
#        hyperparams["dropout_rate"]
#    )
#    model.load_state_dict(checkpoint['model_state_dict'])
#    model.eval()

    # Save hyperparameters to PDF
#    pdf = FPDF()
#    pdf.add_page()
#    pdf.set_font("Arial", size=12)
    
#    pdf.cell(200, 10, txt=f"\n Balanced model:", ln=True)
#    pdf.cell(200, 10, txt=f'Learning rate -> {hyperparams["learning_rate"]}', ln=True)
#    pdf.cell(200, 10, txt=f'Number of layers -> {hyperparams["n_layers"]}', ln=True)
#    pdf.cell(200, 10, txt=f'Dropout rate -> {hyperparams["dropout_rate"]}', ln=True)
#    pdf.cell(200, 10, txt=f'Weight decay -> {hyperparams["weight_decay"]}', ln=True)
#    pdf.cell(200, 10, txt=f'Batch size -> {hyperparams["batch_size"]}', ln=True)
#    pdf.cell(200, 10, txt=f'Number of layers -> {hyperparams["n_layers"]}', ln=True)
#    for i, n in enumerate(neurons):
#        pdf.cell(200, 10, txt=f'Neurons in layer {i} -> {n}', ln=True)

#    pdf_path = os.path.join(out_dir, "pp_Bs_Hyperparameters.pdf")
#    pdf.output(pdf_path)
#    print(f"Hyperparameters PDF saved to {pdf_path}")

    
#    return model, test_loader

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    variables = checkpoint["variables"]

    # Load full dataset
    dataset = ROOTDataset(checkpoint["signal_file"],
                          checkpoint["background_file"],
                          variables)

    # Split using the saved sizes
    split_indices = checkpoint["split_indices"]
    test_set = Subset(dataset, split_indices["test"])
    test_loader = DataLoader(test_set, batch_size=4096, shuffle=False)
    
    # Load model
    input_size = len(variables)
    model = ClassificationModel(input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, test_loader


# --- Get raw model outputs (sigmoid applied) ---
def get_targets_probabilities(model, test_loader, eps=1e-6):
    targets, probabilities = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            probabilities.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return np.array(targets), np.array(probabilities)


# --- Plot probability histogram ---
def plot_histogram(targets, probabilities, out_dir=".", best_thr=0.5):

    plt.figure(figsize=(8, 6))

    # Signal predictions
    signal_predict = probabilities[targets == 1]
    plt.hist(signal_predict, bins=40, density=True, alpha=0.9, label="Signal (MC)", color="blue", range=(0.0, 1.0))

    # Background predictions
    background_predict = probabilities[targets == 0]
    plt.hist(background_predict, bins=40, density=True, alpha=0.5, label="Background (Data)", color="red", hatch="//", edgecolor="black", range=(0.0, 1.0))
    
    plt.axvline(x=best_thr, color='grey', linestyle="--", lw=2, label=f'Threshold = {best_thr:.2f}')
    plt.xlabel("Predicted Probability", fontsize=14, labelpad=15)
    plt.ylabel("Normalized Density (log)", fontsize=14, labelpad=15) 
    plt.yscale("log") 
    plt.legend()
    save_path = os.path.join(out_dir, "pp_Bs_prob_distribution.pdf")
    plt.savefig(save_path)  # Save the plot as a PDF file
    plt.close()

# --- Plot ROC curve ---
def plot_roc_curve(targets, probabilities, out_dir=".", best_point=None):

    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    if best_point != None:
        plt.scatter(best_point[0], best_point[1], color="black", label="Best Threshold")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    save_path = os.path.join(out_dir, "pp_Bs_roc_curve.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")


# --- Save metrics to PDF ---
def save_metrics_pdf(targets, probabilities, out_dir=".", best_thr=0.5):
    """
    Save metrics PDF with a confusion matrix scaled to the full dataset.
    """
    # For binary classification, using 0.5 threshold
    pred_labels = (probabilities >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(targets, pred_labels)
    prec = precision_score(targets, pred_labels)
    rec = recall_score(targets, pred_labels)
    f1 = f1_score(targets, pred_labels)
    conf_matrix = confusion_matrix(targets, pred_labels)

    # PDF output
    pdf_filename = os.path.join(out_dir, f"pp_Bs_Metrics.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 50, f"Evaluation Metrics")

    # Metrics text
    c.setFont("Helvetica", 12)
    y_pos = height - 100
    for metric_name, value in [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1-score", f1), ("Best Threshold", best_thr)]:
        c.drawString(100, y_pos, f"{metric_name}: {value:.4f}")
        y_pos -= 20

    # Confusion Matrix Table
    c.drawString(100, y_pos - 20, "Confusion Matrix:")
    matrix_top = y_pos - 80
    matrix_left = 220
    cell_width = 100
    cell_height = 40
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)

    for i in range(2):
        for j in range(2):
            x = matrix_left + j * cell_width
            y = matrix_top - i * cell_height
            c.rect(x, y, cell_width, cell_height)
            c.drawCentredString(x + cell_width / 2, y + cell_height / 2 - 6, str(int(conf_matrix[i, j])))

    # Labels
    c.setFont("Helvetica-Bold", 12)
    c.drawString(matrix_left + cell_width / 2 - 20, matrix_top + 50, "True: 1")
    c.drawString(matrix_left + 3 * cell_width / 2 - 20, matrix_top + 50, "True: 0")
    c.drawString(matrix_left - 60, matrix_top - cell_height / 2 + 35, "Pred: 1")
    c.drawString(matrix_left - 60, matrix_top - 3 * cell_height / 2 + 35, "Pred: 0")

    c.save()
    print(f"Metrics PDF saved to {pdf_filename}")


def plot_loss_curve(checkpoint_path, out_dir="."):

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    train_loss_curve = checkpoint["train_loss_curve"]
    val_loss_curve = checkpoint["val_loss_curve"]
    best_epoch = checkpoint["best_epoch"]

    indices = range(1, len(train_loss_curve) + 1)
    plt.figure()
    plt.plot(indices, train_loss_curve, label="Training", color="navy", markersize=1)
    plt.plot(indices, val_loss_curve, label="Validation", color="orange", markersize=1)
    plt.scatter(best_epoch + 1, val_loss_curve[best_epoch], color="black", label="Early Stop", s=64)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"pp_Bu_loss_curve.pdf"))
    plt.close()


def save_scalings(data_file, mc_file, output_file=None):
    """
    Load fit parameter JSONs for data and MC, compute scaling factors,
    and update the data JSON with the new values.
    """
    # load params 
    with open(data_file, "r") as f:
        data_params = json.load(f)

    with open(mc_file, "r") as f:
        mc_params = json.load(f)

    # extract values
    nsig_data, _ = data_params["nsig"]
    nsig_mc,     = data_params["nsig_mc"]
    #nsig_mc_RT, _   = mc_params["RT"]["nsig"] 
    #nsig_mc_WT, _ = mc_params["WT"]["nsig"]
    #nsig_mc = nsig_mc_RT + nsig_mc_WT

    nbkg_SR_data = data_params["nbkg_SR"]
    nbkg_SB_data = data_params["nbkg_SB"]

    # calculate scaling factors 
    s_scale = nsig_data / nsig_mc
    b_scale = nbkg_SR_data / nbkg_SB_data
    
    data_params.update({
        "s_scale": s_scale,
        "b_scale": b_scale,
    })

    # save back to file (default overwrite data_file)
    if output_file is None:
        output_file = data_file

    with open(output_file, "w") as fout:
        json.dump(data_params, fout, indent=4)

    print(f"Updated params written to {output_file}")

    return s_scale, b_scale


def calculate_fom(targets, probabilities, s_scale, b_scale, output_dir="."):
    """
    Calculate the ROC curve points and find the best threshold maximizing the figure of merit (FOM).
    """
    thresholds = np.linspace(0.0, 1.0, 500)
    best_thr = 0.5
    best_fom = -np.inf
    best_point = None
    tpr_list = []
    fpr_list = []
    fom_values = []

    total_signal = ((targets == 1) * s_scale).sum().item()
    total_background = ((targets == 0) * b_scale).sum().item()

    if total_signal == 0 or total_background == 0:
        raise ValueError("Targets must contain both signal (1) and background (0) examples.")

    for thr in thresholds:
        predicted = (probabilities >= thr).astype(float)
        tp = (((predicted == 1) & (targets == 1)).astype(float)).sum().item() * s_scale
        fp = (((predicted == 1) & (targets == 0)).astype(float)).sum().item() * b_scale

        fom = tp / np.sqrt(tp + fp) if (tp + fp) > 0 else 0
        tpr = tp / total_signal
        fpr = fp / total_background

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        fom_values.append(fom)

        if fom > best_fom:
            best_fom = fom
            best_thr = thr
            best_point = (fpr, tpr)

    # Plot FoM vs Threshold
    plt.figure()
    plt.plot(thresholds, fom_values, label="FoM")
    plt.axvline(x=best_thr, color='black', linestyle='--', label=f"Best Threshold = {best_thr:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("FoM values")
    plt.legend()
    save_path = os.path.join(output_dir, "pp_Bs_FoM_vs_Thresholds.pdf")
    plt.savefig(save_path)
    plt.close()

    print(f"Best threshold is: {best_thr}")
    return best_thr, best_point


def plot_combined_roc(targets, probabilities, output_dir="."):

    # Model with complete set of variables
    large_checkpoint = "checkpoints/Baseline_pp_Bs_model_checkpoint2.pth"
    large_model, large_test_loader = load_model(large_checkpoint)
    large_targets, large_probs = get_targets_probabilities(large_model, large_test_loader)

    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    fpr_large, tpr_large, _ = roc_curve(large_targets, large_probs)
    roc_auc_large = auc(fpr_large, tpr_large)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Selected Set (AUC = {roc_auc:.4f})')
    plt.plot(fpr_large, tpr_large, color='blue', lw=2,
             label=f'Complete Set (AUC = {roc_auc_large:.4f})')
    plt.scatter(best_point[0], best_point[1], color="black", label="Best Threshold")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.xlim(0, 0.2)
    plt.ylim(0.7, 1)
    save_path = os.path.join(output_dir, "baseline_pp_Bs_combined_roc_curve.pdf")
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")



if __name__ == "__main__":
   
    checkpoint_path = f"checkpoints/optuna_pp_Bs_model_checkpoint2.pth"
    output_dir = f"EvaluationStats"
    os.makedirs(output_dir, exist_ok=True)

    # Load model and test data
    #model, test_loader = load_model_save_params(checkpoint_path)
    model, test_loader = load_model(checkpoint_path)

    # Get model outputs 
    targets, probabilities = get_targets_probabilities(model, test_loader)

    # Get signal and background scalings
    s_scale, b_scale = 0.552, 0.182

    #s_scale, b_scale = save_scalings("scalings/fit_params_data_RTWT.json", "scalings/fit_params_mc_RTWT.json")

    # Maximise FoM 
    best_thr, best_point = calculate_fom(targets, probabilities, s_scale, b_scale, output_dir)
    
    plot_histogram(targets, probabilities, output_dir, best_thr)
    plot_roc_curve(targets, probabilities, output_dir, best_point)
    save_metrics_pdf(targets, probabilities, output_dir, best_thr)
    plot_loss_curve(checkpoint_path, output_dir)
    plot_combined_roc(targets, probabilities, output_dir)

    
