import uproot
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


mc_path = "ROOT_files/MC_pp_Bs_signal.root"
data_path = "ROOT_files/Data_pp_Bs_sidebands.root"

out_dir = "Correlation_Matrices"
os.makedirs(out_dir, exist_ok=True)

# ---------------- ROOT file & tree loading ----------------
try:
    file_mc = uproot.open(mc_path)
    file_data = uproot.open(data_path)
except Exception as e:
    raise FileNotFoundError(f"Could not open ROOT file: {e}")

if "Tsignal" not in file_mc:
    raise RuntimeError("TTree 'Tsignal' not found in MC file.")
if "Tback" not in file_data:
    raise RuntimeError("TTree 'Tback' not found in data file.")

mcTree = file_mc["Tsignal"]
dataTree = file_data["Tback"]

# ---------------- Helpers ----------------
def build_dataframe(tree, variables):
    """
    Extracts branches listed in `variables` from a ROOT TTree into a pandas DataFrame.
    Missing branches are reported and filled with NaN.
    """
    available = set(tree.keys())
    missing = [var for var in variables if var not in available]
    if missing:
        print(f"[WARN] Missing branches in tree: {sorted(missing)}")
    # Only load available branches
    load_vars = [var for var in variables if var in available]
    df = tree.arrays(load_vars, library="pd")
    # Add missing columns as NaN
    for var in missing:
        df[var] = float('nan')
    # Ensure column order matches variables
    df = df.reindex(columns=variables)
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def save_correlation_artifacts(df, data_type):
    """
    Computes the Pearson correlation matrix, saves a PNG heatmap and a CSV.
    Filenames follow the convention expected by downstream scripts.
    """
    corr = df.corr(method='pearson')

    # Save CSV (this is what cumulative_shap_groups.py expects)
    csv_path = os.path.join(out_dir, f"{data_type}_pp_Bs_CorrelationMatrix.csv")
    #csv_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix_final.csv")
    corr.to_csv(csv_path)

    # Save PNG (optional visualization)
    plt.figure(figsize=(27, 25))
    #plt.figure(figsize=(16, 14))
    # annot=False to avoid huge labels; change to True if you really want per-cell text
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.xticks(rotation=45, ha="right")  # 45° inclined
    plt.title(f"{data_type} Bu Correlation Matrix")
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"{data_type}_pp_Bs_CorrelationMatrix.png")
    #png_path = os.path.join(out_dir, f"{data_type}_CorrelationMatrix_final.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[OK] Saved {data_type} CSV → {csv_path}")
    print(f"[OK] Saved {data_type} PNG → {png_path}")

# ---------------- Versions to generate ----------------
print(f"Generating correlation matrices ...")

# Load the variable list for this version
variables = ["Bchi2cl", "Bcos_dtheta", "Bdtheta", "Bnorm_svpvDistance_2D", "Bnorm_trk1Dxy", "Bnorm_trk2Dxy",
        "Bpt", "Btktkpt", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR",
        "BtrkPtimb", "By", "nSelectedChargedTracks", "Btktkmass", "BtktkvProb", "BQvalue"]

#variables = ['bLBSs', 'kstPt', 'IsoPtR_dr04_sum', 'kstTrkpDCABSs', 'kstTrkmDCABSs', 'bVtxCL', 'mumPt', 
#        'muLeadingPt', 'bDCABSs', 'mupPt', 'mupIsoPtR_dr04', 'kstTrkpDCABS', 'mumuPtR']

# Build DataFrames from the trees using the specified variable set
df_signal = build_dataframe(mcTree, variables)
df_bkg = build_dataframe(dataTree, variables)

# Ensure column order matches `variables` exactly (good for downstream alignment)
df_signal = df_signal.reindex(columns=variables)
df_bkg = df_bkg.reindex(columns=variables)

# Save correlation CSVs + PNGs with the expected naming scheme
save_correlation_artifacts(df_signal, "Signal")
save_correlation_artifacts(df_bkg, "Background")

print(f"Done. Correlation matrices generated")
