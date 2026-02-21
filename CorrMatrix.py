import uproot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#"/user/u/u25pedrochan/TMVA/TMVA_Sample/Bu_Rsideband/data_Rsideband_Bu_nochi2.root"
#"/user/u/u25pedrochan/TMVA/TMVA_Sample/Bu_2sideband/data_sidebands_Bu.root" //nochi2
# data_sidebands_Kstar.root
# data_unbinned_Bu_first.root First 2 Cuts
# MC_Bu.root
# MC_afterFirstCut_Bu.root
# MC_afterSecondCut_Bu.root
# data_Rsideband_Bu_afterChi.root
# data_Rsideband_Bu_afterChi_FirstCutted.root
# data_Rsideband_Bu_afterChi_SecondCutted.root
# "/lstore/cms/u25pedrochan/MLNN/ROOT_files/Data_pp_Bs_sidebands.root"
#"/lstore/cms/hlegoinha/DATA_Sharing/Bmesons/Data_Bs.root"
# === Step 1: Open ROOT file and get the TTree ===
file = uproot.open("/lstore/cms/u25pedrochan/MLNN/ROOT_files/Data_pp_Bs_sidebands.root")         # Replace with your actual file
tree = file["Tback"]              # Replace with actual TTree name

# === Step 2: Select branches (variables) for correlation ===
variables = [
       "Bmass", "BQvalue", "Bchi2cl", "Bcos_dtheta", "Bdtheta", "Bnorm_svpvDistance_2D", "Bnorm_trk1Dxy", "Bnorm_trk2Dxy",
        "Bpt", "Btktkmass", "Btktkpt", "Btrk1Pt", "Btrk2Pt", "Btrk1dR", "Btrk2dR",
        "BtrkPtimb", "Bujmass", "By", "nSelectedChargedTracks"  
    # Replace or extend with actual variable names from your tree
]

#B+
#variables = [
#       "Bmass","Bchi2cl", "Bcos_dtheta", "Bdtheta", "Bnorm_svpvDistance_2D",
#        "Bpt", "Btrk1Pt", "Btrk1dR",
#        "Bujmass", "By", "nSelectedChargedTracks"
#    # Replace or extend with actual variable names from your tree
#]


# === Step 3: Load data into a pandas DataFrame ===
df = tree.arrays(variables, library="pd")

# === Step 4: Compute and plot correlation matrix ===
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 6})
plt.xticks(rotation=45, ha="right")  # 45° inclined
plt.title("Correlation Matrix of Selected Variables Data sidebands Presel Cut Bs")
plt.tight_layout()
plt.savefig("CM_sidebands_pp_Presel_Bs.png", dpi=300)
plt.close()
