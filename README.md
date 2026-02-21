# PyTorch Neural Network 

First of all, thanks to Gabriela Sousa and Teresa Escária, for setting the base for Neural Network codes and workflow, also they wrote an excellent instruction as well.
This work is to extend the codes/workflow and include the data standardization which was missing previously, also to with ability of incorporate analysis of more particules.

---

## Instruction for first time setting up:

### 1. Create and Activate a Virtual Environment in your terminal.
```bash
# Load the python module 
module load python
# Create a virtual environment named myenv in your current directory (you could choose another name)   
python -m venv myenv
# Activation
source myenv/bin/activate
# Deactivation
deactivate
```
### 2. Installing all the required Packages from requirements.txt in the Virtual Environment

-While in the Virtual Environment, install the required Packages.
-It might need more packages than those in list.

```bash
pip install -r requirements.txt
```

---

## Analysis Workflow:

### 1. Load the ROOT module 

```bash
module load root
```

### 2. Fitting your MC signal and choosing the signal region. Run "mc_double_gaussian_fit.C" (Contributed by Ricardo Ribeiro).

-Make necessary change to fit, like Double/Triple Gaussian.

```bash
root -l mc_double_gaussian_fit.C 
```

-Registe the fitting parameters for later data fitting.
-Define the signal region (perhaps with x times of effective sigma).

### 3. Now with the signal region defined, produce your MC signal root (for ML learning) and Data sidebands root.

-Make necessary change to "MLprep.C".
-Insert your signal region value (upper and lower limits).
-Please change the Input Tree as accordingly to your root files.
-Please change the Output Tree name as following:
- Tsignal <-> MC Signal
- Tback <-> Data Bkg Sidebands 
Also add all the variables you need for the full analysis (not every variable in the root would be analyzed as you can select in the future).
Please do not include the variable MLscore until you needed it for producing the Data with MLcut.

Depending if you wanted to use all the data points from sidebands or some % of it.
You could use "MLprep2.C" and calculating the shrinking factor for it (later needed for FOM Optimization, for multiplying the fb factor by inverse of shrinking factor).
With this you do not lose out mass range and have precise control of numbers of data points.
Please do not change the seed, for replication purpose.

The naming of root is to your preference.

```bash
root -l MLprep.C 
```

### 4. Plotting the distributions of Data sidebands and MC signal. Make precut so to get scaling factors of fs and fb.

-This code is not included here. But basically is to plot the distributions of Data sidebands and MC signal and compare. 
-Then select a precut (by eyes) that can eliminate most of background.
-And then fit it using "data_fit_Bplus_erfc_ML.C" or "data_fit_Bs.C" (if it's B+ or Bs). For other particles, you need to write a fitting ROOTFIT script for them.
-Get the scaling factors fs and fb for later FOM optimization (Significance).

### 5. Run "correlation_MLprep.py"

-Run it to get the necessary pearson correlation coefficients for later Feature Importance study (using SHAP).

```bash
python correlation_MLprep.py 
```

### 6. Run "NN.py"

-Make necessary change (like variables for analysis and SHAP selected variables).
-Choose a simple Neural Network baseline architecture for first time analysis. I suggest to just use the first one in the code but it's up to your preference.
-Please change the MC signal and Data sidebands roots directories to the one you are analyzing.
-Choose a name for the baseline moodel and later a name for Optuna optimized model. 

### 7. Run "cumu_shap_groups.py"







