#!/bin/bash 
#SBATCH -p lipq 
#SBATCH --job-name=run1_codepy
#SBATCH --output=slurm-%j.out 
#SBATCH --error=slurm-%j.err  
#SBATCH --cpus-per-task=24 
set -euo pipefail 

module purge 
module load root

root -l data_fit_Bs.C
