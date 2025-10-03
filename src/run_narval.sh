#!/bin/bash   #bash shell script for executing the otuna search on a high-performance computing cluster


# SLURM job configs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1ß
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=MOSAIC_OS
#SBATCH --mail-type=ALL

# load required modules and activate Python environment
module load python/3.12.5 
source ~/.mosaicvenv/bin/activate # activate the virtual environment named mosavenv

python optuna_search.py --dataset dreamachine --condition DL --sentences --n_trials 100