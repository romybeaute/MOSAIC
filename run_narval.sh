#!/bin/bash   #bash shell script for executing the grid search on a high-performance computing cluster


# SLURM job configs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1ß
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=MOSAIC_GS
#SBATCH --mail-type=ALL

# load required modules and activate Python environment
module load python/3.10 StdEnv/2023 # load Python 3.10 amnd the standard environment modules
source ~/.mosavenv/bin/activate # activate the virtual environment named mosavenv

python grid_search.py --condition HS --sentences --reduced_GS
