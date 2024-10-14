#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=CompNeuroPhenos
#SBATCH --mail-type=ALL

module load python/3.10 StdEnv/2023
source ~/PY310/bin/activate

python grid_search_colyra.py --condition HS --reduced_GS
#python grid_search_colyra.py --condition DL --reduced_GS
#python grid_search_colyra.py --condition HS
#python grid_search_colyra.py --condition DL
#python grid_search_colyra.py --condition HS --sentences
#python grid_search_colyra.py --condition DL --sentences
