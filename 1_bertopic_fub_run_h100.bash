#!/bin/bash
#SBATCH --job-name=bertopic_fub_run_h100
#SBATCH --output=./logs/slurm/output_fub_bertopic_h100_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G

#SBATCH --time=00-07:00:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/scratch/nicolasal97/.cache/huggingface

echo "Activating virtual environment..."
source ./venv/bertopic/bin/activate

echo "Running python script..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/elcomercio/ -n elcomercio -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models

# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/ojo/ -n ojo -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models

# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/peru21/ -n peru21 -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/correo/ -n correo -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models

# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/trome/ -n trome -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/gestion/ -n gestion -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models


deactivate
module purge

echo "Script finished successfully"