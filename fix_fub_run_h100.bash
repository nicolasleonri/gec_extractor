#!/bin/bash
#SBATCH --job-name=run_h100
#SBATCH --output=./logs/slurm/output_fub_h100_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=30G

#SBATCH --time=01-00:00:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/home/nicolasal97/.cache/huggingface

echo "Activating virtual environment..."
source ./venv/extract_csv/bin/activate

echo "Running python script..."
python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/ -tn H100 -aram 29 -ngpus 1
deactivate

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/home/nicolasal97/.cache/huggingface

echo "Activating virtual environment..."
source ./venv/bertopic/bin/activate

echo "Running python script..."
python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/trome/ -n trome -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models

deactivate
module purge

echo "Script finished successfully"

deactivate
module purge

echo "Script finished successfully"