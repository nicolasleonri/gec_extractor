#!/bin/bash
#SBATCH --job-name=sent_fub_run_rtx2080ti
#SBATCH --output=./logs/slurm/output_fub_sent_rtx2080ti_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=4G

#SBATCH --time=00:15:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
# export HF_HOME=/home/nicolasal97/.cache/huggingface
export HF_HOME=/scratch/nicolasal97/.cache/huggingface

echo "Activating virtual environment..."
source venv/sentiment_analysis/bin/activate

echo "Running python script..."
python3 -u ./src/sentiment_analysis/sentiment_analyzer.py -f ./data/csv/sentiment_analysis

deactivate
module purge

echo "Script finished successfully"