#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=128G
#SBATCH --time=00-00:30
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --job-name=test_mistral_nemo
#SBATCH -o ./logs/slurm/output_potsdam_%j.out

echo "Loading modules..."
module purge
module load tools/virtualenv/20.24.6-GCCcore-13.2.0
module load system/CUDA/11.6.0

echo "Setting folders..."
export HF_HOME=/work/leonrios/.cache/huggingface
export VLLM_CACHE_ROOT=/work/leonrios/.cache/vllm
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Activating virtual environment..."
source ./venv/mistral_nemo/bin/activate

echo "Running python script..."
python3 -u ./src/extract_pipeline/mistral_nemo.py

echo "Script finished!"

deactivate
module purge
