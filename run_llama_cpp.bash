#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --account=leonnial
#SBATCH -o ./logs/slurm/output_llama_cpp_%j.out

echo "Loading modules..."
module load nvidia_hpc_sdk/nvhpc/25.1
module load virtualenv
export HF_HOME=/lustre/romanistik/leonnial/.cache/huggingface
export VLLM_CACHE_ROOT=/lustre/romanistik/leonnial/.cache/vllm
export CC=/software/eb/GCCcore/13.2.0/bin/gcc
export CXX=/software/eb/GCCcore/13.2.0/bin/g++

echo "Running test..."
source ./venv/llama_cpp/bin/activate
python -u ./src/extract_pipeline/llama.py

echo "Script finished!"
