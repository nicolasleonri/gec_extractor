#!/bin/bash
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=2
#SBATCH --mem=96gb
#SBATCH --cpus-per-task=12 
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --account=leonnial
#SBATCH -o ./logs/slurm/output_extract_%j.out

echo "Loading modules..."
module purge
module load nvidia_hpc_sdk/nvhpc/25.1
module load virtualenv
module load cuda/12.6

echo "Setting folders..."
export HF_HOME=/lustre/romanistik/leonnial/.cache/huggingface
export VLLM_CACHE_ROOT=/lustre/romanistik/leonnial/.cache/vllm
export CC=/software/eb/GCCcore/13.2.0/bin/gcc
export CXX=/software/eb/GCCcore/13.2.0/bin/g++

echo "Activating virtual environment..."
source ./venv/extract_pipeline/bin/activate

echo "Running llama..."
python -u ./src/extract_pipeline/llama.py

echo "Running vllm..."
python -u ./src/extract_pipeline/pipeline_vllm.py

echo "Script finished!"
