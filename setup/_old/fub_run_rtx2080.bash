#!/bin/bash
#SBATCH --job-name=run_rtx2080
#SBATCH --output=./logs/slurm/output_fub_rtx2080_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --mem=9G

#SBATCH --time=01:00:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/home/nicolasal97/.cache/huggingface
export VLLM_CACHE_ROOT=/home/nicolasal97/.cache/vllm

echo "Activating virtual environment..."
source ./venv/extract_csv/bin/activate

echo "Running python script..."
python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2018 -tn RTX2080 -aram 9 -ngpus 1

deactivate
module purge