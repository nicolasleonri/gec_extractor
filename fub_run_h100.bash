#!/bin/bash
#SBATCH --job-name=run_h100
#SBATCH --output=./logs/slurm/output_fub_h100_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=hiprio

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=90G

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
python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/test -tn H100 -aram 80 -ngpus 1

deactivate
module purge