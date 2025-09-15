#!/bin/bash
#SBATCH --job-name=fub_run_h100
#SBATCH --output=./logs/slurm/output_fub_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=hiprio

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=384G

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

echoq "Running python script..."
python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/test -tn h100 -gbn 80 -aram 384 -ngpus 1

deactivate
module purge