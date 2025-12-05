#!/bin/bash
#SBATCH --job-name=pretrain_mlm_fub_run_h100
#SBATCH --output=./logs/slurm/output_fub_pretrain_mlm_h100_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=50G
#SBATCH --time=10:00:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NLTK_DATA=/home/nicolasal97/nltk_data
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# export PYTHONUNBUFFERED=1
# export MASTER_ADDR=localhost
# export MASTER_PORT=12345
# export CUDA_VISIBLE_DEVICES=0,1

echo "Activating virtual environment..."
source venv/multi_label/bin/activate

echo "Running python script..."
python3 -u ./src/multi_label/pretrain_mlm.py

# accelerate launch \
#     --multi_gpu \
#     --num_processes 2 \
#     --num_machines 1 \
#     --mixed_precision bf16 \
#     --dynamo_backend cudagraphs \
#     ./src/multi_label/pretrain_mlm.py

deactivate
module purge

echo "Script finished successfully"