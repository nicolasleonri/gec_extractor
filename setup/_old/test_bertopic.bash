#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=96gb
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --account=leonnial
#SBATCH --time=14-00:00:00
#SBATCH --job-name=bertopic
#SBATCH --cpus-per-task=8
#SBATCH --array=0
#SBATCH -o ./logs/slurm/output_%A_%a.out

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
export PATH=/software/nvidia/hpc_sdk/nvhpc_2025_251_Linux_x86_64_cuda_12.6/Linux_x86_64/25.1/compilers/bin:$PATH
export CUDA_HOME=/software/nvidia/hpc_sdk/nvhpc_2025_251_Linux_x86_64_cuda_12.6/Linux_x86_64/25.1/compilers

echo "Activating virtual environment..."
source ./venv/bertopic/bin/activate

echo "First task: BerTopic. Running..."
python -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/correo/ -n correo

deactivate

echo "Script finished!"
