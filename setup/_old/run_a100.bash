#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=96gb
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --account=leonnial
#SBATCH --time=14-00:00:00
#SBATCH --job-name=ocr_llm
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
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
source ./venv/extract_pipeline/bin/activate

FOLDERS=("./data/txt/trome"
"./data/txt/ojo"
"./data/txt/elcomercio"
"./data/txt/gestion"
"./data/txt/peru21"
)

NEWSPAPER=("trome"
"ojo"
"elcomercio"
"gestion"
"peru21")

FOLDER=${FOLDERS[$SLURM_ARRAY_TASK_ID]}
NEWSPAPER=${NEWSPAPER[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Processing folder: $FOLDER on GPU $CUDA_VISIBLE_DEVICES"

echo "Third task: LLM. Running..."
python -u ./src/extract_pipeline/pipeline_vllm.py -n "$NEWSPAPER" -i divided -f "$FOLDER"

deactivate

echo "Script finished!"
