#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=96gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --account=leonnial
#SBATCH -o ./logs/slurm/output_final_%j.out

# echo "First task: Preprocessing images. Loading modules..."
# module purge
# module load OpenCV/4.12.0-gcc-14.2.0-python-3.13.1
# echo "First task: Preprocessing images. Activating virtual environment..."
# source ./venv/opencv/bin/activate
# echo "First task: Preprocessing images. Running..."
# python -u ./src/extract_pipeline/opencv.py -n trome -f ./data/images/2020
# deactivate

echo "Second/Third task: OCR. Loading modules..."
module purge
module load nvidia_hpc_sdk/nvhpc/25.1
module load virtualenv
module load cuda/12.6
echo "Second/Third task: OCR. Setting folders..."
export HF_HOME=/lustre/romanistik/leonnial/.cache/huggingface
export VLLM_CACHE_ROOT=/lustre/romanistik/leonnial/.cache/vllm
export CC=/software/eb/GCCcore/13.2.0/bin/gcc
export CXX=/software/eb/GCCcore/13.2.0/bin/g++
export PATH=/software/nvidia/hpc_sdk/nvhpc_2025_251_Linux_x86_64_cuda_12.6/Linux_x86_64/25.1/compilers/bin:$PATH
export CUDA_HOME=/software/nvidia/hpc_sdk/nvhpc_2025_251_Linux_x86_64_cuda_12.6/Linux_x86_64/25.1/compilers

echo "Second/Third task: OCR. Activating virtual environment..."
source ./venv/extract_pipeline/bin/activate

echo "Second task: OCR. Running..."
python -u ./src/extract_pipeline/llama.py -n trome -f ./results/images/preprocessed/trome

# echo "Third task: LLM. Running..."
# python -u ./src/extract_pipeline/pipeline_vllm.py -l ./logs/ocr_results

deactivate

echo "Script finished!"
