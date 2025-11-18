#!/bin/bash
#SBATCH --job-name=multilabel_fub_run_h100
#SBATCH --output=./logs/slurm/output_fub_multilabel_h100_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=10G
#SBATCH --time=24:00:00 

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

echo "Activating virtual environment..."
source venv/multi_label/bin/activate

echo "Running python script..."

### FINAL RUNS! # Minimum specs: # Normal: 30min + 5GB # With CV: 45min + 6GB
python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 1200 -em macro_f1 -sm True -ta random_swap -aa True
python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 1200 -em macro_f1 -sm True -cv True -ta random_swap -aa True
python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 1200 -em macro_f1 -sm True -ht True -ta random_swap -aa True
python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 1200 -em macro_f1 -sm True -cv True -ht True -ta random_swap -aa True

### TO ANNOTATE!
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/results_pos_2025-11-09.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label/model_None_False_True_1200_macro_f1_2025-11-17 -lm True

deactivate
module purge

echo "Script finished successfully"