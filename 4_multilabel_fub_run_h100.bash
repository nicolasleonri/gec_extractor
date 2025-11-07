#!/bin/bash
#SBATCH --job-name=multilabel_fub_run_h100
#SBATCH --output=./logs/slurm/output_fub_multilabel_h100_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=5G

#SBATCH --time=00:10:00

echo "Loading modules..."
module purge
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Setting folders..."
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Activating virtual environment..."
source venv/multi_label/bin/activate

echo "Running python script..."

# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -sm False -eo macro_f1
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 200 -sm False -eo macro_f1
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 300 -sm False -eo macro_f1
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 400 -sm False -eo macro_f1
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 500 -sm False -eo macro_f1
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -sm False -eo macro_f1
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 700 -sm False -eo macro_f1

# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -sm False
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 200 -sm False
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 300 -sm False
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 400 -sm False
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 500 -sm False
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -sm False
# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 700 -sm False

# python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -sm True -eo macro_f1
python3 -u ./src/multi_label/multi_label.py -f ./data/csv/multi_label/results_pos_2025-10-10.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -lm True

deactivate
module purge

echo "Script finished successfully"