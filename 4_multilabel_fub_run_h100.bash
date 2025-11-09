#!/bin/bash
#SBATCH --job-name=multilabel_fub_run_a5000
#SBATCH --output=./logs/slurm/output_fub_multilabel_a5000_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=5G
#SBATCH --time=06:00:00

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

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ta random_swap -em accuracy -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ta random_swap -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -cv True -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -cv True -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -cv True -ta random_swap -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -cv True -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -cv True -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -cv True -ta random_swap -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ht True -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ht True -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ht True -ta random_swap -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ht True -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ht True -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ht True -ta random_swap -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ht True -cv True -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ht True -cv True -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em accuracy -ht True -cv True -ta random_swap -sm True

python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ht True -cv True -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ht True -cv True -ta random_deletion -sm True
python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp ./models/multi_label -tm True -ns 700 -em macro_f1 -ht True -cv True -ta random_swap -sm True

# python3 -u ./src/multi_label/multi_label_v2.py -f ./data/csv/multi_label/results_pos_2025-11-09.csv -mp ./models/multi_label/model_False_False_accuracy_100_accuracy_2025-11-09 -lm True

deactivate
module purge

echo "Script finished successfully"