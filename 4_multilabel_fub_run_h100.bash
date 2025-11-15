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
#SBATCH --mem=45G
#SBATCH --time=12:00:00

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

echo "Activating virtual environment..."
source venv/multi_label/bin/activate

echo "Running python script..."

### FINAL RUNS!
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -cv True -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -cv True -em macro_f1 -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -cv True -ta random_deletion -em accuracy -sm True

python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -cv True -ta random_swap -em macro_f1 -sm True
python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -cv True -ta random_deletion -em macro_f1 -sm True
python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -cv True -ta random_swap -em accuracy -sm True

### ONE TEST!
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -ta random_deletion -em macro_f1 -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -em accuracy -cv True -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -em macro_f1 -ht True -cv True -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 100 -ta random_deletion -em macro_f1 -ht True -cv True -sm True

### RERUN!
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -em macro_f1 -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ta random_deletion -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ta random_swap -em macro_f1 -sm True

# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -cv True -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -cv True -em macro_f1 -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -cv True -ta random_deletion -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -cv True -ta random_swap -em macro_f1 -sm True

# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -em macro_f1 -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -ta random_deletion -em accuracy -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -ht True -ta random_swap -em macro_f1 -sm True

### TO RERUN AND ANNOTATE!
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/etiquetado_perspectivas.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label -tm True -ns 600 -em macro_f1 -ta random_swap -sm True
# python3 -u ./src/multi_label/multi_label_v3.py -f ./data/csv/multi_label/results_pos_2025-11-09.csv -mp /scratch/nicolasal97/gec_extractor/results/models/multi_label/model_False_False_accuracy_100_accuracy_2025-11-09 -lm True

deactivate
module purge

echo "Script finished successfully"