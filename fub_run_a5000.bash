#!/bin/bash
#SBATCH --job-name=run_a5000
#SBATCH --output=./logs/slurm/output_fub_a5000_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=9G

#SBATCH --time=2-00:00:00

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
python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2017 -tn A5000 -aram 9 -ngpus 1
python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2017 -tn A5000 -aram 9 -ngpus 1

deactivate
module purge

echo "Script finished successfully"

jobstats %j
seff %j

###################### 01: Extraction task ###########################
# source ./venv/extract_csv/bin/activate

# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2016 -tn A5000 -aram 9 -ngpus 1
# * python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2015 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2015 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/ojo/2017 -tn A5000 -aram 9 -ngpus 1

# FAILED -> moved to H100
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2015 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2015 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/2016 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2017 -tn A5000 -aram 9 -ngpus 1
# FAILED

# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2016 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2016 -tn A5000 -aram 9 -ngpus 1
# * python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2016 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2016 -tn A5000 -aram 9 -ngpus 1

# FAILED -> moved 3x to H100
# (moved) python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/2017 -tn A5000 -aram 9 -ngpus 1
# (moved) python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2017 -tn A5000 -aram 9 -ngpus 1
# (moved) python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2017 -tn A5000 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2017 -tn A5000 -aram 9 -ngpus 1
# * python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2017 -tn A5000 -aram 9 -ngpus 1
# FAILED

###################### 02: BERTopic task ###########################
# source ./venv/bertopic/bin/activate
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/elcomercio/ -n elcomercio -c 1 --load_model False

