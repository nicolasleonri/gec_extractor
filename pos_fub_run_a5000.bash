#!/bin/bash
#SBATCH --job-name=sent_run_a5000
#SBATCH --output=./logs/slurm/output_fub_a5000_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --time=00:15:00

echo "Loading modules..."
module purge
module load virtualenv/20.26.2-GCCcore-13.3.0

echo "Activating virtual environment..."
source venv/pos_tagging/bin/activate

echo "Running python script..."
python3 -u ./src/pos_tagging/pos_tagging.py -f ./data/csv/pos_tagging -l ./data/csv/pos_tagging/lexems.txt

deactivate
module purge

echo "Script finished successfully"

###################### 01: Extraction task ###########################
# export VLLM_CACHE_ROOT=/home/nicolasal97/.cache/vllm
# source ./venv/extract_csv/bin/activate

# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2018 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2018 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2018 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/ojo/2018 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2018 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2018 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/2018 -tn H100 -aram 9 -ngpus 1

# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2014/ -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2013 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2013 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/ojo/2015 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2013 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2013 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/2014 -tn H100 -aram 9 -ngpus 1

# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2015/ -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2014 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2014 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/ojo/2016 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2014 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2014 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/2015 -tn H100 -aram 9 -ngpus 1

# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/correo/2019 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/elcomercio/2019 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/gestion/2019 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/ojo/2019 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/peru21/2019 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/publimetro/2019 -tn H100 -aram 9 -ngpus 1
# python3 -u ./src/extract_pipeline/extract_csv.py -f ./data/txt/trome/2019 -tn H100 -aram 9 -ngpus 1

###################### 02: BERTopic task ###########################
# source ./venv/bertopic/bin/activate

# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/elcomercio/ -n elcomercio -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/correo/ -n correo -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/peru21/ -n peru21 -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/trome/ -n trome -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/ojo/ -n ojo -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/publimetro/ -n publimetro -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/gestion/ -n gestion -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models
# python3 -u ./src/bertopic/bertopic_analyzer.py -f ./results/csv/ -n total -c 1 -mp /scratch/nicolasal97/gec_extractor/results/models

