#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --partition=standard
#SBATCH --account=leonnial
#SBATCH --job-name=pos_tag
#SBATCH --cpus-per-task=4
#SBATCH --array=0
#SBATCH -o ./logs/slurm/output_%A_%a.out

echo "Loading modules..."
module purge
module load virtualenv

echo "Activating virtual environment..."
source ./venv/pos_tagging/bin/activate

FOLDERS=("./data/csv/")
LEXEM=./data/lexems.txt

FOLDER=${FOLDERS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Processing folder: $FOLDER"

echo "Third task: LLM. Running..."
python -u ./src/pos_tagging/pos_tagging.py -f "$FOLDER" -l "$LEXEM"

deactivate

echo "Script finished!"
