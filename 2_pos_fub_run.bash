#!/bin/bash
#SBATCH --job-name=pos_fub_run_cpu
#SBATCH --output=./logs/slurm/output_fub_pos_cpu_%j.out

#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --time=00:45:00

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