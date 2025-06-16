import csv
from pathlib import Path
import subprocess
import torch

def load_csv_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        return list(reader)

def process_csvs(input_folder):
    csvs = {}
    csv_path = Path(input_folder)
    
    for file_path in csv_path.glob('*.csv'):
        stem = str(file_path.stem.split('/'))
        csv_data = load_csv_file(str(file_path))
        csvs[stem] = csv_data

    return csvs

def print_gpu_usage(note=""):
    print(f"\n🔍 GPU Memory Usage {note}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        print(output)
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")
