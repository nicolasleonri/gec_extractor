from vllm import LLM, SamplingParams
from pathlib import Path
import subprocess
import argparse
import pandas as pd
import time

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    print("GPU Memory:", result.stdout.strip())

def get_logs_files(directory):
  SUPPORTED_FORMATS = ['.csv', '.tiff']
  logs_files = []
  
  for file in Path(directory).rglob('*'):
    if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
      logs_files.append(file)

  output = sorted(logs_files)
  return output

def main():
    parser = argparse.ArgumentParser(description='Preprocessor for document images.')
    parser.add_argument('-l', '--logs_folder', required=True, help='Folder with OCR results')
    args = parser.parse_args()

    llm = LLM(model="unsloth/phi-4-unsloth-bnb-4bit",)

    log_files = get_logs_files(str(args.logs_folder))
    print(log_files)
    
    # # Read all prompts from CSV files
    prompts = []
    for csv_file in log_files:
        print(f"Reading {csv_file}")
        print(csv_file)
        df = pd.read_csv(csv_file)
        print(df)
        texts = df['extracted_text'].dropna().tolist()
        texts = [t for t in texts if t.strip()]
        prompts.extend(texts)
    

    print(f"Total prompts: {len(prompts)}")

    time_start = time.time()
    check_gpu()
    outputs = llm.generate(prompts, sampling_params)
    print(outputs)
    check_gpu()
    total_time = time.time() - time_start

    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Processed {len(prompts)} prompts in {total_time:.2f} seconds")
    print(f"Avg per (text) input: {total_time / len(prompts):.2f} sec")


if __name__ == "__main__":
    main()