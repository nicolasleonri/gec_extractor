from vllm import LLM, SamplingParams
from pathlib import Path
import subprocess
import argparse
import pandas as pd
import time
import csv
from io import StringIO
import os
import gc
import re

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    top_k=40,
    min_p=0.05,
    max_tokens=4096,
    n=1,                     
    seed=42,
)

instructions = """
GOAL: Given the image, extract and structure the following information:
- headline of the article (string or "NA")
- subheadline of the article (string or "NA")
- author of the article (string or "NA")
- content of the article (string or "NA")

IMPORTANT:
- Focus only on articles that contain meaningful journalistic content.
- Exclude very short notices such as: date blocks, weather updates, advertisements, public announcements.
- Ask yourself: is this content relevant for media or discourse analysis? If not, skip it.
- If any field is missing or unknown, write "NA".

RETURN FORMAT:
Strictly output a valid CSV in the following format:
"headline";"subheadline";"author";"content"
"El loco del martillo";"NA";"La Seño María";"Hoy en día, uno pensaría que..."
"Contento por fin de cuarentena";"Habla Trome";"Ismael Lazo, Vecino de San Luis";"Estoy feliz porque..."

RULES:
- Do NOT include explanations, extra text, or commentary.
- Enclose each field in double quotes.
- Use semicolons (`;`) as field separators.
- Do NOT insert semicolons inside fields. If needed, replace them with commas.
- Each row represents one article. The first row must always be the CSV header.

CONTEXT:
You are an expert in analyzing and structuring newspaper content. Extracting accurate information is your professional responsibility. Be precise and thorough. If you make a mistake, the CSV will break and your credibility will suffer.
"""

def extract_code_block(text: str, language_hint: str = "csv") -> str:
    """Extracts a code block (e.g., CSV) from a markdown-formatted LLM response.

    Args:
        text (str): Full response string from LLM.
        language_hint (str, optional): Language label to look for (e.g., "csv").

    Returns:
        str: Cleaned code block string (e.g., CSV content).
    """
    if language_hint:
        pattern_lang = rf"```{language_hint}\n(.*?)```"
        match = re.search(pattern_lang, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    pattern_any = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_any, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

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
    parser.add_argument('-n', '--newspaper', required=True, help='Newspaper name (required)')

    args = parser.parse_args()

    log_files = get_logs_files(str(args.logs_folder))
    
    dfs = []
    for csv_file in log_files:
        df = pd.read_csv(csv_file)
        df = df[df['extracted_text'].notna() & df['extracted_text'].str.strip().astype(bool)]
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Total prompts: {len(combined_df)}")
        
    check_gpu()

    llm = LLM(
        model="unsloth/phi-4-unsloth-bnb-4bit",
        tensor_parallel_size=1,
        max_num_seqs=8192,
        enable_prefix_caching=True,
        enforce_eager=True,
        swap_space=0,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
        block_size=256,
        quantization="bitsandbytes",
        enable_chunked_prefill=True
    )

    check_gpu()

    time_start = time.time()

    for idx, val in enumerate(combined_df["extracted_text"].tolist()):
        try:
            conversation = [
                {"role": "system", "content": str(instructions)},
                {"role": "user", "content": str(val)},
            ]
            filename = combined_df["filename"].tolist()[int(idx)]
            path = Path(str(filename))
            parts = path.parts
            idx = parts.index("preprocessed")
            section = parts[idx + 1]
            year, month, day = parts[idx + 2: idx + 5]
            date_str = f"{year}/{month}/{day}"

            if str(section) != str(args.newspaper):
                print("Section found is not newspaper given.")
                continue

            parts = list(path.parts)
            idx = parts.index("images")
            parts[idx] = "csv"
            idx = parts.index("preprocessed")
            parts[idx] = "postprocessed"
            output_file = Path(*parts).with_suffix(".csv")
            os.makedirs(output_file.parent, exist_ok=True)

            if output_file.exists():
                print("Output already exists.")
                continue

            outputs = llm.chat(conversation, sampling_params)
            answer = outputs[0].outputs[0].text.strip()

            answer = extract_code_block(answer, language_hint="csv")

            f = StringIO(answer)
            reader = csv.reader(f, delimiter=';', quotechar='"')
            rows = list(reader)
            df = pd.DataFrame(rows[1:], columns=rows[0])  # assume first row is header
            df[["newspaper"]] = section
            df[["date"]] = date_str
            df.to_csv(str(output_file), index=False, quoting=csv.QUOTE_ALL)

            del df, outputs
            gc.collect()

            print(f"✓ Extracted text to {output_file.name}")
        except Exception as e:
            filename = combined_df["filename"].tolist()[int(idx)]
            gc.collect()
            print(f"✗ Failed to process {filename}: {e}")

    total_time = time.time() - time_start
    check_gpu()

    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Processed {len(combined_df)} prompts in {total_time:.2f} seconds")
    print(f"Avg per (text) input: {total_time / len(combined_df):.2f} sec")


if __name__ == "__main__":
    main()