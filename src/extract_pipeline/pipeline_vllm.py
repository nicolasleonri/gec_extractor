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
    temperature=0.1,
    top_p=0.8,
    top_k=10,
    min_p=0.2,
    max_tokens=8192,
    n=1,
    seed=42
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

def extract_prompt_length_from_error(error_message):
    """Extract the actual prompt length from the error message"""
    match = re.search(r'decoder prompt \(length (\d+)\)', str(error_message))
    if match:
        return int(match.group(1))
    return None

def chunk_text_by_tokens(text, max_input_tokens):
    """Split text into chunks based on token limits"""
    # Rough estimation: 1 token ≈ 4 characters for most models
    chars_per_token = 4
    max_chars = max_input_tokens * chars_per_token
    
    if len(text) <= max_chars:
        return [text]
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

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

def get_logs_files(directory, divided=False):
    if divided == False:
        SUPPORTED_FORMATS = ['.csv', '.tiff']
    else:
        SUPPORTED_FORMATS = ['.txt']
    
    logs_files = []
    
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            logs_files.append(file)
            
    output = sorted(logs_files)
    return output

def main():
    # Token budget calculation
    MAX_MODEL_TOKENS = 16384
    OUTPUT_TOKENS = 8192
    SYSTEM_PROMPT_TOKENS = 350
    MAX_INPUT_TOKENS = MAX_MODEL_TOKENS - OUTPUT_TOKENS - SYSTEM_PROMPT_TOKENS 

    parser = argparse.ArgumentParser(description='Preprocessor for document images.')
    parser.add_argument('-f', '--input_folder', required=True, help='Folder with OCR results')
    parser.add_argument('-n', '--newspaper', required=True, help='Newspaper name (required)')
    parser.add_argument('-i', '--input_type', choices=['complete', 'divided'], required=True, help='Type of input (required)')

    args = parser.parse_args()

    if args.input_type == 'divided':
        log_files = get_logs_files(str(args.input_folder), True)
        txt_files = []
        contents = []
        for txt_file in log_files:
            txt_files.append(str(txt_file))
            with open(txt_file, 'r', encoding='utf-8') as f:
                contents.append(f.read())
        combined_df = pd.DataFrame({
        'filename': txt_files,
        'extracted_text': contents
        })
    else:
        log_files = get_logs_files(str(args.input_folder))
        dfs = []
        for csv_file in log_files:
            df = pd.read_csv(csv_file)
            df = df[df['extracted_text'].notna() & df['extracted_text'].str.strip().astype(bool)]
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Total files: {len(combined_df)}")

    check_gpu()

    llm = LLM(
        model="unsloth/phi-4-unsloth-bnb-4bit",
        tensor_parallel_size=1,
        max_num_seqs=4096,
        enable_prefix_caching=True,
        enforce_eager=True,
        swap_space=16,
        max_num_batched_tokens=8192,
        max_model_len=16384,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
        block_size=160, #128
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
            if args.input_type == 'divided':
                idx = parts.index("txt")
                section = parts[idx + 1]
                year, month = parts[idx + 2: idx + 4]
                basename = path.name 
                day = basename.split("_")[1].split(".")[0]
                date_str = f"{year}/{month}/{day}"
            else:
                idx = parts.index("preprocessed")
                section = parts[idx + 1]
                year, month, day = parts[idx + 2: idx + 5]
                date_str = f"{year}/{month}/{day}"

            if str(section) != str(args.newspaper):
                print("Section found is not newspaper given.")
                continue

            parts = list(path.parts)
            if args.input_type == 'divided':
                idx = parts.index("txt")
                parts[idx] = "csv"
                idx = parts.index("data")
                parts[idx] = "results"
                csv_filename = f"{args.newspaper}_{year}_{month}_{day}.csv"
                output_file = Path(*parts).with_suffix(".csv")
                output_file = output_file.parent / csv_filename
            else:
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
            
            check_gpu()
            print(f"✓ Extracted text to {output_file.name}")
        except Exception as e:
            if "longer than the maximum model length" in str(e):
                actual_prompt_length = extract_prompt_length_from_error(str(e))
                print(f"Text too long ({actual_prompt_length} tokens), chunking: {filename}")
                safe_input_tokens = MAX_INPUT_TOKENS * 0.9
                
                chunks = chunk_text_by_tokens(str(val), int(safe_input_tokens))
                chunk_results = []
                
                for chunk_idx, chunk in enumerate(chunks):
                    print(f"Processing chunk: {int(chunk_idx)+1}/{len(chunks)}")

                    chunk_conversation = [
                        {"role": "system", "content": str(instructions)},
                        {"role": "user", "content": f"{chunk}"},
                    ]
                    
                    chunk_outputs = llm.chat(chunk_conversation, sampling_params)
                    chunk_result = chunk_outputs[0].outputs[0].text.strip()
                    chunk_result = extract_code_block(chunk_result, language_hint="csv")
                    print(chunk_result)

                    try:
                        f = StringIO(chunk_result)
                        reader = csv.reader(f, delimiter=';', quotechar='"')
                        rows = list(reader)
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        print(df)
                        df[["newspaper"]] = section
                        df[["date"]] = date_str

                        if output_file.exists():
                            existing_df = pd.read_csv(str(output_file))
                            combined_df_to_csv = pd.concat([existing_df, df], ignore_index=True)
                            combined_df_to_csv.to_csv(str(output_file), index=False, quoting=csv.QUOTE_ALL)
                        else:
                            df.to_csv(str(output_file), index=False, quoting=csv.QUOTE_ALL)
    
                        print(f"✓ Extracted chunk {int(chunk_idx)+1}/{len(chunks)} from {output_file.name}")
                    except Exception as e:
                        print(f"Error getting csv from {output_file.name} in chunk {int(chunk_idx)+1}/{len(chunks)}")
                        continue
                
                del chunks, chunk_outputs
                gc.collect()
                check_gpu()
                print(f"✓ Extracted text to {output_file.name}")
            else:
                del df, outputs
                gc.collect()
                check_gpu()                 
                filename = combined_df["filename"].tolist()[int(idx)]
                print(f"✗ Failed to process {filename}: {e}")

    total_time = time.time() - time_start
    check_gpu()

    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Processed {len(combined_df)} prompts in {total_time:.2f} seconds")
    print(f"Avg per (text) input: {total_time / len(combined_df):.2f} sec")


if __name__ == "__main__":
    main()