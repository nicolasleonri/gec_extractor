#!/usr/bin/env python3
"""
LLM Text Extraction Pipeline - Argument Parser and Configuration
"""

import gc
import os
import re
import sys
import csv
import torch
import math
import tiktoken
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from io import StringIO
from pathlib import Path
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
from huggingface_hub import hf_hub_download
from vllm.distributed.parallel_state import destroy_model_parallel

def check_cpu():
   total, used, free = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
   print("RAM: ", used, " (used)", free, " (free)")

def check_gpu():
   result = subprocess.run(
      ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
      stdout=subprocess.PIPE, text=True
   )
   print("GPU Memory:", result.stdout.strip())

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract structured information from text files using LLM", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-f", "--input_folder", type=str, required=True, help="Path to folder containing input text files"
    )
    parser.add_argument("-tn", "--type_of_nvidia_card", type=str, required=True,
        choices=["H100", "A5000", "RTX2080", "A100", "V100"], help="Type of NVIDIA GPU card"
    )
    parser.add_argument("-aram", "--available_ram", type=int, required=True, help="Available system RAM in GB"
    )
    parser.add_argument("-ngpus", "--num_gpus", type=int, required=True, help="Number of GPUs available for processing")
    
    return parser.parse_args()

def count_tokens(text: str) -> int:
        try:
            # Use cl100k_base encoding (GPT-4/GPT-3.5-turbo) as approximation
            # This gives us a reasonable estimate for most modern LLMs
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough approximation (1 token ≈ 4 characters for English)
            return len(text) // 4

def validate_arguments(args) -> Dict[str, Any]:
    config = {}
    errors = []
    
    if not os.path.exists(args.input_folder):
        errors.append(f"Input folder does not exist: {args.input_folder}")
    else:
        txt_files = list(Path(args.input_folder).rglob("*.txt"))
        if len(txt_files) == 0:
            errors.append(f"No .txt files found in input folder: {args.input_folder}")
        config["txt_files_count"] = len(txt_files)
        config["input_folder"] = args.input_folder
    
    if args.available_ram <= 0:
        errors.append("Available RAM must be positive")
    if args.num_gpus <= 0:
        errors.append("Number of GPUs must be positive")
    
    config["gpu_type"] = args.type_of_nvidia_card
    config["system_ram_gb"] = args.available_ram
    config["num_gpus"] = args.num_gpus
    
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def get_txt_files(directory):
    SUPPORTED_FORMATS = ['.txt']
    
    logs_files = []
    
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            logs_files.append(file)
            
    output = sorted(logs_files)

    return output

def read_and_preprocess_files(txt_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    def preprocess_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines -> double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        text = re.sub(r'\n ', '\n', text)  # Remove spaces at start of lines
        
        # Remove common file artifacts
        text = re.sub(r'\x0c', '', text)  # Form feed characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Control chars
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")  # Smart quote
        text = text.replace('â€œ', '"')  # Smart quote
        text = text.replace('â€', '"')   # Smart quote
        text = text.replace('â€"', '—')  # Em dash
        text = text.replace('â€"', '–')  # En dash
        
        # Normalize quotes and dashes
        # text = re.sub(r'["""]', '"', text)  # Various quotes to standard
        # text = re.sub(r'['']', "'", text)  # Various apostrophes to standard
        # text = re.sub(r'[—–]', '-', text)   # Various dashes to hyphen
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots -> ellipsis
        text = re.sub(r'[!]{2,}', '!', text)   # Multiple exclamation -> single
        text = re.sub(r'[?]{2,}', '?', text)   # Multiple question -> single
        
        # Strip and ensure we don't have empty result
        text = text.strip()
        
        return text

    processed_files = {}
    failed_files = []

    for file_path in tqdm(txt_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_content = f.read()
            
            clean_content = preprocess_text(raw_content)
            
            if not clean_content:
                print(f"Skipping empty file: {file_path}")
                continue
            
            token_count = count_tokens(clean_content)
                        
            processed_files[file_path.name] = {
                "content": clean_content,
                "num_tokens": token_count,
                "file_path": str(file_path)
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Successfully processed: {len(processed_files)} files")

    if processed_files:
        total_tokens = sum(data["num_tokens"] for data in processed_files.values())
        avg_tokens = total_tokens / len(processed_files)
        max_tokens = max(data["num_tokens"] for data in processed_files.values())
        min_tokens = min(data["num_tokens"] for data in processed_files.values())
        
        # print(f"Token Statistics:")
        # print(f"  Total tokens: {total_tokens:,}")
        # print(f"  Average tokens per file: {avg_tokens:,.0f}")
        # print(f"  Max tokens in a file: {max_tokens:,}")
        # print(f"  Min tokens in a file: {min_tokens:,}")
        
        # # Chunking recommendations
        # context_window = 128000  # Mistral-Nemo context window
        # files_needing_chunking = sum(1 for data in processed_files.values() 
        #                         if data["num_tokens"] > context_window * 0.8)  # 80% threshold
        
        # if files_needing_chunking > 0:
        #     print(f"  📝 Files likely needing chunking: {files_needing_chunking}")
    
    return processed_files

def get_model_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    model_config = {
        "model_name": "curiousmind147/microsoft-phi-4-AWQ-4bit-GEMM",
        "quantization": "awq_marlin",
        "tokenizer_mode": "mistral",
        "swap_space": math.floor((int(config["system_ram_gb"])-10)/4),
        "cpu_offload": math.floor((int(config["system_ram_gb"])-10)/4*3),
        "tensor_parallel_size": int(config["num_gpus"])
    }

    if config["gpu_type"] == "H100":
        model_config.update({
            "max_model_len": 16000, # for phi4
            "gpu_memory_utilization": 0.90,
        })
    elif config["gpu_type"] == "RTX2080":
        model_config.update({
            "max_model_len": 4000,
            "gpu_memory_utilization": 0.80,
        })
    elif config["gpu_type"] == "A100":
        model_config.update({
            "max_model_len": 80000,
            "gpu_memory_utilization": 0.85,
        })
    elif config["gpu_type"] == "A5000":
        model_config.update({
            "max_model_len": 8000, # 65536 for mistral, 8096 for phi4
            "gpu_memory_utilization": 0.9,
        })

    return model_config

def process_chunking_decision(processed_files: Dict[str, Dict[str, Any]], model_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    def calculate_chunk_parameters(model_config: Dict[str, Any]) -> Dict[str, int]:
        max_model_len = model_config["max_model_len"]
        
        usable_context = int(max_model_len * 0.40)  # reserve 60% of context for output and prompt

        overlap_ratio = 0.1   # 10% overlap
        chunk_size = min(usable_context, 100000)
        
        overlap_size = int(chunk_size * overlap_ratio)
        
        return {
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "overlap_ratio": overlap_ratio,
            "usable_context": usable_context
        }
    
    def smart_text_splitter(text: str, chunk_size: int, overlap_size: int) -> List[str]:
        if not text:
            return []
        
        chunk_chars = chunk_size * 4
        overlap_chars = overlap_size * 4
        
        if len(text) <= chunk_chars:
            return [text]  # No chunking needed
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(text):
            end_pos = min(start_pos + chunk_chars, len(text))
            
            if end_pos < len(text):
                paragraph_break = text.rfind('\n\n', start_pos, end_pos)
                if paragraph_break > start_pos + chunk_chars * 0.7:  # At least 70% of chunk size
                    end_pos = paragraph_break + 2  # Include the newlines
                else:
                    # Look for sentence breaks
                    sentence_break = text.rfind('. ', start_pos, end_pos)
                    if sentence_break > start_pos + chunk_chars * 0.7:
                        end_pos = sentence_break + 2  # Include the period and space
                    else:
                        # Look for line breaks
                        line_break = text.rfind('\n', start_pos, end_pos)
                        if line_break > start_pos + chunk_chars * 0.7:
                            end_pos = line_break + 1  # Include the newline
                        # Otherwise, use hard cutoff (fallback)
            
            chunk = text[start_pos:end_pos].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            if end_pos >= len(text):
                break  # We've reached the end
                
            next_start = max(end_pos - overlap_chars, start_pos + chunk_chars // 2)
            start_pos = next_start
        
        return chunks

    chunk_params = calculate_chunk_parameters(model_config)

    files_needing_chunking = []
    files_no_chunking = []
    
    for filename, file_data in processed_files.items():
        if file_data["num_tokens"] > chunk_params["usable_context"]:
            files_needing_chunking.append(filename)
        else:
            files_no_chunking.append(filename)
    
    # print(f"Chunking Analysis:")
    # print(f" Files that fit in context: {len(files_no_chunking)}")
    # print(f" Files needing chunking: {len(files_needing_chunking)}")
    
    total_chunks_created = 0
    
    for filename in tqdm(processed_files.keys(), desc="Chunking analysis"):
        file_data = processed_files[filename]
        
        if file_data["num_tokens"] > chunk_params["usable_context"]:
            # print(f"Chunking {filename} ({file_data['num_tokens']:,} tokens)")

            chunks = smart_text_splitter(
                file_data["content"],
                chunk_params["chunk_size"],
                chunk_params["overlap_size"]
            )
            
            file_data["is_chunked"] = True
            file_data["chunks"] = chunks
            file_data["num_chunks"] = len(chunks)
            file_data["chunk_params"] = chunk_params.copy()
            
            chunk_tokens = []
            for i, chunk in enumerate(chunks):
                chunk_token_count = count_tokens(chunk)
                chunk_tokens.append(chunk_token_count)
            
            file_data["chunk_token_counts"] = chunk_tokens
            file_data["total_chunk_tokens"] = sum(chunk_tokens)
            
            total_chunks_created += len(chunks)
            
            # print(f"Created {len(chunks)} chunks, total tokens: {sum(chunk_tokens):,}")
        else:
            file_data["is_chunked"] = False
            file_data["chunks"] = [file_data["content"]]  # Single "chunk" for consistency
            file_data["num_chunks"] = 1
            file_data["chunk_token_counts"] = [file_data["num_tokens"]]
            file_data["total_chunk_tokens"] = file_data["num_tokens"]
    
    return processed_files

def prepare_prompts(processed_txt_files, prompt_prefix=""):

    all_chunks = []

    for filename, data in processed_txt_files.items():
        chunks = data.get("chunks", [])
        token_counts = data.get("chunk_token_counts", [])

        paired = sorted(zip(chunks, token_counts), key=lambda x: x[1])

        for chunk, count in paired:
            all_chunks.append((filename, chunk, count))

    all_chunks.sort(key=lambda x: x[2])

    prompts = [prompt_prefix + chunk for _, chunk, _ in all_chunks]

    return all_chunks, prompts

def save_outputs_to_csv(outputs, all_chunks, config, processed_txt_files):
    def extract_code_block(text: str, language_hint: str = "csv") -> str:
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

    def extract_metadata_from_path(file_path):
        parts = os.path.normpath(file_path).split(os.sep)
        filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        filename_parts = filename_no_ext.split("_")

        newspaper = filename_parts[0]
        day = filename_parts[-1].zfill(2) if len(filename_parts) > 1 and filename_parts[-1].isdigit() else None

        year = month = None
        for i in range(len(parts)):
            if parts[i].isdigit() and len(parts[i]) == 4:  # Year
                year = parts[i]
                month = parts[i + 1] if i + 1 < len(parts) else "01"
                break

        date_str = None
        if year and month:
            date_str = f"{month.zfill(2)}-{year}"
            if day:
                date_str = f"{day}-{month.zfill(2)}-{year}"

        return newspaper, date_str

    for (filename, _, _), output_obj in zip(all_chunks, outputs):
        try:
            if hasattr(output_obj, "outputs"):
                if output_obj.outputs and hasattr(output_obj.outputs[0], "text"):
                    output = output_obj.outputs[0].text
                else:
                    raise ValueError(f"No text found in output for {filename}")
            elif isinstance(output_obj, str):
                output = output_obj
            else:
                raise TypeError(f"Unsupported output type: {type(output_obj)} for {filename}")

            answer = extract_code_block(output, language_hint="csv")

            if not answer:
                raise ValueError(f"No CSV block found in output for {filename}")

            f = StringIO(answer)
            reader = csv.reader(f, delimiter=';', quotechar='"')
            rows = list(reader)
            
            if not rows:
                raise ValueError(f"CSV parsing failed for {filename}")

            df = pd.DataFrame(rows[1:], columns=rows[0])  # assume first row is header

            full_path = Path(processed_txt_files[filename]["file_path"])
            newspaper, date_str = extract_metadata_from_path(full_path)

            df["newspaper"] = newspaper
            df["date"] = date_str if date_str else "NA"

            output_file = full_path.parent / f"{newspaper}_{date_str}.csv"

            if os.path.exists(output_file):
                df.to_csv(output_file, mode="a", header=False, index=False, quoting=csv.QUOTE_ALL)
                print(f"Appended rows to existing CSV: {output_file}")
            else:
                df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
                print(f"Created new CSV: {output_file}")
        except Exception as e:
            fallback_file = os.path.join(
                config["input_folder"], 
                f"{os.path.splitext(filename)[0]}_error.txt"
            )
            with open(fallback_file, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Error processing {filename}, saved raw output to {fallback_file}. Error: {e}")

def main():
    args = parse_arguments()
        
    config = validate_arguments(args)

    txt_files = get_txt_files(config["input_folder"])

    processed_txt_files = read_and_preprocess_files(txt_files)

    model_config = get_model_configuration(config)

    processed_txt_files = process_chunking_decision(processed_txt_files, model_config)

    all_chunks, prompts = prepare_prompts(processed_txt_files)

    sampling_params = SamplingParams(
        n=1, # number of completions to generate
        temperature=0.0,
        top_p=0.95, # nucleus filtering enabled for diversity
        top_k=0, # small top-k to reduce very unlikely tokens
        seed=42,
        min_p=0.0,
        # min_tokens=int(model_config["max_model_len"] * 0.225),  # minimum length
        max_tokens=int(model_config["max_model_len"] * 0.5),  # maximum length
        repetition_penalty=1.25
    )

    repo_id = "MaziyarPanahi/phi-4-GGUF"
    filename = "phi-4.Q4_K_M.gguf"
    tokenizer = "microsoft/phi-4"
    model = hf_hub_download(repo_id, filename=filename)

    llm = LLM(
        model=model,
        # tokenizer_mode=tokenizer,
        # load_format="safetensors",
        # config_format="mistral",
        max_model_len=model_config["max_model_len"], 
        gpu_memory_utilization=model_config["gpu_memory_utilization"],
        seed=42,
        # quantization=model_config["quantization"],
        swap_space=model_config["swap_space"],
        cpu_offload_gb=0, # for h100 with enough VRAM
        max_seq_len_to_capture=16384,
        tensor_parallel_size=model_config["tensor_parallel_size"],
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        # task='generate',
        disable_log_stats=True,
        enforce_eager=False
    )

    check_gpu()
    check_cpu()

    instructions = """
    THIS IS CRITICAL: Failure is not an option. If you do not strictly output a valid CSV, people may DIE!
    You are a newspaper content extractor and CSV formatter.
    Input: raw OCR text from a Spanish (Peruvian) newspaper page.
    Task: extract all meaningful articles and produce a CSV ONLY.
    Do NOT add commentary, reasoning, explanations, or any text outside the CSV.
    Preserve Spanish accents and punctuation.
    Columns: headline; subheadline; author; content
    Use "NA" if a field is missing.
    Enclose all fields in double quotes.
    Replace semicolons inside fields with commas.
    Replace newlines inside fields with spaces.
    Escape internal quotes by doubling them.
    Output must start with the header row exactly as:
    "headline";"subheadline";"author";"content"
    Each row = one article.
    No extra lines or formatting allowed.
    """

    user_prompt_template = """
    Output: a valid CSV with header "headline";"subheadline";"author";"content"
    Strictly output CSV only, no extra text or explanation.
    Your life depends on it—failure is fatal!

    Example:
    "headline";"subheadline";"author";"content"
    "Noticias del día";"Resumen principal";"NA";"El contenido principal del artículo va aquí.

    OCR TEXT:
    {ocr_text}
    "
    """

    chat_messages = []
    for _, chunk, _ in all_chunks:
        chat_messages.append([
            {"role": "system", "content": str(instructions)},
            {"role": "user", "content": user_prompt_template.format(ocr_text=chunk)}
        ])

    # outputs = llm.generate(prompts, sampling_params)
    outputs = llm.chat(chat_messages, sampling_params)

    for i, (prompt, output_obj) in enumerate(zip(prompts, outputs)):
        if hasattr(output_obj, "outputs") and output_obj.outputs and hasattr(output_obj.outputs[0], "text"):
            output_text = output_obj.outputs[0].text
        elif isinstance(output_obj, str):
            output_text = output_obj
        else:
            output_text = f"[Unexpected output type: {type(output_obj)}]"

        print("=" * 80)
        # print(f"🔹 Prompt {i+1}:\n{prompt}\n")
        print(f"🔸 Output {i+1}:\n{output_text}\n")
        print("=" * 80)

    check_gpu()
    check_cpu()

    save_outputs_to_csv(outputs, all_chunks, config, processed_txt_files)

    destroy_model_parallel()
    del llm 
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


# TODO: Fix intel_extension_for_pytorch
# TODO: Try with FlashInfer