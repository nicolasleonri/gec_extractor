#!/usr/bin/env python3
"""
LLM Text Extraction Pipeline - Argument Parser and Configuration
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any
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
    parser.add_argument("-gbn", "--gb_of_nvidia_card", type=int, required=True, 
        help="VRAM in GB of each NVIDIA card"
    )
    parser.add_argument("-aram", "--available_ram", type=int, required=True, help="Available system RAM in GB"
    )
    parser.add_argument("-ngpus", "--num_gpus", type=int, required=True, help="Number of GPUs available for processing")
    
    return parser.parse_args()

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
    
    if args.gb_of_nvidia_card <= 0:
        errors.append("GPU memory must be positive")
    if args.available_ram <= 0:
        errors.append("Available RAM must be positive")
    if args.num_gpus <= 0:
        errors.append("Number of GPUs must be positive")
    
    config["gpu_type"] = args.type_of_nvidia_card
    config["gpu_memory_gb"] = args.gb_of_nvidia_card
    config["system_ram_gb"] = args.available_ram
    config["num_gpus"] = args.num_gpus
    config["total_gpu_memory"] = args.gb_of_nvidia_card * args.num_gpus
    
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def main():
    args = parse_arguments()
        
    config = validate_arguments(args)

    print(config)

    prompts = [
        "Fix this grammar: I are going to the store.",
        "Correct this sentence: She don't like apples.",
        "Sing me a song.",
        "Translate this to Spanish: How are you today?",
        "Summarize this: Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence.",
        "Explain this like I'm five: Why is the sky blue?",
        "Make this sentence more formal: Gimme that report ASAP.",
        "Turn this into a haiku: The sun sets slowly / Painting the clouds with bright fire / Night embraces all.",
        "Fix this grammar: I are going to the store.",
        "Correct this sentence: She don't like apples.",
        "Sing me a song.",
        "Translate this to Spanish: How are you today?",
        "Summarize this: Artificial intelligence is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence.",
        "Explain this like I'm five: Why is the sky blue?",
        "Make this sentence more formal: Gimme that report ASAP.",
        "Turn this into a haiku: The sun sets slowly / Painting the clouds with bright fire / Night embraces all.",
    ]

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0, # disable nucleus filtering
        top_k=0, # disable top-k filtering (greedy decode)
        seed=42,
        min_tokens=1,
        repetition_penalty=1.0
    )

    #model="GameScribes/Mistral-Nemo-AWQ",

    llm = LLM(
        model="GameScribes/Mistral-Nemo-AWQ", # TODO: Try with AWQ model
        tokenizer_mode="mistral",
        # load_format="safetensors",
        # config_format="mistral",
        max_model_len=8192, # TODO: Increase to 76512
        gpu_memory_utilization=0.9,
        seed=42,
        quantization="awq_marlin",
        swap_space=30,
        cpu_offload_gb=50,
        max_seq_len_to_capture=8192,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        task='generate',
        disable_log_stats=True,
        enforce_eager=False,
        # block_size=16,
    )

    check_gpu()
    check_cpu()

    outputs = llm.generate(prompts, sampling_params) # Trying with batch of 16 prompts

    check_gpu()
    check_cpu()

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 50)

    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm # Isn't necessary for releasing memory, but why not
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("TEST")
    main()