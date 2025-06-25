import re
import os
from collections import defaultdict

def log_processing_info(log_file_path, processed_filename, config_number, ocr_name, model_display_name, time_elapsed):
    """Log processing information with immediate flush"""
    log_entry = f"File: {processed_filename} - Config: {config_number} - OCR: {ocr_name} - LLM: {model_display_name} - Time needed: {time_elapsed}s\n"
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)
        f.flush()  # Immediate flush to disk

def extract_filename_and_config(filepath):
    """Extract processed filename and config number in one line each"""
    filename = os.path.basename(filepath)
    config_number = int(re.search(r'_config(\d+)', filename).group(1)) if re.search(r'_config(\d+)', filename) else None
    processed_filename = str(re.sub(r'_config\d+', '', filename))
    # ocr_number = 
    return processed_filename, config_number

def parse_ocr_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to match each entry
    pattern = r'File: (.*?) - Config: (.*?) - OCR: (.*?) - Time needed: (.*?) - \[(.*?)\]'
    # pattern = r'File: (.*?) - Time needed: (.*?) - Config: (.*?) - \[(.*?)\]'
    
    # Find all matches
    matches = re.findall(pattern, content, re.DOTALL)
    # print(len(matches))

    idx = 0
    results = defaultdict(dict)

    for file_path, config, ocr, time, text in matches:
        clean_text = ' '.join(text.split())
        results[idx] = {'filepath': str(file_path), 'config': str(config), 'ocr': str(ocr), 'text': str(clean_text)} # Config of OCR
        idx += 1
    
    return results