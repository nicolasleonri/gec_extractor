from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from llama_cpp import Llama
from pathlib import Path
from PIL import Image
import base64
import time
import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import io

llm = None
prompt = None
log_file = None

def shrink_image(image_path, max_dim=1024):
  """
  Shrinks an image so that its largest side <= max_dim
  """
  img = Image.open(image_path)
  w, h = img.size

  if max(w, h) <= max_dim:
      return img  # already small enough

  # compute new dimensions preserving aspect ratio
  if w > h:
    new_w = max_dim
    new_h = int(h * max_dim / w)
  else:
    new_h = max_dim
    new_w = int(w * max_dim / h)

  return img.resize((new_w, new_h), Image.BICUBIC)

def init_worker(shared_prompt, shared_log_file):
  """Initialize model once per process."""
  global llm, prompt, log_file

  chat_handler = Qwen25VLChatHandler.from_pretrained(
    repo_id="unsloth/Nanonets-OCR-s-GGUF",
    filename="mmproj-BF16.gguf",
  )

  llm = Llama.from_pretrained(
    repo_id="unsloth/Nanonets-OCR-s-GGUF",
    # filename="Nanonets-OCR-s-Q4_K_M.gguf",
    filename="Nanonets-OCR-s-Q5_K_M.gguf",
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=16384,
    n_threads=int(os.cpu_count()),
    n_batch=16384,
    n_ubatch=16384,
    use_mmap=True,
    use_mlock=True,
    numa=True,
    split_mode=1,
    flash_attn=True,
    verbose=False
  )

  prompt = shared_prompt
  log_file = shared_log_file

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    print("GPU Memory:", result.stdout.strip())

def encode_image(image_path):
  with open(image_path, 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  return encoded_string

def get_image_files(directory):
  SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']
  image_files = []
  
  for file in Path(directory).rglob('*'):
    if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
      image_files.append(file)

  output = sorted(image_files)
  return output

def save_to_csv_log(filename, extracted_text, log_file):
  file_exists = Path(log_file).exists()
  
  with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    
    if not file_exists:
      writer.writerow(['filename', 'extracted_text'])
    
    writer.writerow([filename, extracted_text])

def process_image(file_path):
  """Use preloaded model to process one image."""
  global llm, prompt, log_file

  start_time  = time.time()

  try:
    img = shrink_image(file_path, max_dim=1024)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_input = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    # img_input = encode_image(file_path)

    response = llm.create_chat_completion(
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_input}"}},
        ]}
      ],
      temperature=0.0,
      max_tokens=16384,
      repeat_penalty=1.125,
      top_p=0.95,
      top_k=5,
      min_p=0.1,
      stream=False,
      seed=42,
      typical_p=0.95,
      tfs_z=0.95,
      )

    check_gpu()
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")

    extracted_text = response["choices"][0]["message"]["content"]
    save_to_csv_log(str(file_path), extracted_text, log_file)

    del img_input, response, extracted_text
    gc.collect()

    return f"✓ Extracted text from {file_path.name}"

  except Exception as e:
      gc.collect()
      return f"✗ Failed to process {file_path.name}: {e}"

def main():
  shared_prompt = """
  You are an expert OCR system. Extract ALL text content from this newspaper image with perfect accuracy.

  CRITICAL REQUIREMENTS:
  - Read every single word, number, date, and punctuation mark visible in the image
  - The text is in SPANISH - preserve all Spanish accents, tildes, and special characters (ñ, á, é, í, ó, ú, ü)
  - Preserve the original text layout and structure (headlines, paragraphs, columns)
  - Maintain proper spacing between words and sentences
  - Include ALL content: headlines, subheadings, body text, captions, advertisements, page numbers, dates
  - Handle multiple columns by reading left-to-right, top-to-bottom within each column
  - Preserve special characters, accents, and non-English text exactly as shown
  - Do NOT skip any text, even if partially obscured or small
  - Do NOT add explanations, interpretations, or markdown formatting
  - Do NOT summarize or paraphrase - extract the exact text as written
  
  Return ONLY the raw extracted text content, preserving the natural reading flow of the newspaper.
  DO NOT REPEAT CONTENT. IF YOU REPEAT CONTENT MORE THAN TWICE, YOU WILL RECEIVE A NEGATIVE GRADE (REINFORCEMENT LEARNING, RL)

  WARNING: If you return anything other than raw text (explanations, apologies, formatting, etc.), 
  the entire OCR pipeline will fail and all downstream processing will be corrupted. 
  Your response must contain ONLY the extracted text - nothing else.

  CONTEXT: This is a test. You are being compared to other VLMs. You have to be quick and good.

  TIP: If something is being repeated more than twice, it is -for sure- an error.
  """

  log_timestamp = time.strftime("%Y%m%d_%H%M%S")
  shared_log_file = f"./logs/ocr_results/ocr_log_{log_timestamp}.csv"

  img_list = get_image_files('./results/images/preprocessed')

  start_time = time.time()

  with ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(shared_prompt, shared_log_file)) as executor:
    futures = [executor.submit(process_image, f) for f in img_list]
    for future in as_completed(futures):
      print(future.result())

  total_time = time.time() - start_time
  print(f"Processed {len(img_list)} images in {total_time:.2f} seconds")
  print(f"Avg per image: {total_time / len(img_list):.2f} sec")
  print(f"Log saved to {shared_log_file}")

  return None

if __name__ == "__main__":
  main()