from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
import base64
from PIL import Image
import time

def encode_image(image_path):
  image_path = shrink(image_path)
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode("utf-8")

def shrink(path, max_size=512):
  img = Image.open(path)
  img.thumbnail((max_size, max_size))
  out = path.replace(".jpg", "_small.jpg")
  img.save(out, "JPEG", quality=85)
  return out

def image_to_base64_data_uri(file_path):
  with open(file_path, "rb") as img_file:
      base64_data = base64.b64encode(img_file.read()).decode('utf-8')
      return f"data:image/png;base64,{base64_data}"

chat_handler = Qwen25VLChatHandler.from_pretrained(
  repo_id="unsloth/Nanonets-OCR-s-GGUF",
  filename="mmproj-BF16.gguf",
)

llm = Llama.from_pretrained(
  repo_id="unsloth/Nanonets-OCR-s-GGUF",
  filename="Nanonets-OCR-s-Q4_K_M.gguf",
  chat_handler=chat_handler,
  n_ctx=8192,
  n_threads=8,
  n_gpu_layers=-1
)

file_path = './data/images/test.jpg'
data_uri = encode_image(file_path)

time_start = time.time()
response = llm.create_chat_completion(
  messages=[
    {"role": "system", "content": "You are an assistant who perfectly describes images."},
    {
        "role": "user",
        "content": [ 
          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data_uri}"}},
          {"type": "text", "text": "Describe this image in detail please."}
        ]
    }
]
)
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
print(response["choices"][0]["message"]["content"])