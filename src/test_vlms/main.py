# https://huggingface.co/ds4sd/SmolDocling-256M-preview
# https://huggingface.co/nanonets/Nanonets-OCR-s
# https://huggingface.co/allenai/olmOCR-7B-0225-preview
import torch
import base64
import urllib.request
import io

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

def image_to_base64png(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode("utf-8")
    return base64_string

image = Image.open("./data/images/trome#2021-03-02#02.png")  # Replace with your image path

# Convert to RGB
rgb_image = image.convert("RGB")

base64_png = image_to_base64png(rgb_image)

# Initialize the model
model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Build the full prompt
messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the articles' headlines, subheadlines, authors and content as json"},
                    {"type": "image", "image": rgb_image},
                ],
            }
        ]

# Apply the chat template and processor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[text],
    images=[rgb_image],
    padding=True,
    return_tensors="pt",
)
inputs = {key: value.to(device) for (key, value) in inputs.items()}


# Generate the output
output = model.generate(
            **inputs,
            temperature=0.8,
            max_new_tokens=100000,
            num_return_sequences=1,
            do_sample=True,
        )

# Decode the output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(
    new_tokens, skip_special_tokens=True
)

print(text_output)
# ['{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":false,"natural_text":"Molmo and PixMo:\\nOpen Weights and Open Data\\nfor State-of-the']
