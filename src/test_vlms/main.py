from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image

print("TEST")

image_path = "./data/images/trome#2021-03-02#02.png"
local_image = Image.open(image_path).convert("RGB")

model_path = "nanonets/Nanonets-OCR-s"

model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype="auto", 
    device_map="auto", 
    # attn_implementation="flash_attention_2"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)


def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

image_path = "./data/images/trome#2021-03-02#02.png"
result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=15000)
print(result)

# prompt = """
#         GOAL: Given the raw text output from an OCR-Engine, extract and structure the following information:
#         - headline of the article: string or NA
#         - subheadline of the article: string or NA
#         - author of the article:: string or NA
#         - content of the article:: string

#         Focus on extracting articles (small or long) with actual information. Exclude any brief news items like: date, weather, public announcements, or other short notices that do not contain substantial content. Ask yourself if the content is relevant for media and discourse anaylisis. If any field is missing or unavailable, use "NA" for its value.

#         RETURN FORMAT: Format the output strictly as a pythonic dictionary with keys and values like the following example:
#         "headline";"subheadline";"author";"content"
#         "El loco del martillo";"NA";"La Seño María";"Hoy en día, uno pensaría que..."
#         "Contento por fin de cuarentena";"Habla Trome";"Ismael Lazo, Vecino de San Luis";"Estoy feliz porque..."

#         WARNING: Only return the CSV exactly as specified. Do not add any explanation or commentary.
#         WARNING: Avoid CSV parsing errors like empty or malformed outputs. After the header, each row should contain the extracted information for one article, with fields separated by semicolons and declared in quotation marks.

#         CONTEXT: You are an expert in analyzing extracted newspaper content. Your task is to carefully extract articles and brief news items from the provided text. If you fail in this task, you will lose your job and your family will be very disappointed in you.
#         """ 

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype="auto", device_map="auto"
# )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": local_image,
#                 # local_image
#             },
#             {"type": "text", "text": prompt},
#         ],
#     }
# ]

# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to("cuda")

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

# # Writing objects:  46% (1426/3073)