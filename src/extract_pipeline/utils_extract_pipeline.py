import time
from openai import OpenAI
import base64

def get_client_model():
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    return model, client

def ask_llm(client, model, instructions, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=0.70,                # Nucleus sampling
        max_tokens=3000,           # Maximum tokens to generate
        n=1,                      # Number of completions
        stream=False,             # Whether to stream response
        seed=42,                  # Random seed for reproducibility
        extra_body={ 
            # Aggressive sampling
            "top_k": 5,             # Very small candidate pool
            "min_p": 0.15,            # High threshold

            # Disable all penalties and extras
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,

            # Speed optimizations
            "use_beam_search": False,
            "best_of": 1,
            "skip_special_tokens": True,
            "spaces_between_special_tokens": False,
            "min_tokens": 20,        # Minimum for CSV header

            # Disable overhead
            "stop_token_ids": [],
            "include_stop_str_in_output": False,
            "ignore_eos": False,
            "prompt_logprobs": None,
            "allowed_token_ids": None,
            "bad_words": [],

            "prompt_logprobs": None,         # Disable for speed
            "allowed_token_ids": None,       # Don't restrict (faster)
            "bad_words": [],                 # Empty list (no filtering overhead)
        }
    )
    return response.choices[0].message.content

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ask_vlm(client, model, instructions, img_base64):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": instructions,
                    },
                ],
            }
        ],
        temperature=0.0,
        top_p=0.60,                # Nucleus sampling
        max_tokens=3000,           # Maximum tokens to generate
        n=1,                      # Number of completions
        stream=False,             # Whether to stream response
        seed=42,                  # Random seed for reproducibility
        extra_body={ 
            # Aggressive sampling
            "top_k": 3,             # Very small candidate pool
            "min_p": 0.1,            # High threshold

            # Disable all penalties and extras
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,

            # Speed optimizations
            "use_beam_search": False,
            "best_of": 1,
            "skip_special_tokens": True,
            "spaces_between_special_tokens": False,
            "min_tokens": 10,        # Minimum for CSV header

            # Disable overhead
            "stop_token_ids": [],
            "include_stop_str_in_output": False,
            "ignore_eos": False,
            "prompt_logprobs": None,
            "allowed_token_ids": None,
            "bad_words": [],

            "prompt_logprobs": None,         # Disable for speed
            "allowed_token_ids": None,       # Don't restrict (faster)
            "bad_words": [],                 # Empty list (no filtering overhead)
        }
    )
    return response.choices[0].message.content