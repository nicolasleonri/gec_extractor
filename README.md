# OCR Pipeline Documentation

This project implements a comprehensive OCR pipeline with preprocessing, multiple OCR engines, postprocessing with LLMs, and evaluation capabilities.

## Pipeline Components

### 1. Preprocessing (`preprocessing.py`)

Image preprocessing module with multiple techniques for image enhancement.

#### Classes and Methods:

**Binarization**
- `basic(image)`: Basic thresholding at value 127
- `otsu(image, with_gaussian=False)`: 
  - Otsu's thresholding method
  - `with_gaussian`: Apply Gaussian blur before thresholding

- `adaptive_mean(image)`: Adaptive mean thresholding
- `adaptive_gaussian(image)`: Adaptive Gaussian thresholding
- `yannihorne(image, show=False)`:
  - Custom thresholding based on mean and standard deviation
  - `show`: Normalize output for visualization

- `niblack(image, show=False, window_size=25, k=-0.2)`:
  - Niblack thresholding method
  - `show`: Normalize output for visualization
  - `window_size`: Size of the local window
  - `k`: Weight factor for standard deviation

**SkewCorrection**
- `boxes(image)`: Correction using minimum area rectangle
- `hough_transform(image)`: Correction using Hough line detection
- `moments(image)`: Correction using image moments
- `topline(image)`: Correction based on text top line
- `scanline(image)`: Correction using scan lines

**NoiseRemoval**
- `mean_filter(image, kernel_size=3)`
- `gaussian_filter(image, kernel_size=3, sigma=0)`
- `median_filter(image, kernel_size=3)`
- `conservative_filter(image, kernel_size=3)`
- `laplacian_filter(image)`
- `frequency_filter(image)`
- `crimmins_speckle_removal(image)`
- `unsharp_filter(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0)`

### 2. OCR Processing (`ocr.py`)

Multiple OCR engine implementation with parallel processing capabilities.

#### OCR Classes:

**TesseractOCR**
- Simple text extraction without specific flags
- Uses system's Tesseract installation

**KerasOCR**
- GPU-accelerated OCR
- Methods:
  - `process(image)`: Basic text extraction
  - `finetune_detector(data_dir)`: For detector fine-tuning
  - `finetune_recognizer(data_dir)`: For recognizer fine-tuning

**EasyOCR**
- GPU-enabled with English language model
- Flags in `process(image)`:
  - Uses GPU by default
  - Language set to English

**PaddleOCR (PaddlePaddle)**
- Comprehensive OCR with multiple features
- Flags in `process(image)`:
  - `use_angle_cls=True`: Enable angle classification
  - `lang='en'`: English language model
  - `use_gpu=True`: GPU acceleration
  - `show_log=False`: Suppress logging

**docTR**
- Document Text Recognition
- Models:
  - Detection: 'db_resnet50'
  - Recognition: 'crnn_vgg16_bn'
  - `pretrained=True`: Use pretrained weights

### 3. Postprocessing (`postprocessing.py`)

LLM-based text structuring and information extraction.

#### AnyScaleLLM Class
- Initialization parameters:
  - `model_name`: LLM model identifier
  - `api_key`: AnyScale API key
  - `base_url`: API endpoint (default: "https://api.endpoints.anyscale.com/v1")

- Supported Models:
  - "meta-llama/Meta-Llama-3-8B-Instruct"
  - "mistralai/Mistral-7B-Instruct-v0.1"
  - "google/gemma-7b-it"

- Methods:
  - `chat_completion(prompt, question)`:
    - `temperature=0.1`: Low temperature for consistent outputs
  
  - `extract_test_results(ocr_text)`:
    - Structures OCR output into JSON format
    - Extracts: dates, patient info, test results

### 4. Evaluation (`evaluate.py`)

Performance evaluation and comparison of different OCR methods.

#### Metrics:
- Accuracy
- Precision
- Recall
- F1-score

#### Features:
- Fuzzy string matching for comparison
- Support for multiple OCR outputs
- JSON-based result storage

## Usage

Each script can be run independently:

```bash
# Preprocessing
python preprocessing.py

# OCR Processing
python ocr.py

# Postprocessing
python postprocessing.py

# Evaluation
python evaluate.py
```

## Directory Structure
```
.
├── data/
│   └── images/         # Input images
├── results/
│   ├── txt/           # OCR text outputs
│   ├── csv/           # Evaluation results
│   └── images/        # Processed images
└── requirements/       # Additional requirements
```

## Requirements

See `requirements.txt` for complete list of dependencies. 