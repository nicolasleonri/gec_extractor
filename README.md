# Newspaper OCR Pipeline
This project extracts structured data from newspaper page images using a multi-stage pipeline:
1. 🧼 **Preprocessing** — binarizes, deskews, and denoises images
2. 🧠 **OCR Extraction** — applies multiple OCR models (Tesseract, etc.)
3. 🤖 **Postprocessing with LLMs** — uses Ollama-based models to structure text into CSV
4. 📊 **Evaluation** - automatically compares extracted CSV-files with gold standards
---

## 🔁 Pipeline Overview

| Stage         | Description                                        | Output                            |
|---------------|----------------------------------------------------|-----------------------------------|
| Preprocessing | Enhances image quality for OCR                    | Cleaned `.tiff` images            |
| OCR           | Extracts raw text using multiple engines          | `.txt` files + log                |
| Postprocessing| Converts OCR text to structured CSV via LLMs      | `.csv` (or `.json`) per article   |
| Evaluation    | Compares generated CSV to gold labels             | `.csv` with results               |

---

## 📦 Folder Structure

```
/data/images/                     → Raw input images
/results/images/preprocessed/     → Generated preprocessed image outputs
/results/txt/extracted/           → Generated raw OCR `.txt` or `.json`
/results/csv/extracted/           → Final structured CSV output
/logs/                            → Saved logs
/scripts/                         → Core pipeline scripts
/docs/                            → Script-specific documentation
/requirements/*.txt               → Dependencies for each script
```

---

## 🚀 Quick Start

0. **Install dependencies**
```bash
pip3 install -r requirements/<script>_requirements.txt
```

1. **Run scripts independently**
```bash
python3 src/preprocess/preprocess.py
python3 src/ocr/ocr.py
python3 src/postprocess/postprocess.py
python3 src/evaluate/evaluate.py
```

                **OR:**

1. **Run Pipeline in a single Bash-script**
```bash
chmod +x run_ocr_pipeline.sh
./run_ocr_pipeline.sh
```

## 📢 Get quick help

1. **Get help for each script**
```bash
python <script>.py --help
```

2. **Run unit tests**
```bash
python <script>.py --test
```
---

## 🧪 Requirements

- Python 3.10+
- Tesseract (installed locally)
- Ollama (installed locally and running LLMs)
- Dependencies (see `/requirements/*.txt`)

---

## 🧾 License
GPL License