# Newspaper OCR Pipeline
This project extracts structured data from newspaper page images using a multi-stage pipeline:
1. 🧼 **Preprocessing** — binarizes, deskews, and denoises images
2. 🧠 **OCR Extraction** — applies multiple OCR models (Tesseract, EasyOCR, etc.)
3. 🤖 **Postprocessing with LLMs** — uses Ollama-based models to structure text into CSV

---

## 🔁 Pipeline Overview

| Stage         | Description                                        | Output                            |
|---------------|----------------------------------------------------|-----------------------------------|
| Preprocessing | Enhances image quality for OCR                    | Cleaned `.tiff` images            |
| OCR           | Extracts raw text using multiple engines          | `.txt` files + log                |
| Postprocessing| Converts OCR text to structured CSV via LLMs      | `.csv` (or `.json`) per article   |

---

## 📦 Folder Structure

```
/data/images/                     → Raw input images
/results/images/preprocessed/     → Preprocessed image outputs
/results/txt/extracted/           → Raw OCR `.txt` or JSON
/results/csv/extracted/           → Final structured CSV output
/logs/                            → Processing logs
/scripts/                         → Core pipeline code
/docs/                            → Script-specific documentation
```

---

## 🚀 Quick Start

1. **Preprocess images**
```bash
python preprocess.py
```

2. **Run OCR with multiple engines**
```bash
python ocr.py
```

3. **Postprocess with LLMs**
```bash
python postprocess.py
```

4. **Run unit tests**
```bash
python <script>.py --test
```

---

## 🧪 Requirements

- Python 3.10+
- OpenCV, NumPy, PaddleOCR, EasyOCR, docTR, keras-ocr
- Ollama (installed locally and running LLMs)

---

## 🧾 License
GPL License