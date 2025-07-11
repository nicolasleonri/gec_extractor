# preprocess.py

Applies image enhancement techniques to prepare image files for OCR.

---

## 📥 Input
- Raw images (e.g. `.png`, `.tiff`) in `./data/images/`

## 📤 Output
- Preprocessed images saved to `./results/images/preprocessed/`
- Log file: `./logs/preprocess.out`

---

## ⚙️ Key Features
- Runs multiple preprocessing configurations:
  - Binarization (Otsu, Niblack, etc.)
  - Skew correction (Hough, boxes, moments)
  - Noise filters (median, conservative, unsharp, etc.)
- Parallel processing (per-image or global mode)
- `--test` flag runs unit tests
- `--help` shows usage

---

## 🧪 Example
```bash
python preprocess.py --threads 4 --per-image
python preprocess.py --test
```