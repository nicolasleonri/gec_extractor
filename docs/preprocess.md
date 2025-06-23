# preprocess.py

Applies image enhancement techniques to prepare newspaper images for OCR.

---

## 📥 Input
- Raw `.jpg`, `.png`, or `.tiff` images in `./data/images/`

## 📤 Output
- Cleaned images saved to `./results/images/preprocessed/`
- Log file: `./logs/preprocess.out`

---

## ⚙️ Key Features
- Runs multiple preprocessing configurations:
  - Binarization (Otsu, Niblack, etc.)
  - Skew correction (Hough, boxes, moments)
  - Noise filters (median, conservative, unsharp, etc.)
- Parallel processing (per-image or global mode)
- `--test` flag runs unit tests

---

## 🧪 Example
```bash
python preprocess.py --threads 4 --per-image
python preprocess.py --test
```