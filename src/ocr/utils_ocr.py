"""
utils_ocr.py

Utility functions for discovering image files for OCR processing.
"""
from pathlib import Path
from typing import List

def get_image_files(directory: str) -> List[Path]:
    """Returns a list of supported image files from the given directory.

    Args:
        directory (str): Path to the folder containing image files.

    Returns:
        List[Path]: A list of valid image file paths.
    """
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']
    image_files = []
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file)
    return image_files