from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
from numpy import ndarray
import numpy as np
import argparse
import time
import cv2
import sys
import re
import os


class Binarization:
    """Provides several binarization methods for thresholding grayscale images."""
    @staticmethod
    def none(image: ndarray) -> ndarray:
        gray = to_grayscale(image)
        return gray

    @staticmethod
    def otsu(image: ndarray, with_gaussian: bool = False) -> ndarray:
        """Applies Otsu's thresholding, optionally with Gaussian blur.

        Args:
            image (ndarray): Input image.
            with_gaussian (bool): Whether to apply Gaussian blur before thresholding.

        Returns:
            ndarray: Binarized image.
        """
        gray = to_grayscale(image)
        src = cv2.GaussianBlur(gray, (5, 5), 0) if with_gaussian else gray
        _, output = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return output

    @staticmethod
    def adaptive_mean(image: ndarray) -> ndarray:
        """Applies adaptive mean thresholding after median blur.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Binarized image.
        """
        gray = to_grayscale(image)
        medblur = cv2.medianBlur(gray, 5)
        return cv2.adaptiveThreshold(medblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

class NoiseRemoval:
    """Provides several noise removal methods for correcting binarized images."""
    @staticmethod
    def none(image: ndarray) -> ndarray:
        return image

    """Provides multiple filtering techniques to reduce noise in document images."""
    @staticmethod
    def mean_filter(image: ndarray, kernel_size: int = 3) -> ndarray:
        """Applies a mean (box) filter to the image.

        Args:
            image (ndarray): Input image.
            kernel_size (int): Size of the kernel.

        Returns:
            ndarray: Smoothed image.
        """
        return cv2.blur(image, (kernel_size, kernel_size))
    
    @staticmethod
    def laplacian_filter(image: ndarray) -> ndarray:
        """Enhances edges using the Laplacian operator.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Edge-enhanced image.
        """
        laplacian = cv2.Laplacian(image, cv2.CV_8U)
        inverted_laplacian = 255 - laplacian # Invert to highlight dark edges on light background
        return inverted_laplacian

    @staticmethod
    def gaussian_filter(image: ndarray, kernel_size: int = 3, sigma: float = 0) -> ndarray:
        """Applies a Gaussian blur to the image.

        Args:
            image (ndarray): Input image.
            kernel_size (int): Size of the kernel (should be odd).
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            ndarray: Blurred image.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # @staticmethod
    # def crimmins_speckle_removal(image: ndarray) -> ndarray:
    #     """Reduces speckle noise using the Crimmins algorithm.

    #     Args:
    #         image (ndarray): Input image.

    #     Returns:
    #         ndarray: Denoised image.
    #     """
    #     output = image.copy().astype(np.int32)
    #     total_iterations = 2 * (image.shape[0] - 2) * (image.shape[1] - 2)

    #     current_iteration = 0
    #     for _ in range(2):
    #         for i in range(1, image.shape[0] - 1):
    #             for j in range(1, image.shape[1] - 1):
    #                 current_iteration += 1
    #                 current_pixel = output[i, j]
    #                 neighbors = [output[i-1, j], output[i+1, j],
    #                              output[i, j-1], output[i, j+1]]
    #                 med = np.median(neighbors)
    #                 if abs(current_pixel - med) > abs(current_pixel - np.mean(neighbors)):
    #                     output[i, j] = med

    #     return output.astype(np.uint8)

    @staticmethod
    def crimmins_speckle_removal(image: np.ndarray) -> np.ndarray:
        """Reduces speckle noise using the Crimmins algorithm (optimized)."""
        output = image.copy().astype(np.int32)
        
        for _ in range(2):
            # Extract inner region and all 4-connected neighbors using slicing
            inner = output[1:-1, 1:-1]
            neighbors = np.stack([
                output[0:-2, 1:-1],  # up
                output[2:, 1:-1],    # down  
                output[1:-1, 0:-2],  # left
                output[1:-1, 2:]     # right
            ], axis=0)
            
            # Vectorized median and mean calculation
            medians = np.median(neighbors, axis=0)
            means = np.mean(neighbors, axis=0)
            
            # Apply condition vectorized
            condition = np.abs(inner - medians) > np.abs(inner - means)
            output[1:-1, 1:-1] = np.where(condition, medians, inner)
        
        return output.astype(np.uint8)

def is_grayscale(image: ndarray) -> bool:
    """Checks whether the image is in grayscale format.

    Args:
        image (numpy.ndarray): Image to check.

    Returns:
        bool: True if grayscale, False otherwise.
    """
    return len(image.shape) == 2

def to_grayscale(image: ndarray) -> ndarray:
    """Converts an image to grayscale if it is not already.

    Args:
        image (numpy.ndarray): Color or grayscale image.

    Returns:
        numpy.ndarray: Grayscale image.
    """
    if not is_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image

def read_image(path: Union[str, Path]) -> Union[ndarray, None]:
    """Reads an image from the given file path using OpenCV.

    Args:
        path (str or Path): Path to the image file.

    Returns:
        numpy.ndarray: Image as a NumPy array, or None if it fails.
    """
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

def get_image_files(directory: Union[str, Path]) -> List[Path]:
    """Gets a list of image files from a given directory.

    Args:
        directory (str or Path): Directory to search for image files.

    Returns:
        list: Sorted list of Path objects pointing to image files.
    """
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp']
    image_files = []
    
    # Use rglob to recursively search all subdirectories
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            image_files.append(file)

    output = sorted(image_files)
    return output

def match(choice: str) -> None:
    dict = {
        "19": {
            "binarization": Binarization.otsu,
            "noise_removal": NoiseRemoval.mean_filter,
        },
        "24": {
            "binarization": Binarization.otsu,
            "noise_removal": NoiseRemoval.laplacian_filter,
        },
        "28": {
            "binarization": Binarization.adaptive_mean,
            "noise_removal": NoiseRemoval.mean_filter,
        },
        "2": {
            "binarization": Binarization.none,
            "noise_removal": NoiseRemoval.gaussian_filter,
        },
        "27": {
            "binarization": Binarization.adaptive_mean,
            "noise_removal": NoiseRemoval.none,
        },
        "5": {
            "binarization": Binarization.none,
            "noise_removal": NoiseRemoval.crimmins_speckle_removal,
        },
        "1": {
            "binarization": Binarization.none,
            "noise_removal": NoiseRemoval.mean_filter,
        },
    }

    match choice:
        case "trome":
            return dict["19"]
        case "ojo":
            return dict["24"]
        case "publimetro":
            return dict["28"]
        case "peru21":
            return dict["2"]
        case "elcomercio":
            return dict["27"]
        case "correo":
            return dict["1"]
        case "gestion":
            return dict["5"]
        case _:
            print("Newspaper not recognized. Available options: trome, ojo, publimetro, peru21, elcomercio, correo, gestion.")
            return None

def preprocess_image(image_path, config_list, newspaper) -> Tuple[Path, bool, Optional[str]]:
    """Preprocess a single image with the given configuration.
    
    Args:
        image_path: Path to the image file
        config_list: Dictionary containing binarization and noise_removal functions
        
    Returns:
        Tuple of (image_path, success, error_message)
    """
    try:
        image = read_image(image_path)
        if image is None:
            return (image_path, False, "Could not read image")
        
        binarization_func = config_list["binarization"]
        processed_image = binarization_func(image)
        
        noise_removal_func = config_list["noise_removal"]
        final_image = noise_removal_func(processed_image)
        
        numeric_parts = re.findall(r'/(\d+)', str(image_path.parent))
        date_path = '/'.join(numeric_parts)

        output_path = "results/images/preprocessed/" + str(newspaper) + "/" + date_path + "/"
        os.makedirs(output_path, exist_ok=True)

        if image_path.suffix.lower() != '.png':
            new_filename = image_path.stem + '.png'
        else:
            new_filename = image_path.name

        output_file = output_path + new_filename
        
        success = cv2.imwrite(str(output_file), final_image)

        if success:
            return (image_path, True, None)
        else:
            return (image_path, False, "Failed to save processed image")
            
    except Exception as e:
        return (image_path, False, str(e))

def process_images_multithreaded(image_paths, config_list, newspaper, max_workers) -> Dict:
    """Process multiple images using multithreading.
    
    Args:
        image_paths: List of image file paths
        config: Processing configuration dictionary
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary with processing results
    """
    results = {
        'success': [],
        'failed': [],
        'summary': {}
    }
    
    print(f"Processing {len(image_paths)} images using {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(preprocess_image, path, config_list, newspaper): path 
            for path in image_paths
        }
        
        for future in as_completed(future_to_path):
            image_path, success, error = future.result()
            
            if success:
                results['success'].append(str(image_path))
                print(f"✓ Processed: {image_path}")
            else:
                results['failed'].append((str(image_path), error))
                print(f"✗ Failed: {image_path} - {error}")
    
    results['summary'] = {
        'total': len(image_paths),
        'successful': len(results['success']),
        'failed': len(results['failed'])
    }
    
    print(f"\nSummary: {results['summary']['successful']}/{results['summary']['total']} images processed successfully")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Preprocessor for document images.')
    parser.add_argument('-n', '--newspaper', required=True, help='Newspaper name (required)')
    parser.add_argument('-f', '--folder_file', required=True, help='Input folder path (required)')
    
    args = parser.parse_args()

    # Reads important parameters
    workers = os.cpu_count()
    newspaper = args.newspaper.lower()
    img_list = get_image_files(str(args.folder_file))
    config_list = match(str(args.newspaper))

    time_start = time.time()
    results = process_images_multithreaded(img_list, config_list, str(newspaper), max_workers=int(workers))
    total_time = time.time() - time_start
    print(f"Processed {len(img_list)} images in {total_time:.2f} seconds")
    print(f"Avg per image: {total_time / len(img_list):.2f} sec")

if __name__ == "__main__":
    main()

