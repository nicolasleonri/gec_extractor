from utils_preprocessing import read_image, save_image, show_image, get_image_files, to_grayscale
from skimage.filters import threshold_niblack
import numpy as np
import math
import time
import os
import itertools
import cv2
from pdf2image import convert_from_path
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock
import gc
import psutil
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

class Binarization:
    @staticmethod
    def basic(image):
        gray = to_grayscale(image)
        _, output = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return output

    @staticmethod
    def otsu(image, with_gaussian=False):
        gray = to_grayscale(image)
        _, output = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if with_gaussian == True:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, output = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return output

        return output

    @staticmethod
    def adaptive_mean(image):
        gray = to_grayscale(image)
        medblur = cv2.medianBlur(gray, 5)
        return cv2.adaptiveThreshold(medblur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def adaptive_gaussian(image):
        gray = to_grayscale(image)
        medblur = cv2.medianBlur(gray, 5)
        return cv2.adaptiveThreshold(medblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def yannihorne(image, show=False):
        gray = to_grayscale(image)
        mean, std = cv2.meanStdDev(gray)
        threshold = mean[0][0] + std[0][0]
        _, output = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        output = output.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=2)
        if show == True:
            output = cv2.normalize(output, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return output
        else:
            return output

    @staticmethod
    def niblack(image, show=False, window_size=25, k=-0.2):
        gray = to_grayscale(image)
        thresh_niblack = threshold_niblack(gray, window_size=window_size, k=k)
        binary = gray <= thresh_niblack

        output = binary.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, iterations=2)

        if show == True:
            output = cv2.normalize(output, None, 0, 255,
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return output
        else:
            return output


class SkewCorrection:
    @staticmethod
    def boxes(image):
        gray = to_grayscale(image)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate the image to correct skew
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def projection_profile(image):
        # TODO
        return None

    @staticmethod
    def hough_transform(image):
        def compute_skew(image):
            # load in grayscale:
            gray = to_grayscale(image)
            edges = cv2.Canny(gray, 50, 200)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
            angle = 0.0

            try:
                nb_lines = len(lines)
            except TypeError as e:
                print(f"COMMON ERROR: {e}")
            
            for line in lines:
                angle += math.atan2(line[0][3]*1.0 - line[0]
                                    [1]*1.0, line[0][2]*1.0 - line[0][0]*1.0)

            angle /= nb_lines*1.0

            return angle * 180.0 / np.pi

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        try:
            M = cv2.getRotationMatrix2D(center, compute_skew(image), 1.0)
        except Exception as e:
            print(f"NEW ERROR: {e}")
            return image
            
        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate the image to correct skew
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def moments(image):
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # Calculate image moments
        moments = cv2.moments(binary)

        # Compute the skew angle
        skew_angle = 0.5 * \
            np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
        skew_angle = np.degrees(skew_angle)

        # Rotate the image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate the image to correct skew
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def topline(image):
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Compute the horizontal projection profile
        horizontal_projection = np.sum(binary, axis=1)

        # Find the first significant row (topline)
        topline_idx = np.argmax(horizontal_projection)

        # Calculate the skew angle based on topline
        angle = np.arctan2(topline_idx - binary.shape[0] // 2, binary.shape[1])
        angle = np.degrees(angle)

        # Rotate the image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate the image to correct skew
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def scanline(image):
        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int8(box)

        # Compute the angle of the bounding box
        angle = rect[-1]

        # Rotate the image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Adjust the bounding box size to fit the entire rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate the image to correct skew
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


class NoiseRemoval:
    @staticmethod
    def mean_filter(image, kernel_size=3):
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def gaussian_filter(image, kernel_size=3, sigma=0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    @staticmethod
    def median_filter(image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def conservative_filter(image, kernel_size=3):
        pad_size = kernel_size // 2
        padded = np.pad(image, pad_size, mode='edge')
        result = np.zeros_like(image)

        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                region = padded[i - pad_size:i + pad_size +
                                1, j - pad_size:j + pad_size + 1]
                min_val = np.min(region)
                max_val = np.max(region)
                if image[i - pad_size, j - pad_size] < min_val:
                    result[i - pad_size, j - pad_size] = min_val
                elif image[i - pad_size, j - pad_size] > max_val:
                    result[i - pad_size, j - pad_size] = max_val
                else:
                    result[i - pad_size, j - pad_size] = image[i -
                                                               pad_size, j - pad_size]
        return result

    @staticmethod
    def laplacian_filter(image):
        laplacian = cv2.Laplacian(image, cv2.CV_8U)
        inverted_laplacian = 255 - laplacian  # Invert the Laplacian result
        return inverted_laplacian

    @staticmethod
    def frequency_filter(image):
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        return img_back

    @staticmethod
    def crimmins_speckle_removal(image):
        output = image.copy().astype(np.int32)
        # Total number of iterations
        total_iterations = 2 * (image.shape[0] - 2) * (image.shape[1] - 2)

        current_iteration = 0
        for _ in range(2):
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    current_iteration += 1
                    # if current_iteration == total_iterations // 4:
                    #     print("Loading: 25%")
                    # elif current_iteration == total_iterations // 2:
                    #     print("Loading: 50%")
                    # elif current_iteration == 3 * (total_iterations // 4):
                    #     print("Loading: 75%")

                    current_pixel = output[i, j]
                    neighbors = [output[i-1, j], output[i+1, j],
                                 output[i, j-1], output[i, j+1]]
                    med = np.median(neighbors)
                    if abs(current_pixel - med) > abs(current_pixel - np.mean(neighbors)):
                        output[i, j] = med

        # print("Loading: 100%")
        return output.astype(np.uint8)

    @staticmethod
    def unsharp_filter(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened


def process_image_configuration(args):
    """
    Process a single image with a single configuration.
    This function will be run in parallel.
    """
    image_data, image_file, config, idx, processed_dir = args
    
    try:
        processed_image = image_data.copy()  # Work on a copy
        techniques = []
        
        start_time = time.time()
        
        # Apply each preprocessing step
        for step, method in config["preprocess"]:
            processed_image = getattr(step, method)(processed_image)
            techniques.append(f"{step.__name__}.{method}")
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        
        # Save the processed image
        filepath = os.path.join(processed_dir, f"{image_file.stem}_config{idx}.tiff")
        save_image(processed_image, filepath)
        
        # Create log entry
        log_entry = f"{image_file.name} - Time needed: {time_elapsed:.4f}s - Config {idx}: {', '.join(techniques)}\n"
        
        del processed_image
        gc.collect()
        del image_data

        return {
            'success': True,
            'log_entry': log_entry,
            'config_idx': idx,
            'techniques': techniques,
            'time_elapsed': time_elapsed,
            'image_name': image_file.name
        }
        
    except Exception as e:
        error_msg = f"ERROR processing {image_file.name} with config {idx}: {str(e)}\n"
        return {
            'success': False,
            'log_entry': error_msg,
            'config_idx': idx,
            'error': str(e),
            'image_name': image_file.name
        }

def process_single_image_all_configs(image_file, configurations, processed_dir, max_workers=None):
    """
    Process a single image with all configurations using threading.
    """
    print(f"Processing: {image_file.name}")
    
    # Read image once
    try:
        image = read_image(image_file)
    except Exception as e:
        print(f"Error reading {image_file.name}: {e}")
        return []
    
    # Prepare arguments for all configurations for this image
    args_list = [
        (image, image_file, config, idx, processed_dir) 
        for idx, config in enumerate(configurations)
    ]
    
    results = []
    completed_count = 0
    total_configs = len(configurations)
    
    # Use ProcessPoolExecutor for I/O bound operations (image saving)
    # For CPU-bound operations, you might want to use ProcessPoolExecutor instead
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(process_image_configuration, args): args 
            for args in args_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_args):
            result = future.result()
            results.append(result)
            completed_count += 1
            
            # if result['success']:
            #     print(f"✓ Config {result['config_idx']}: {', '.join(result['techniques'])} ({result['time_elapsed']:.3f}s)")
            # else:
            #     print(f"✗ Config {result['config_idx']}: {result['error']}")
            
            # Progress update
            if completed_count % 5 == 0 or completed_count == total_configs:
                print(f"Progress: {completed_count}/{total_configs} configurations completed")
    
    print(f"Completed processing {image_file.name}")
    print("-" * 50)
    return results

def print_help():
    help_text = """
    Preprocessing Pipeline Help
    --------------------------

    This script runs a comprehensive image preprocessing pipeline for OCR, testing multiple combinations of:

    Binarization Methods:
        - Basic thresholding
        - Otsu's method (with/without Gaussian)
        - Adaptive Mean
        - Adaptive Gaussian
        - Yannihorne method
        - Niblack method (configurable window size and k value)

    Skew Correction Methods:
        - Bounding boxes
        - Hough transform
        - Moments
        - Topline detection
        - Scanline detection

    Noise Removal Methods:
        - Mean filter
        - Gaussian filter
        - Median filter
        - Conservative filter
        - Laplacian filter
        - Frequency domain filter
        - Crimmins speckle removal
        - Unsharp masking

    Usage:
        python preprocessing.py              # Run full pipeline with all combinations based on (default) data folder "./data/images/"
        python preprocessing.py --help/--h/-h     # Show this help message

    Input/Output:
        - Input images should be in ./data/images/
        - Processed images will be saved in ./results/images/preprocessed/
        - Processing log will be saved as ./logs/preprocess.out
    """
    print(help_text)

def main():
    gc.enable()

    # Parse command line arguments
    max_threads = mp.cpu_count()
    processing_mode = "per-image"  # default mode

    if len(sys.argv) > 1:
        if any(arg in ['--help', '--h', '-h'] for arg in sys.argv):
            print_help()
            return
        
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == '--threads' and i + 1 < len(sys.argv):
                try:
                    max_threads = int(sys.argv[i + 1])
                except ValueError:
                    print("Invalid thread count. Using default.")
            elif arg == '--per-image':
                processing_mode = "per-image"
            elif arg == '--global':
                processing_mode = "global"

    print(f"Using {max_threads} threads in {processing_mode} mode")

    # Define preprocessing methods
    binarization_methods = ["basic", "otsu", "adaptive_mean", "adaptive_gaussian", "yannihorne", "niblack"] #  
    skew_correction_methods = ["boxes", "hough_transform", "topline", "scanline", "moments"] #  
    noise_removal_methods = ["mean_filter", "gaussian_filter", "median_filter", "conservative_filter", "crimmins_speckle_removal", "laplacian_filter", "frequency_filter", "unsharp_filter"] #  

    # Generate all possible configurations
    configurations = [
        {
            "preprocess": [
                (Binarization, bin_method),
                (SkewCorrection, skew_method),
                (NoiseRemoval, noise_method)
            ]
        }
        for bin_method, skew_method, noise_method in itertools.product(
            binarization_methods, skew_correction_methods, noise_removal_methods
        )
    ]

    print(f"Total configurations to test: {len(configurations)}")
    
    # Setup directories
    image_files = get_image_files("./data/images/")
    processed_dir = "./results/images/preprocessed/"
    log_file_path = os.path.join("./logs/", "preprocess.out")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Filter out PDF files (as in original)
    image_files = [f for f in image_files if "pdf" not in str(f)]
    print(f"Processing {len(image_files)} images")
    
    start_time = time.time()
    all_results = []
    
    if processing_mode == "per-image":
        # Process each image sequentially, but all configs for each image in parallel
        for image_file in image_files:
            results = process_single_image_all_configs(
                image_file, configurations, processed_dir, max_threads
            )
            all_results.extend(results)

    elif processing_mode == "global":
        # Process all image/configuration combinations globally in parallel
        print("Processing all combinations globally in parallel...")
        
        # Prepare all tasks
        all_tasks = []
        for image_file in image_files:
            try:
                image = read_image(image_file)
                for idx, config in enumerate(configurations):
                    all_tasks.append((image, image_file, config, idx, processed_dir))
            except Exception as e:
                print(f"Error reading {image_file.name}: {e}")
        
        print(f"Total tasks: {len(all_tasks)}")
        
        # Process all tasks
        with ProcessPoolExecutor(max_workers=max_threads) as executor:
            future_to_task = {
                executor.submit(process_image_configuration, task): task 
                for task in all_tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                all_results.append(result)
                completed += 1
                
                if completed % 5 == 0:
                    print(f"Progress: {completed}/{len(all_tasks)} tasks completed")

    # Write all results to log file
    with open(log_file_path, 'w') as log_file:
        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]
        
        # Write successful results
        for result in successful_results:
            log_file.write(result['log_entry'])
        
        # Write failed results
        if failed_results:
            log_file.write("\n--- ERRORS ---\n")
            for result in failed_results:
                log_file.write(result['log_entry'])
    
    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    successful_count = len([r for r in all_results if r['success']])
    failed_count = len([r for r in all_results if not r['success']])
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful processes: {successful_count}")
    print(f"Failed processes: {failed_count}")
    print(f"Average time per successful process: {total_time/max(successful_count, 1):.3f} seconds")
    print(f"Processing log saved to: {log_file_path}")
    print("="*60)

if __name__ == "__main__":
    main()
