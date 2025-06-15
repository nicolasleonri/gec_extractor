from utils_ocr import get_image_files
from paddleocr import PaddleOCR
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import torch
import time
import pytesseract
import keras_ocr
import easyocr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

class TesseractOCR:
    @staticmethod
    def process(image):
        result = pytesseract.image_to_string(image)
        result = result.replace("\n", " ")
        return result


class KerasOCR:
    @staticmethod
    def process(image):
        torch.cuda.empty_cache()
        pipeline = keras_ocr.pipeline.Pipeline()
        read_image = keras_ocr.tools.read(image)
        prediction_groups = pipeline.recognize([read_image])
        # prediction_groups is a list of (word, box) tuples
        output = [str(y[0]) for i in prediction_groups for y in i]
        return " ".join(output)

    def finetune_detector(data_dir):
        # https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_detector.html
        # https://www.kaggle.com/discussions/general/243859
        return None

    def finetune_recognizer(data_dir):
        # https://keras-ocr.readthedocs.io/en/latest/examples/fine_tuning_recognizer.html
        return None


class EasyOCR:
    @staticmethod
    def process(image):
        torch.cuda.empty_cache()
        reader = easyocr.Reader(['en'], gpu=True)
        result = reader.readtext(image)
        output = [j for _, j, _ in result]
        return " ".join(output)


class PaddlePaddle:
    @staticmethod
    def process(image):
        # need to run only once to download and load model into memory
        ocr = PaddleOCR(use_angle_cls=True, lang='en',
                     use_gpu=True, show_log=False)
        result = ocr.ocr(image, cls=True, det=True, rec=True)
        output = [j[1][0] for i in result for j in i]
        return " ".join(output)


class docTR:
    @staticmethod
    def extract_text_from_document(document):
        # Extract all the words and join them into a single string
        text = " ".join(
            word.value
            for page in document.pages
            for block in page.blocks
            for line in block.lines
            for word in line.words
        )
        return text

    @staticmethod
    def process(image):
        model = ocr_predictor(det_arch='db_resnet50',
                              reco_arch='crnn_vgg16_bn', pretrained=True)
        single_img_doc = DocumentFile.from_images(image)
        result = model(single_img_doc)
        output = docTR.extract_text_from_document(result)
        return output

def process_single_ocr(image_path, method_name, method, log_queue):
    """Process a single image with a single OCR method"""
    start = time.time()
    output = ""
    error_msg = ""
    
    try:
        output = method(image_path)
    except ValueError as ve:
        if "unable to read file" in str(ve):
            error_msg = f'COMMON ERROR: {ve}'
        else:
            error_msg = f'NEW ERROR: {ve}'
    except Exception as e:
        if "CUDA out of memory" in str(e):
            error_msg = f'COMMON ERROR: {e}'
        elif "can not handle images with 32-bit samples" in str(e):
            error_msg = f'COMMON ERROR: {e}'
        else:
            error_msg = f'NEW ERROR: {e}'
    
    end = time.time()
    time_elapsed = end - start
    
    # Create result dictionary
    result = {
        'image_file': os.path.basename(image_path),
        'method_name': method_name,
        'time_elapsed': time_elapsed,
        'output': output,
        'error_msg': error_msg
    }
    
    # Thread-safe logging
    log_queue.put(result)
    
    return result

def log_writer(log_queue, log_file_path, total_tasks):
    """Dedicated thread for writing logs"""
    completed_tasks = 0
    
    with open(log_file_path, 'w') as log_file:
        while completed_tasks < total_tasks:
            try:
                result = log_queue.get(timeout=1)
                
                # Write to log file
                log_file.write(f"File: {result['image_file']} - "
                             f"Time needed: {result['time_elapsed']} - "
                             f"Config: {result['method_name']} - "
                             f"[{result['output']}]\n")
                log_file.flush()
                
                # Print progress
                # print(f"Completed: {result['image_file']} with {result['method_name']} "
                #       f"in {round(result['time_elapsed'], 5)}s")
                
                if result['error_msg']:
                    print(f"Error: {result['error_msg']}")
                
                completed_tasks += 1
                log_queue.task_done()
                
            except:
                continue  # Timeout, continue waiting

def main():
    # Define the OCR classes and their corresponding process methods
    ocr_methods = {
        "TesseractOCR": TesseractOCR.process,
        # "KerasOCR": KerasOCR.process,
        # "EasyOCR": EasyOCR.process,
        # "PaddleOCR": PaddlePaddle.process,
        # "docTR": docTR.process
    }
    
    # Define the directory containing image files
    image_files = get_image_files("./results/images/preprocessed/")
    processed_dir = "./results/txt/extracted/"
    log_file_path = os.path.join(processed_dir, "ocr_results_log.txt")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create thread-safe queue for logging
    log_queue = Queue()
    
    # Calculate total number of tasks
    total_tasks = len(image_files) * len(ocr_methods)
    print(f"Starting processing of {len(image_files)} images with {len(ocr_methods)} methods")
    print(f"Total tasks: {total_tasks}")
    
    # Start log writer thread
    log_thread = threading.Thread(
        target=log_writer, 
        args=(log_queue, log_file_path, total_tasks)
    )
    log_thread.start()

    # Option 1: Process each image with all OCR methods in parallel
    # Good if you have enough memory and want maximum parallelism
    def process_by_image():
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust based on your system
            futures = []
            
            for image_file in image_files:
                image_path = os.path.join(os.getcwd(), image_file)
                # print(f"Submitting tasks for: {image_file}")
                
                for method_name, method in ocr_methods.items():
                    future = executor.submit(
                        process_single_ocr, 
                        image_path, 
                        method_name, 
                        method, 
                        log_queue
                    )
                    futures.append(future)
            
            completed = 0
            total_futures = len(futures)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    completed += 1
                    if completed % 5 == 0:
                        print(f"Progress: {completed}/{total_futures} tasks completed")
                except Exception as e:
                    completed += 1
                    print(f"Task failed with error: {e}")

    # Option 2: Process one method at a time across all images
    # Better for memory management if OCR methods are memory-intensive
    def process_by_method():
        for method_name, method in ocr_methods.items():
            print(f"Processing all images with {method_name}")
            
            with ThreadPoolExecutor(max_workers=2) as executor:  # Lower for memory safety
                futures = []
                
                for image_file in image_files:
                    image_path = os.path.join(os.getcwd(), image_file)
                    future = executor.submit(
                        process_single_ocr, 
                        image_path, 
                        method_name, 
                        method, 
                        log_queue
                    )
                    futures.append(future)
                
                completed = 0
                total_futures = len(futures)

                # Wait for current method to complete on all images
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        completed += 1
                        if completed % 5 == 0:
                            print(f"Progress: {completed}/{total_futures} tasks completed")
                    except Exception as e:
                        completed += 1
                        print(f"Task failed with error: {e}")
    
    # Choose your processing strategy
    start_time = time.time()
    
    process_by_image()  # Change this based on your preference
    
    # Wait for log writer to finish
    log_thread.join()
    
    total_time = time.time() - start_time    

    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Processing log saved to: {log_file_path}")
    print("="*60)
    

if __name__ == "__main__":
    main()