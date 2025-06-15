import re
import os
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from ollama import chat, ChatResponse
from utils_postprocessing import parse_ocr_results
import multiprocessing as mp


class ollama:
    def __init__(self, model_name):
        self.model_name = model_name

    def chat_completion(self, prompt, question):
        response: ChatResponse = chat(
            model=self.model_name, 
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": question}],
            options={"temperature": 0},
        )
        response = response['message']['content']
        return response

    def extract_test_results(self, ocr_text):
        prompt = """
        CONTEXT: You are an expert in analyzing blood test results in a laboratory. Your work is extremely important and will be used in a life or death situation.  

        TASK: Given the following text extracted from a blood test report, please extract and structure the following information:
        - Date of the test
        - Patient information (if available)
        - Test results, including:
            - Test name
            - Result value
            - Unit of measurement
            - Reference range (if available)

        Format the output as a JSON object. If a piece of information is not available, use null for its value.

        EXAMPLE:
        {
        "Date": "XX.XX.XXX",
        "patient_information": {
            "patient_id": [FILL],
            "patient_name": "[FILL]",
            "patient_sex": "[FILL]",
            "patient_age": [FILL]
            },
        "test_results": [
            {
                "test_name": "[FILL]",
                "result_value": [FILL],
                "unit_of_measurement": "[FILL]",
                "reference_range": "[FILL]"
            }
            ]
        }
        
        IMPORTANT: Just return the JSON object. Do not make any further comment. Otherwise, the patient will die.
        IMPORTANT: Avoid the following error "JSON Parse ERROR: Expecting value: line 1 column 1 (char 0)" as this is a typical error.
        """
        response = self.chat_completion(prompt, ocr_text)

        return response

    # def read_txt_file(self, file_path):
    #     ocr_results = parse_ocr_results(os.path.join(os.getcwd(), file_path))

    #     for key, value in ocr_results.items():
    #         print(f"Prompting extracted text from {value['filepath']} using module {value['config']}:")
    #         start = time.time()
    #         structured_results = self.extract_test_results(value['text'])
    #         ocr_results[key]["Output"] = structured_results
    #         end = time.time()
    #         time_elapsed = end - start
    #         print(f"Time nedded: {round(time_elapsed, 5)}")

    #     return ocr_results

def process_single_result(key, value, model_name, model_display_name, progress_queue):
    """Process a single OCR result with LLM"""
    try:
        # Create LLM instance for this thread
        llm = ollama(model_name=model_name)
        
        start = time.time()
        structured_results = llm.extract_test_results(value['text'])
        end = time.time()
        time_elapsed = end - start
        
        # Prepare result
        result = {
            'key': key,
            'value': value,
            'output': structured_results,
            'time_elapsed': time_elapsed,
            'model_name': model_display_name,
            'success': True,
            'error': None
        }
        
        # Send progress update
        progress_queue.put(result)
        
        return result
        
    except Exception as e:
        error_result = {
            'key': key,
            'value': value,
            'output': None,
            'time_elapsed': 0,
            'model_name': model_display_name,
            'success': False,
            'error': str(e)
        }
        
        progress_queue.put(error_result)
        return error_result


def progress_monitor(progress_queue, total_tasks, results_dict, lock):
    """Monitor progress and handle results"""
    completed = 0
    
    while completed < total_tasks:
        try:
            result = progress_queue.get(timeout=1)
            completed += 1
            
            # Thread-safe update of results
            with lock:
                results_dict[result['key']] = result
            
            # Print progress
            percentage = round(completed / total_tasks * 100, 1)
            if result['success']:
                print(f"✓ Progress: {completed}/{total_tasks} ({percentage}%) - "
                      f"Processed {result['value']['filepath']} with {result['model_name']} "
                      f"in {round(result['time_elapsed'], 2)}s")
            else:
                print(f"✗ Progress: {completed}/{total_tasks} ({percentage}%) - "
                      f"Failed {result['value']['filepath']} with {result['model_name']} "
                      f"- Error: {result['error']}")
            
            progress_queue.task_done()
            
        except:
            continue  # Timeout, continue waiting

        
def save_results_to_json(results_dict, model_display_name):
    """Save all results to JSON files"""
    saved_count = 0
    error_count = 0
    
    for key, result in results_dict.items():
        if not result['success']:
            error_count += 1
            print(f"Skipping failed result for {result['value']['filepath']}")
            continue
            
        try:
            value = result['value']
            filename = os.path.splitext(os.path.basename(value['filepath']))[0]
            config_name = value["config"]
            pathfile = f"{filename}_{config_name}_{model_display_name}"
            file_name_json = f"./results/txt/extracted/{pathfile}.json"
            
            # Parse JSON output
            try:
                json_object = json.loads(result['output'])
            except (ValueError, json.JSONDecodeError) as e:
                print(f"JSON Parse ERROR for {value['filepath']}: {e}")
                # print(f"Original answer: {result['output']}")
                error_count += 1
                continue
            
            # Save to file
            with open(file_name_json, 'w', encoding='utf-8') as f:
                json.dump(json_object, f, ensure_ascii=False, indent=4)
            
            print(f"✓ Results saved: {file_name_json}")
            saved_count += 1
            
        except Exception as e:
            print(f"Save ERROR for {result['value']['filepath']}: {e}")
            error_count += 1
    
    return saved_count, error_count

def process_model_multithreaded(model_name, model_display_name, ocr_results, max_workers=3):
    """Process all OCR results for a single model using multithreading"""
    
    print(f"\n{'='*60}")
    print(f"Starting multithreaded processing for model: {model_display_name}")
    print(f"Total OCR results to process: {len(ocr_results)}")
    print(f"Using {max_workers} worker threads")
    print(f"{'='*60}")
    
    # Create progress tracking
    progress_queue = Queue()
    results_dict = {}
    lock = threading.Lock()
    
    # Start progress monitor thread
    progress_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_queue, len(ocr_results), results_dict, lock)
    )
    progress_thread.start()
    
    # Process all results in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Submit all tasks
        for key, value in ocr_results.items():
            future = executor.submit(
                process_single_result,
                key, value, model_name, model_display_name, progress_queue
            )
            futures.append(future)
        
        # Wait for all tasks to complete
        completed_tasks = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                completed_tasks += 1
            except Exception as e:
                completed_tasks += 1
                print(f"Task execution error: {e}")
    
    # Wait for progress monitor to finish
    progress_thread.join()
    
    processing_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Completed processing for {model_display_name}")
    print(f"Total processing time: {round(processing_time, 2)} seconds")
    print(f"Average time per result: {round(processing_time / len(ocr_results), 2)} seconds")
    
    # Save results to JSON files
    print(f"\nSaving results to JSON files...")
    saved_count, error_count = save_results_to_json(results_dict, model_display_name)
    
    print(f"✓ Successfully saved: {saved_count} files")
    print(f"✗ Errors encountered: {error_count} files")
    print(f"{'='*60}")
    
    return results_dict

def main():
    max_threads = mp.cpu_count()
        
    models = {
        # "phi4:latest": "phi4",
        # "llama4:latest": "llama4",
        "gemma3:27b": "gemma3",
        # "qwen3:32b": "qwen3",
        # "deepseek-r1:32b": "deepseek-r1",
        # "magistral:24b": "magistral",
    }
    
    print("Starting multithreaded LLM postprocessing...")
    print(f"Models to process: {list(models.values())}")
    
    # Load OCR results once
    print("\nLoading OCR results...")
    ocr_results = parse_ocr_results(os.path.join(os.getcwd(), "./results/txt/extracted/ocr_results_log.txt"))
    print(f"Loaded {len(ocr_results)} OCR results")
    
    # Process each model
    all_results = {}
    total_start_time = time.time()
    
    for model_name, model_display_name in models.items():
        try:
            model_results = process_model_multithreaded(
                model_name, 
                model_display_name, 
                ocr_results, 
                max_workers=max_threads  # Adjust based on your system capabilities
            )
            all_results[model_display_name] = model_results
            
        except Exception as e:
            print(f"Error processing model {model_display_name}: {e}")
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total execution time: {round(total_time, 2)} seconds")
    print(f"Models processed: {len(all_results)}")
    print(f"Total OCR results per model: {len(ocr_results)}")
    print(f"Total processing tasks: {len(models) * len(ocr_results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
