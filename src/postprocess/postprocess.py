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
import typing as T
from csv import DictReader
from io import StringIO
import csv

def extract_code_block(text: str, language_hint: str = "") -> str:
    # 1. Try to match a code block with the language hint
    if language_hint:
        pattern_lang = rf"```{language_hint}\n(.*?)```"
        match = re.search(pattern_lang, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 2. Try to match any code block (with or without language)
    pattern_any = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_any, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Fallback: assume the whole response *is* the output
    return text.strip()

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

    def extract_test_results_as_csv(self, ocr_text):
        prompt = """
        GOAL: Given the raw text output from an OCR-Engine, extract and structure the following information:
        - headline of the article: string or NA
        - subheadline of the article: string or NA
        - author of the article:: string or NA
        - content of the article:: string

        Focus on extracting articles (small or long) with actual information. Exclude any brief news items like: date, weather, public announcements, or other short notices that do not contain substantial content. Ask yourself if the content is relevant for media and discourse anaylisis. If any field is missing or unavailable, use "NA" for its value.

        RETURN FORMAT: Format the output strictly as a pythonic dictionary with keys and values like the following example:
        "headline";"subheadline";"author";"content"
        "El loco del martillo";"NA";"La Seño María";"Hoy en día, uno pensaría que..."
        "Contento por fin de cuarentena";"Habla Trome";"Ismael Lazo, Vecino de San Luis";"Estoy feliz porque..."

        WARNING: Only return the CSV exactly as specified. Do not add any explanation or commentary.
        WARNING: Avoid CSV parsing errors like empty or malformed outputs. After the header, each row should contain the extracted information for one article, with fields separated by semicolons and declared in quotation marks.

        CONTEXT: You are an expert in analyzing extracted newspaper content. Your task is to carefully extract articles and brief news items from the provided text. If you fail in this task, you will lose your job and your family will be very disappointed in you.
        """ 

        response = self.chat_completion(prompt, ocr_text)

        return response

    def extract_test_results_as_json(self, ocr_text):
        prompt = """
        CONTEXT: You are an expert in analyzing extracted newspaper content. Your task is to carefully extract articles and brief news items from the provided text.

        TASK: Given the extracted content, please extract and structure the following information into a JSON object containing only:

        - articles: a list of article objects, each with:
            - headline (string)
            - subheadline (string or null)
            - byline (string or null)
            - content (string)

        Focus on extracting articles (small or long) with actual information. Exclude any brief news items like: date, weather, public announcements, or other short notices that do not contain substantial content.

        If any field is missing or unavailable, use null for its value.

        Format the output strictly as a JSON object like the example below.

        EXAMPLE:
        {
        "articles": [
            {
            "headline": "El loco del martillo",
            "subheadline": "La Seño María",
            "byline": null,
            "content": "Full article content here..."
            },
            {
            "headline": "Contento por fin de cuarentena",
            "subheadline": "Habla Trome",
            "byline": "Ismael Lazo, Vecino de San Luis",
            "content": "Brief news content here..."
            }
            // More articles and briefs...
        ]
        }

        IMPORTANT: Only return the JSON object exactly as specified. Do not add any explanation or commentary.
        IMPORTANT: Avoid JSON parsing errors like empty or malformed outputs.
        """
        response = self.chat_completion(prompt, ocr_text)

        return response

    def read_txt_file(self, file_path):
        ocr_results = parse_ocr_results(os.path.join(os.getcwd(), file_path))

        for key, value in ocr_results.items():
            print(f"Prompting extracted text from {value['filepath']} using module {value['config']}:")
            start = time.time()
            structured_results = self.extract_test_results_as_json(value['text'])
            ocr_results[key]["Output"] = structured_results
            end = time.time()
            time_elapsed = end - start
            print(f"Time nedded: {round(time_elapsed, 5)}")

        return ocr_results

def process_single_result(key, value, model_name, model_display_name, progress_queue):
    """Process a single OCR result with LLM"""
    try:
        # Create LLM instance for this thread
        llm = ollama(model_name=model_name)
        
        start = time.time()
        # structured_results = llm.extract_test_results_as_json(value['text'])
        structured_results = llm.extract_test_results_as_csv(value['text'])
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

class Article(T.NamedTuple):
    headline: str
    subheadline: str
    author: str
    content: str
    
    @classmethod
    def from_row(cls, row: dict):
        return cls(**{
            key: type_(row[key]) for key, type_ in cls._field_types.items()
        })
    
def validate_csv(reader: DictReader) -> bool:
    for row in reader:
        try:
            Article.from_row(row)
        except Exception as e:
            print('type: {} msg: {}'.format(type(e), e))
            return False
    return True

def save_results_to_csv(results_dict, model_display_name):
    """Save all results to CSV files"""
    saved_count = 0
    error_count = 0

    for key, result in results_dict.items():
        try:
            value = result['value']
            filename = os.path.splitext(os.path.basename(value['filepath']))[0]
            config_name = value["config"]
            pathfile = f"{filename}_{config_name}_{model_display_name}"
            file_name_csv = f"./results/csv/extracted/{pathfile}.csv"

            metadata_parts = filename.split('_')[0].split('#')

            if len(metadata_parts) == 3:
                newspaper_name, publication_date, page_str = metadata_parts
                try:
                    page_number = int(page_str)
                except ValueError:
                    page_number = None
            else:
                newspaper_name = publication_date = None
                page_number = None

            document_metadata = {
                "newspaper_name": newspaper_name,
                "publication_date": publication_date,
                "page_number": page_number
            }

            data = extract_code_block(result['output'], language_hint="csv")

            f = StringIO(data)
            reader = csv.reader(f, delimiter=';', quotechar='"')

            # Save to CSV file
            with open(file_name_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(
                    f,
                    delimiter=';',
                    quotechar='"',
                    quoting=csv.QUOTE_ALL  # ← Force quotation around every field
                )
                
                # Extended header
                header = ["headline", "subheadline", "author", "content", "newspaper_name", "publication_date", "page_number"]
                writer.writerow(header)
                
                # Write data rows
                for row in reader:
                    if row == ["headline", "subheadline", "author", "content"]:
                        continue
                    if len(row) == 4:
                        full_row = row + [
                            document_metadata["newspaper_name"],
                            document_metadata["publication_date"],
                            document_metadata["page_number"]
                        ]
                        writer.writerow(full_row)

            # # Read file to validate CSV structure
            # with open(file_name_csv, 'r', encoding='utf-8') as f:
            #     f.seek(0)
            #     reader = csv.DictReader(f, delimiter=';', quotechar='"')
            #     for row in reader:
            #         print(row)

            print(f"✓ Results saved: {file_name_csv}")
            saved_count += 1
        except Exception as e:
            print(f"Error processing result for {value['filepath']}: {e}")
            error_count += 1
            continue

    return saved_count, error_count
        
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

            metadata_parts = filename.split('_')[0].split('#')

            if len(metadata_parts) == 3:
                newspaper_name, publication_date, page_str = metadata_parts
                try:
                    page_number = int(page_str)
                except ValueError:
                    page_number = None
            else:
                newspaper_name = publication_date = None
                page_number = None

            document_metadata = {
                "newspaper_name": newspaper_name,
                "publication_date": publication_date,
                "page_number": page_number
            }
            
            # Parse JSON output
            try:
                json_object = json.loads(result['output'])
            except (ValueError, json.JSONDecodeError) as e:
                print(f"JSON Parse ERROR for {value['filepath']}: {e}")
                # print(f"Original answer: {result['output']}")
                error_count += 1
                continue

            # Add document_metadata key if not already present
            if "metadata" not in json_object:
                json_object["document_metadata"] = document_metadata
            
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
    
    print(f"\nSaving results to CSV files...")
    saved_count, error_count = save_results_to_csv(results_dict, model_display_name)
    # saved_count, error_count = save_results_to_json(results_dict, model_display_name)
    
    print(f"✓ Successfully saved: {saved_count} files")
    print(f"✗ Errors encountered: {error_count} files")
    print(f"{'='*60}")
    
    return results_dict

def main():
    max_threads = mp.cpu_count()
        
    models = {
        "phi4:latest": "phi4",
        "llama4:latest": "llama4",
        "gemma3:27b": "gemma3",
        "qwen3:32b": "qwen3",
        "deepseek-r1:32b": "deepseek-r1",
        "magistral:24b": "magistral",
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
