"""
postprocess.py

This script performs postprocessing on raw OCR-extracted newspaper text using LLMs via Ollama.
It converts unstructured text into structured CSV files, extracting articles, headlines, subheadlines,
authors, and content for each image-configuration pair.

Supports multithreaded processing and multiple model runs.

Author: @nicolasleonri (GitHub)
License: GPL
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils_postprocessing import parse_ocr_results, extract_filename_and_config, log_processing_info
from ollama import chat, ChatResponse
import multiprocessing as mp
from csv import DictReader
from io import StringIO
from queue import Queue
import typing as T
import threading
import unittest
import time
import json
import sys
import csv
import re
import os

def extract_code_block(text: str, language_hint: str = "") -> str:
    """Extracts a code block (e.g., CSV) from a markdown-formatted LLM response.

    Args:
        text (str): Full response string from LLM.
        language_hint (str, optional): Language label to look for (e.g., "csv").

    Returns:
        str: Cleaned code block string (e.g., CSV content).
    """
    if language_hint:
        pattern_lang = rf"```{language_hint}\n(.*?)```"
        match = re.search(pattern_lang, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    pattern_any = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern_any, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

class ollama:
    """Wrapper around Ollama-based LLMs for structured postprocessing of OCR content."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name (str): Name of the LLM model to use via Ollama.
        """
        self.model_name = model_name

    def chat_completion(self, prompt: str, question: str) -> str:
        """Sends a chat completion request to Ollama.

        Args:
            prompt (str): System-level instruction.
            question (str): User query containing OCR text.

        Returns:
            str: LLM response content.
        """
        response: ChatResponse = chat(
            model=self.model_name, 
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": question}],
            options={"temperature": 0.1},
        )
        response = response['message']['content']
        return response

    def extract_test_results_as_csv(self, ocr_text: str) -> str:
        """Uses an LLM to extract OCR content into CSV-formatted string.

        Args:
            ocr_text (str): Raw OCR text.

        Returns:
            str: CSV string in markdown code block format.
        """

        #TODO: Try different prompts in different languages
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

    def extract_test_results_as_json(self, ocr_text: str) -> str:
        """Uses an LLM to extract OCR content into a structured JSON string.

        Args:
            ocr_text (str): Raw OCR text.

        Returns:
            str: JSON string with article metadata and content.
        """

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

    def read_txt_file(self, file_path: str) -> dict:
        """Reads OCR log file and prompts each result using the LLM.

        Args:
            file_path (str): Path to the OCR log output.

        Returns:
            dict: Dictionary of enriched OCR results keyed by index.
        """
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

def process_single_result(key: int, value: dict, model_name: str, model_display_name: str, progress_queue: Queue, log_file_path = str) -> dict:
    """Processes one OCR result with a specified LLM model.

    Args:
        key (int): Dictionary key for the OCR result.
        value (dict): OCR result metadata and text.
        model_name (str): LLM model identifier.
        model_display_name (str): Short label for display/logging.
        progress_queue (Queue): Shared queue for reporting progress.

    Returns:
        dict: Result including extracted data, runtime, and status.
    """

    try:
        llm = ollama(model_name=model_name) # Create LLM instance for this thread
        
        start = time.time()
        # structured_results = llm.extract_test_results_as_json(value['text'])
        structured_results = llm.extract_test_results_as_csv(value['text'])
        end = time.time()
        time_elapsed = end - start
        ocr_name = str(value['ocr'])
        config_no = str(value['config'])
        processed_filename, config_number = extract_filename_and_config(value['filepath'])
        log_processing_info(log_file_path, processed_filename, config_no, ocr_name, model_display_name, time_elapsed)
        
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
        
        progress_queue.put(result) # Send progress update
        
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


def progress_monitor(progress_queue: Queue, total_tasks: int, results_dict: dict, lock: threading.Lock) -> None:
    """Consumes result objects from the queue and tracks processing progress.

    Args:
        progress_queue (Queue): Queue receiving LLM task results.
        total_tasks (int): Total number of expected results.
        results_dict (dict): Shared dictionary storing all results.
        lock (threading.Lock): Thread lock for safe writes.
    """
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
    """Typed representation of a structured newspaper article."""
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
    """Validates that a CSV reader produces rows compatible with Article.

    Args:
        reader (DictReader): CSV reader object.

    Returns:
        bool: True if all rows are valid Article entries; False otherwise.
    """
    for row in reader:
        try:
            Article.from_row(row)
        except Exception as e:
            print('type: {} msg: {}'.format(type(e), e))
            return False
    return True

def save_results_to_csv(results_dict: dict, model_display_name: str) -> T.Tuple[int, int]:
    """Saves structured OCR results (CSV-formatted strings) into CSV files with metadata.

    Args:
        results_dict (dict): Dictionary of processed results.
        model_display_name (str): Short model name to include in filenames.

    Returns:
        Tuple[int, int]: (successful saves, errors encountered)
    """
    saved_count = 0
    error_count = 0

    for key, result in results_dict.items():
        try:
            value = result['value']
            filename = os.path.splitext(os.path.basename(value['filepath']))[0]
            ocr_name = value["ocr"]
            config_no = value['config']
            pathfile = f"{filename}_config{config_no}_{ocr_name}_{model_display_name}"
            file_name_csv = f"./results/csv/extracted/{pathfile}.csv"

            # Extract metadata from filename (e.g., newspaper#date#page_imgXX)
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

            # Extract CSV content from LLM response
            data = extract_code_block(result['output'], language_hint="csv")
            f = StringIO(data)
            reader = csv.reader(f, delimiter=';', quotechar='"')

            # Write CSV with extended metadata columns
            with open(file_name_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(
                    f,
                    delimiter=';',
                    quotechar='"',
                    quoting=csv.QUOTE_ALL  # Force quotation around every field
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

            print(f"✓ Results saved: {file_name_csv}")
            saved_count += 1
        except Exception as e:
            print(f"✗ Error saving for {value['filepath']}: {e}")
            error_count += 1
            
            # Save empty CSV with the same filename and header
            try:
                # Use the same filename construction as above
                filename = os.path.splitext(os.path.basename(value['filepath']))[0]
                ocr_name = value["ocr"]
                config_no = value['config']
                pathfile = f"{filename}_config{config_no}_{ocr_name}_{model_display_name}"
                file_name_csv = f"./results/csv/extracted/{pathfile}.csv"

                with open(file_name_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(
                        f,
                        delimiter=';',
                        quotechar='"',
                        quoting=csv.QUOTE_ALL
                    )
                    header = ["headline", "subheadline", "author", "content", "newspaper_name", "publication_date", "page_number"]
                    writer.writerow(header)
                print(f"✓ Empty CSV saved due to error: {file_name_csv}")
            except Exception as inner_e:
                print(f"✗ Failed to save empty CSV on error: {inner_e}")
            continue

    return saved_count, error_count
        
def save_results_to_json(results_dict: dict, model_display_name: str) -> T.Tuple[int, int]:
    """Saves structured OCR results (JSON) into JSON files.

    Args:
        results_dict (dict): Dictionary of processed results.
        model_display_name (str): Short model name to include in filenames.

    Returns:
        Tuple[int, int]: (successful saves, errors encountered)
    """
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

def process_model_multithreaded(model_name: str, model_display_name: str, ocr_results: dict, max_workers: int = 3, log_file_path: str = "None") -> dict:
    """Processes all OCR results using a specific LLM with multithreading.

    Args:
        model_name (str): Internal LLM model identifier for Ollama.
        model_display_name (str): Friendly name for filenames/logs.
        ocr_results (dict): Parsed OCR results.
        max_workers (int, optional): Thread count.

    Returns:
        dict: Result dictionary of processed entries.
    """
    
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
                key, value, model_name, model_display_name, progress_queue, log_file_path
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

def print_help() -> None:
    """Display usage instructions for the LLM postprocessing script."""
    print("""
    LLM Postprocessing Script (Newspaper OCR Pipeline)
    --------------------------------------------------

    This script uses Ollama to postprocess raw OCR-extracted text into structured data (CSV or JSON).
    It extracts article headlines, subheadlines, authors, and content from `.txt` logs.

    Usage:
        python postprocess.py                   Run full postprocessing for all models
        python postprocess.py --test            Run a unit test with a mock LLM output
        python postprocess.py --help | -h       Show this help message

    Input:
        ./results/txt/extracted/ocr_results_log.txt   ← from the OCR step

    Output:
        ./results/csv/extracted/                    ← structured CSVs
        ./results/txt/extracted/                    ← structured JSONs (optional)

    Dependencies:
        - ollama (local models)
        - LLMs must be available locally via Ollama
    """)

class TestPostprocessing(unittest.TestCase):
    def test_csv_mock(self):
        """Test code block extraction from mock CSV response."""
        mock_response = '''```csv
"headline";"subheadline";"author";"content"
"Test Title";"Test Subtitle";"John Doe";"This is a test article."
```'''
        csv_block = extract_code_block(mock_response, language_hint="csv")
        reader = csv.reader(StringIO(csv_block), delimiter=';', quotechar='"')
        rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], "Test Title")
        self.assertEqual(rows[1][2], "John Doe")

    def test_single_ollama_model(self):
        """Run a real or mock LLM call using one model and minimal input."""
        from postprocess import ollama

        sample_text = (
            "1. Headline: 'Breaking News'\n"
            "   Subheadline: 'This is a subhead'\n"
            "   Author: John Doe\n"
            "   Content: Today something very important happened in the city."
        )

        try:
            model = ollama(model_name="phi4:latest")  # Replace with any model available locally
            response = model.extract_test_results_as_csv(sample_text)
        except Exception as e:
            self.fail(f"Ollama test failed: {e}")


def main() -> None:
    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return

    if '--test' in sys.argv:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        return
    
    max_threads = mp.cpu_count()
    os.makedirs("./results/csv/extracted/", exist_ok=True)

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "postprocess.txt")

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    models = {
        "phi4:latest": "phi4",
        # "llama4:latest": "llama4",
        # "gemma3:27b": "gemma3",
        # # "qwen3:32b": "qwen3",
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
                max_workers=max_threads,
                log_file_path=log_file_path  # Adjust based on your system capabilities
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
