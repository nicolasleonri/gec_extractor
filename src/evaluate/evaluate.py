from utils_evaluate import *
# import numpy as np
from pathlib import Path
import csv

def main():
    final_results = []

    gold_standard_dir = "./data/csv/"
    results_dir = "./results/csv/extracted/"

    gold_standards = process_gold_standards(gold_standard_dir)

    results_path = Path(results_dir)
    eval_files = list(results_path.glob('*.csv'))

    best_f1 = 0.0
    best_model_info = ""

    # for eval_file in results_path.glob('*.json'):
    for idx, eval_file in enumerate(eval_files, 1):
        total = len(eval_files)
        print(f"[{idx}/{total}] Processing: {eval_file.name}")

        stem = eval_file.stem.split('_')[0]
        config = eval_file.stem.split('_')[1]
        ocr_module = eval_file.stem.split('_')[2]
        llm_model = eval_file.stem.split('_')[3]

        if stem in gold_standards:
            # print(f"Processing: {eval_file.name}")

            gold_data = gold_standards[stem]
            try:
                eval_data = load_csv_file(str(eval_file))
                # eval_data = load_json_file(str(eval_file))
            except Exception as e:
                print(f"Error loading eval file {eval_file.name}: {e}")
                continue

            comparison_results = preprocess_data(gold_data, eval_data)
            
            if comparison_results == None:
                y_true, y_pred = [1, 1, 1], [0, 0, 0]
            else:
                y_true, y_pred = prepare_for_sklearn_metrics(comparison_results)

            (accuracy, precision, recall, f1) = calculate_metrics(y_true, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                # best_model_info = f"{stem}_{config}_{ocr_module}_{llm_model}"
                print(f"Best F1 so far: {best_f1:.5f}")
            
            # Calculate metrics            
            # print("Metrics:")
            # print(f"Accuracy: {accuracy}")
            # print(f"Precision: {precision}")
            # print(f"Recall: {recall}")
            # print(f"F1-Score: {f1}")

            if np.isnan(accuracy):
                accuracy = 0.0

            list_to_add = [stem, config[6:], ocr_module, llm_model, float(accuracy), float(precision), float(recall), float(f1)]
            final_results.append(list_to_add)

        else:
            print(f"Warning: No matching gold standard found for {eval_file.name}")

    # print(final_results)

    with open('./results/csv/evaluation.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        head = ['Filename', 'Config', 'OCR_module', 'LLM_model', "Accuracy", "Precision", "Recall", "F1_Score"]
        csvwriter.writerow(head)

        # Write each row from the list
        for row in final_results:
            csvwriter.writerow(row)
    


if __name__ == "__main__":
    main()