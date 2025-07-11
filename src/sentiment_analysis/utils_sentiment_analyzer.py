import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

def process_row(row):
    output_parts = []
    for col in ["headline", "subheadline", "content"]:
        val = row.get(col)
        if pd.notnull(val) and val != 'NA':
            output_parts.append(str(val))
    output = ". ".join(output_parts).strip()
    return output if output else None

def process_all_rows(csv_files, max_workers=64):
    print(f"🧵 Using ProcessPoolExecutor with {max_workers} workers")
    all_documents = []
    row_mappings = []
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for stem, csv_data in csv_files.items():
            for idx, row in enumerate(csv_data):
                futures.append(executor.submit(process_row, row))

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                all_documents.append(result)
                row_mappings.append(result)

            if i % 1000 == 0:
                print(f"✅ Processed {i} rows...")

    return all_documents, row_mappings


def aggregate_from_model_results(all_model_results, columns=["headline", "subheadline", "content"]):
    num_rows = len(next(iter(all_model_results.values()))[columns[0]]["label"])
    
    aggregated_results = {i: {} for i in range(num_rows)}

    for col in columns:
        for i in range(num_rows):
            label_votes = []
            score_map = defaultdict(list)

            for model_name, model_data in all_model_results.items():
                label = model_data[col]["label"][i]
                score = model_data[col]["score"][i]

                if label != "NA" and score != "NA":
                    label_votes.append(label)
                    score_map[label].append(float(score))

            if label_votes:
                majority_label = Counter(label_votes).most_common(1)[0][0]
                mean_score = sum(score_map[majority_label]) / len(score_map[majority_label])
            else:
                majority_label = "NA"
                mean_score = "NA"

            aggregated_results[i][col] = {
                "label": majority_label,
                "score": mean_score
            }

    return aggregated_results

def aggregate_single_row(all_model_results, i, columns=["headline", "subheadline", "content"]):
    row_result = {}
    for col in columns:
        label_votes = []
        score_map = defaultdict(list)

        for model_name, model_data in all_model_results.items():
            label = model_data[col]["label"][i]
            score = model_data[col]["score"][i]

            if label != "NA" and score != "NA":
                label_votes.append(label)
                score_map[label].append(float(score))

        if label_votes:
            majority_label = Counter(label_votes).most_common(1)[0][0]
            mean_score = sum(score_map[majority_label]) / len(score_map[majority_label])
        else:
            majority_label = "NA"
            mean_score = "NA"

        row_result[col] = {
            "label": majority_label,
            "score": mean_score
        }
    return row_result

def aggregate_row(i):
    return aggregate_single_row(all_model_results, i)

def read_csv_file(file):
    return pd.read_csv(file, sep=";", na_values="NA", quotechar='"')