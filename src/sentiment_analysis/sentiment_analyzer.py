from transformers import pipeline
import multiprocessing as mp
from utils_sentiment_analyzer import *
import os
import pandas as pd
import csv
from glob import glob
from tqdm import tqdm
import concurrent.futures
from datasets import Dataset

def main():
    input_folder = "./results/csv/extracted/"
    results_dir = "./results/csv/"
    output_csv = os.path.join(results_dir, "results_sentiment.csv")

    # Read and append CSV files
    all_files = glob(os.path.join(input_folder, "*.csv"))
    df_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map files to read_csv_file with progress bar
        for df_chunk in tqdm(executor.map(read_csv_file, all_files), total=len(all_files), desc="Reading CSV files"):
            df_list.append(df_chunk)

    df = pd.concat(df_list, ignore_index=True)

    # For applying process_row, you can keep tqdm as before
    df["combined_text"] = tqdm(df.apply(process_row, axis=1), total=len(df), desc="Processing rows")
    df = df[df["combined_text"].notnull()].reset_index(drop=True)

    print(f"📄 Total valid rows: {len(df)}")

    # Convert pandas DataFrame to HF Dataset
    dataset = Dataset.from_pandas(df)

    # Define models to loop over
    models = {
        "sabert": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
        "robertuito": "pysentimiento/robertuito-sentiment-analysis",
        # "lxyuan": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        # "edumunozsala": "edumunozsala/roberta_bne_sentiment_analysis_es",
        "UMUTeam": "UMUTeam/roberta-spanish-sentiment-analysis"
    }

    label_maps = {
        "sabert": {
            "Positive": "Positive",
            "Negative": "Negative",
            "Neutral": "Neutral",
        },
        "lxyuan": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        },
        "robertuito": {
            "POS": "Positive",
            "NEG": "Negative",
            "NEU": "Neutral",
        },
        "edumunozsala": {
            "Positivo": "Positive",
            "Negativo": "Negative",
            "Neutral": "Neutral",
        },
        "UMUTeam": {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
        },
    }

    all_model_results = {}

    for model_name, model_path in models.items():
        print(f"Loading model: {model_name}")
        classifier = pipeline("text-classification", model=model_path, batch_size=64)

        def classify_batch(batch):
            results = {}
            for col in ["headline", "subheadline", "content"]:
                texts = batch.get(col, [])
                texts = [text if text is not None and text != "NA" else "" for text in texts]

                preds = classifier(texts, truncation=True)

                labels = [label_maps.get(model_name, {}).get(p["label"], p["label"]) for p in preds]
                scores = [p["score"] for p in preds]

                results[f"{col}_label"] = labels
                results[f"{col}_score"] = scores
            return results
        
        # Run batch inference on dataset
        dataset = dataset.map(classify_batch, batched=True, batch_size=64)

        # Store the model's results in dict to merge later
        all_model_results[model_name] = {
            col: {
                "label": dataset[f"{col}_label"],
                "score": dataset[f"{col}_score"]
            }
            for col in ["headline", "subheadline", "content"]
        }

        del classifier

    # Now, add those results into df as columns
    for model_name, res in all_model_results.items():
        for col in ["headline", "subheadline", "content"]:
            df[f"{model_name}_{col}_label"] = res[col]["label"]
            df[f"{model_name}_{col}_score"] = res[col]["score"]

    print("🗳️ Aggregating majority votes and mean scores across models...")
    num_rows = len(next(iter(all_model_results.values()))["headline"]["label"])

    def aggregate_row(i):
        return aggregate_single_row(all_model_results, i)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(aggregate_row, range(num_rows)), total=num_rows))

    aggregated_results = {i: results[i] for i in range(num_rows)}
    
    for col in ["headline", "subheadline", "content"]:
        df[f"agreed_{col}_label"] = [aggregated_results[i][col]["label"] for i in range(len(df))]
        df[f"agreed_{col}_score"] = [aggregated_results[i][col]["score"] for i in range(len(df))]

    print(f"✅ Results written to: {output_csv}")

    # Save results
    df.to_csv(
        output_csv,
        index=False,
        sep=';',
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        encoding='utf-8'
    )

    print(f"Saved sentiment results to: {output_csv}")

    return None

if __name__ == "__main__":
    main()