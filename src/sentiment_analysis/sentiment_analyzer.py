from transformers import pipeline
import multiprocessing as mp
from utils_sentiment_analyzer import *
import os
import pandas as pd
import csv
from glob import glob
from tqdm import tqdm

def main():
    input_folder = "./results/csv/extracted/"
    results_dir = "./results/csv/"
    output_csv = os.path.join(results_dir, "results_sentiment.csv")

    # Read and append CSV files
    all_files = glob(os.path.join(input_folder, "*.csv"))
    df_list = []
    for file in tqdm(all_files, desc="Reading CSV files"):
        df_list.append(pd.read_csv(file, sep=";", na_values="NA", quotechar='"'))
    df = pd.concat(df_list, ignore_index=True)
    
    # Apply process_row with progress bar
    df["combined_text"] = tqdm(df.apply(process_row, axis=1), total=len(df), desc="Processing rows")
    df = df[df["combined_text"].notnull()].reset_index(drop=True)
    print(f"📄 Total valid rows: {len(df)}")

    # Define models to loop over
    models = {
        # "sabert": "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis",
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

        # Prepare containers for each column
        results = {
            "headline": {"label": [], "score": []},
            "subheadline": {"label": [], "score": []},
            "content": {"label": [], "score": []},
        }

        print(f"Running sentiment analysis with {model_name}...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Sentiment analysis ({model_name})"):
            for col in ["headline", "subheadline", "content"]:
                text = row.get(col)
                if pd.notnull(text) and text != 'NA':
                    result = classifier(str(text), truncation=True)[0]
                    normalized_label = label_maps.get(model_name, {}).get(result["label"], result["label"])
                    results[col]["label"].append(normalized_label)
                    results[col]["score"].append(result["score"])
                else:
                    results[col]["label"].append("NA")
                    results[col]["score"].append("NA")

        all_model_results[model_name] = results
        del classifier

    # Now, add those results into df as columns
    for model_name, res in all_model_results.items():
        for col in ["headline", "subheadline", "content"]:
            df[f"{model_name}_{col}_label"] = res[col]["label"]
            df[f"{model_name}_{col}_score"] = res[col]["score"]

    # Aggregates final label and score using majority vote and mean score
    aggregated_results = aggregate_from_model_results(all_model_results)
    
    for col in ["headline", "subheadline", "content"]:
        df[f"agreed_{col}_label"] = [aggregated_results[i][col]["label"] for i in range(len(df))]
        df[f"agreed_{col}_score"] = [aggregated_results[i][col]["score"] for i in range(len(df))]

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