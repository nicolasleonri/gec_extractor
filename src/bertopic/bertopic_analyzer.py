from bertopic import BERTopic
from utils_bertopic import *
import os
import plotly.io as fig
from sentence_transformers import SentenceTransformer
from collections import Counter
import gc
import torch
import time

def print_gpu_usage(note=""):
    print(f"\n🔍 GPU Memory Usage {note}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    try:
        output = subprocess.check_output(["nvidia-smi"], encoding='utf-8')
        print(output)
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")

def majority_vote(votes):
    most_common = Counter(votes).most_common(1)
    return most_common[0][0] if most_common else None

def main():
    input_folder = "./results/csv/extracted/"
    results_dir = "./results/csv/"
    output_csv = os.path.join(results_dir, "results_topics.csv")

    csv_files = process_csvs(input_folder)

    all_documents = []
    row_mappings = []  # Save tuples of (row, combined_text)

    for stem, csv_data in csv_files.items():
        for idx, row in enumerate(csv_data):
            output_parts = []

            if row['headline'] != 'NA':
                output_parts.append(row['headline'])

            if row['subheadline'] != 'NA':
                output_parts.append(row['subheadline'])

            if row['content'] != 'NA':
                output_parts.append(row['content'])

            output = ". ".join(output_parts)

            if output.strip() != "":
                row['combined_text'] = output
                all_documents.append(output)
                row_mappings.append(row)
    
    print(f"Total valid documents: {len(all_documents)}")

    embedding_models = [
    SentenceTransformer("hiiamsid/sentence_similarity_spanish_es"),
    SentenceTransformer("Qwen/Qwen3-Embedding-8B"),
    SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")
    ]

    bertopic_models = [
        BERTopic(embedding_model=emb, language="multilingual", calculate_probabilities=True)
        for emb in embedding_models
    ]
    
    if all_documents:
        topics_by_model = []
        for model in bertopic_models:
            print_gpu_usage("BEFORE model fit")

            print(f"Fitting model with embedding: {model.embedding_model}")
            topics, _ = model.fit_transform(all_documents)
            topics_by_model.append(topics)

            # Clear model and free GPU memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

            time.sleep(10)  # Give some time for GPU memory to clear
            print_gpu_usage("AFTER model fit")

        reference_model = bertopic_models[0]
        topics, probs = reference_model.fit_transform(all_documents)

        # Clean up
        del reference_model
        gc.collect()
        torch.cuda.empty_cache()
        
        majority_agreed_topics = []
        for i in range(len(all_documents)):
            print(f"Processing agreement of document {i+1}/{len(all_documents)}")
            votes = [topics_by_model[m][i] for m in range(3)]
            if len(set(votes)) < 3:  # At least 2 models agree
                agreed_topic = majority_vote(votes)
                majority_agreed_topics.append((i, agreed_topic))

        for i, row in enumerate(row_mappings):
            row['combined_text'] = all_documents[i]  # Add combined text for context
            row['topic'] = topics[i]
            row['topic_prob'] = probs[i] if probs[i] is not None else ""

        agreed_rows = []
        for i, agreed_topic in majority_agreed_topics:
            print(f"Document {i+1} agreed topic: {agreed_topic}")
            row = row_mappings[i]
            row['agreed_topic'] = agreed_topic
            agreed_rows.append(row)

        # TODO: Before writing to CSV, clean everything not needed (dictionaries)

        # Save all enriched rows to one output CSV
        print(f"Writing {len(agreed_rows)} agreed topics to CSV...")
        fieldnames = list(row_mappings[0].keys())
        with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(
            f, 
            fieldnames=fieldnames,
            delimiter=';',
            quotechar='"',
            quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(agreed_rows)

        print(f"✅ Topics written to: {output_csv}")
        print(f"📊 Top 10 topics:\n{topic_model.get_topic_info().head(10)}")

        # output_vis = os.path.join(results_dir, "topic_visualization.html")
        # fig.write_html(topic_model.visualize_topics(), output_vis)
        # print(f"✅ Topic visualization saved to: {output_vis}")
    else:
        print("⚠️ No valid documents found. Exiting.")

    return None

if __name__ == "__main__":
    main()