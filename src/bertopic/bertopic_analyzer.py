from bertopic import BERTopic
from utils_bertopic import *
import os
import plotly.io as fig
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import multiprocessing as mp
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
import gc
import torch
import time
import argparse
import csv
import re
import numpy as np
import pickle
from pathlib import Path


def run_single_model(documents, embedding_model_name, chunk_size=1000):    
    # Create fresh embedding model and BERTopic instance
    embedding_model = SentenceTransformer(embedding_model_name)

    spanish_stopwords = [
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por',
    'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero',
    'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando',
    'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien',
    'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra',
    'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos',
    'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos',
    'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas',
    'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus',
    'ellas', 'nosotras', 'vosostros', 'vosostras', 'os', 'mío', 'mía', 'míos',
    'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas',
    'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros',
    'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis',
    'están', 'esté', 'estés', 'estemos', 'estéis', 'estén']

    # Lightweight vectorizer with basic bigrams
    vectorizer_model = CountVectorizer(
        stop_words=spanish_stopwords,    # Swap for "spanish" or None if not English-based
        ngram_range=(1, 2),
        max_features=5000        # Limit vocab size to speed up
    )

    # Speed-tuned UMAP
    umap_model = UMAP(
        n_neighbors=10,          # Fewer neighbors = faster + tighter clusters
        n_components=5,          # Reduce dimensions more aggressively
        min_dist=0.25,           # Looser clustering
        metric='cosine',
        random_state=42
    )


    # # Define topics and their seed words
    # seeded_topics = {
    #     "sports": ["fútbol", "gol", "liga", "deporte"],
    #     "politics": ["elecciones", "gobierno", "presidente"],
    #     "health": ["virus", "salud", "hospital"],
    # }

    # # Faster HDBSCAN with slightly larger clusters
    # hdbscan_model = HDBSCAN(
    #     min_cluster_size=20,     # Controls topic granularity (20 = decent detail)
    #     metric='euclidean',
    #     min_samples=5,
    #     prediction_data=True
    # )

    model = BERTopic(
        embedding_model=embedding_model, 
        language="multilingual", 
        min_topic_size=10,                  # Filters out tiny clusters
        top_n_words=10,                     # Words per topic
        # seed_topic_list=list(seeded_topics.values()),
        # representation_model=KeyBERTInspired(),
        calculate_probabilities=True,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        # hdbscan_model=hdbscan_model,
        verbose=True,
        low_memory=True  
        )

    model_suffix = re.sub(r'\W+', '_', embedding_model_name.split('/')[-1])
    model_path = f"./results/models/bertopic_model_{model_suffix}"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        pickle_file = model_path + ".pkl"

        if os.path.exists(model_path) and os.path.exists(pickle_file):
            print(f"📦 Loading existing BERTopic model from {model_path}")
            # model = BERTopic.load(model_path)

            with open(pickle_file, "rb") as f:
                results = pickle.load(f)
        else:
            print(f"🧠 Training BERTopic model with {embedding_model_name}")
            model.verbose = True
            topics, probs = model.fit_transform(documents)
            topic_keywords = get_topics_keywords(model)

            model.save(model_path, save_embedding_model=True)

            topic_info = model.get_topic_info().to_dict('records') if hasattr(model, 'get_topic_info') else None
            results = {
                'topics': topics,
                'probs': probs,
                'topic_info': topic_info,
                'topic_keywords': topic_keywords
            }

            topic_info = model.get_topic_info()
            print(topic_info.head())

            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
    finally:
        del model
        del embedding_model
        clear_gpu_memory()
        time.sleep(2)  # Brief pause for cleanup
    
    return results


def main():
    input_folder = "./results/csv/test/"
    results_dir = "./results/csv/"
    output_csv = os.path.join(results_dir, "results_topics.csv")

    csv_files = process_csvs(input_folder)

    all_documents = []
    row_mappings = []

    print(f"Processing {len(csv_files)} CSV files from {input_folder}")

    max_threads = mp.cpu_count()
    all_documents, row_mappings = process_all_rows(csv_files, max_workers=max_threads)

    print(f"📄 Total valid documents: {len(all_documents)}")

    if not all_documents:
        print("No valid documents found to process.")
        return None

    embedding_model_names = [
        "hiiamsid/sentence_similarity_spanish_es",
        "Qwen/Qwen3-Embedding-8B", 
        "Linq-AI-Research/Linq-Embed-Mistral"
    ]
    
    all_model_results = []
    
    for model_name in embedding_model_names:
        print(f"\n{'='*50}")
        print(f"Processing with model: {model_name}")
        print(f"{'='*50}")
        
        try:
            results = run_single_model(all_documents, model_name, chunk_size=1000)
            if results is None:
                print(f"⚠️ No results from model {model_name}, skipping...")
                continue
            
            all_model_results.append(results)
            print(f"✅ Completed {model_name}")
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
            clear_gpu_memory()
            time.sleep(5)
            continue
        
        clear_gpu_memory()
        time.sleep(5)  # Longer pause between models

    valid_results = [r for r in all_model_results if r is not None]
    
    if len(valid_results) < 2:
        print("❌ Not enough valid model results for majority voting")
        return None
    
    print(f"\n📊 Processing majority vote from {len(valid_results)} models")
    
    topics_by_model = [result['topics'] for result in valid_results]
    probs_by_model = [result['probs'] for result in valid_results]
    
    reference_topics = valid_results[0]['topics']
    reference_probs = valid_results[0]['probs']
    
    majority_agreed_topics = []
    for i in range(len(all_documents)):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing agreement: {i+1}/{len(all_documents)}")
        
        votes = [topics[i] for topics in topics_by_model if i < len(topics)]
        
        if len(votes) >= 2 and len(set(votes)) < len(votes):  # At least 2 models agree
            agreed_topic = majority_vote(votes)
            
            # Calculate mean probability of agreeing models
            agreeing_probs = []
            for model_idx, vote in enumerate(votes):
                if vote == agreed_topic and model_idx < len(probs_by_model):
                    if i < len(probs_by_model[model_idx]):
                        prob_vector = probs_by_model[model_idx][i]
                        if isinstance(prob_vector, (list, tuple, np.ndarray)) and agreed_topic < len(prob_vector):
                            topic_prob = prob_vector[agreed_topic]
                            if topic_prob is not None:
                                agreeing_probs.append(topic_prob)
                        elif isinstance(prob_vector, float):
                            agreeing_probs.append(prob_vector)
                    else:
                        agreeing_probs.append(0.5)  # fallback if index out of range
            
            mean_agreed_prob = sum(agreeing_probs) / len(agreeing_probs) if agreeing_probs else 0.5
            majority_agreed_topics.append((i, agreed_topic, mean_agreed_prob))

    topic_labels = valid_results[0]['topic_keywords']
    agreed_rows = []
    for i, agreed_topic, mean_agreed_prob in majority_agreed_topics:
        row = row_mappings[i].copy()
        row['combined_text'] = all_documents[i]
        # row['topic'] = reference_topics[i] if i < len(reference_topics) else -1
        # row['topic_prob'] = reference_probs[i] if i < len(reference_probs) and reference_probs[i] is not None else ""
        row['agreed_topic'] = agreed_topic
        row['agreed_topic_prob'] = round(mean_agreed_prob, 4)  
        row['agreed_topic_label'] = topic_labels.get(agreed_topic, "Unknown")
        agreed_rows.append(row)

    # Write results to CSV
    if agreed_rows:
        print(f"\n💾 Writing {len(agreed_rows)} agreed topics to CSV...")
        fieldnames = list(agreed_rows[0].keys())
        
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
        
        print(f"✅ Results written to: {output_csv}")
    else:
        print("❌ No agreed topics found to write")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERTopic Voting & Model Saving")
    parser.add_argument("--visualize-model", type=str, default=None,
                        help="Path to a saved BERTopic model to load instead of training")
    
    args = parser.parse_args()
    if args.visualize_model:
        print(f"📦 Loading BERTopic model from {args.visualize_model}")
        model = BERTopic.load(args.visualize_model)

        # Clean model name for filenames (e.g., strip paths, spaces, slashes)
        model_id = Path(args.visualize_model).stem
        model_id = re.sub(r'\W+', '_', model_id)  # Replace non-alphanumeric with underscores
            
        topic_info = model.get_topic_info()
        print(topic_info.head())

        # Define output directory for visualizations
        viz_output_dir = "./results/visualizations/"
        os.makedirs(viz_output_dir, exist_ok=True)

        # Save visualizations as interactive HTML
        print("📊 Saving visualizations...")

        #TODO: Fix visualizations
        # model.visualize_documents().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))
        # model.visualize_hierarchy().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))            
        # model.visualize_topics_per_class().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_barchart.html"))
        model.visualize_topics().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))
        model.visualize_barchart(top_n_topics=20).write_html(os.path.join(viz_output_dir, f"{model_id}_topics_barchart.html"))
        model.visualize_heatmap().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_heatmap.html"))
        model.visualize_term_rank().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_termrank.html"))
        
        print(f"✅ Visualizations saved to {viz_output_dir}")
    else:
        main()