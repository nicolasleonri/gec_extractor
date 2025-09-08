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
from pathlib import Path
import gc
import torch
import time
import argparse
import csv
import re
import numpy as np
import pickle
from pathlib import Path

def get_csv_files(directory):
    SUPPORTED_FORMATS = ['.csv']

    logs_files = []
    
    for file in Path(directory).rglob('*'):
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            logs_files.append(file)
            
    output = sorted(logs_files)

    return output

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE, text=True
    )
    print("GPU Memory:", result.stdout.strip())

def run_single_model(documents, embedding_model_name, newspaper, chunk_size=1000):
    check_gpu()

    start_time = time.time()
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

    vectorizer_model = CountVectorizer(
        stop_words=spanish_stopwords,
        decode_error="replace",
        ngram_range=(1, 5),
        max_features=None,
        strip_accents="unicode",
        min_df=2,  # Remove very rare terms
        max_df=0.95,  # Remove very common terms
        token_pattern=r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{2,}\b'  # Spanish-aware tokenization
    )

    # Speed-tuned UMAP
    umap_model = UMAP(
        n_neighbors=200,
        n_components=100, 
        min_dist=0.01,
        metric='cosine',
        random_state=42,
        low_memory=False,
        n_jobs=-1,
        verbose=True
    )

    # # Define topics and their seed words
    # seeded_topics = {
    #     "sports": ["fútbol", "gol", "liga", "deporte"],
    #     "politics": ["elecciones", "gobierno", "presidente"],
    #     "health": ["virus", "salud", "hospital"],
    # }

    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=-1,  # Use all CPU cores
        algorithm='boruvka_kdtree'
    )

    model = BERTopic(
        embedding_model=embedding_model, 
        language="multilingual", 
        min_topic_size=3,
        top_n_words=50, # Words per topic
        # seed_topic_list=list(seeded_topics.values()),
        # representation_model=KeyBERTInspired(),
        calculate_probabilities=True,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        low_memory=True,
        nr_topics=None,
        )

    model_suffix = re.sub(r'\W+', '_', embedding_model_name.split('/')[-1])
    model_path = f"./results/models/bertopic/bertopic_model_{model_suffix}_{newspaper}"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"⏱️ Initializing time: {time.time() - start_time:.2f} seconds")
    check_gpu()

    try:
        pickle_file = model_path + ".pkl"

        if os.path.exists(model_path) and os.path.exists(pickle_file):
            print(f"📦 Loading existing BERTopic model from {model_path}")
            model = BERTopic.load(model_path)

            with open(pickle_file, "rb") as f:
                results = pickle.load(f)
        else:
            print(f"🧠 Training BERTopic model with {embedding_model_name}")
            start_time = time.time()

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

            print("📊 Topic info (head):")
            topic_info = model.get_topic_info()
            print(topic_info.head())

            print(f"⏱️ Training time: {time.time() - start_time:.2f} seconds")
            check_gpu()

            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
    finally:
        del model
        del embedding_model
        clear_gpu_memory()
        time.sleep(2)  # Brief pause for cleanup
    
    return results


def bertopic(input_files, newspaper, input_folder):
    results_dir = "./results/csv/bertopic/"
    csv_filename = f"results_topics_{newspaper}.csv"
    output_csv = os.path.join(results_dir, csv_filename)
    os.makedirs(os.path.dirname(results_dir), exist_ok=True)

    all_documents = []
    row_mappings = []

    print(f"Processing {len(input_files)} CSV files from {str(input_folder)}")

    max_threads = mp.cpu_count()
    all_documents, row_mappings = process_all_rows(input_files, max_workers=max_threads)

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
            results = run_single_model(all_documents, model_name, newspaper, chunk_size=1000)
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

    topic_labels = valid_results[1]['topic_keywords']
    print(topic_labels)

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
    parser.add_argument('-f', '--input_folder', required=True, help='Folder with OCR results')
    parser.add_argument('-n', '--newspaper', required=True, help='Newspaper name (required)')

    args = parser.parse_args()

    csv_files = get_csv_files(str(args.input_folder))

    start_time = time.time()

    bertopic(csv_files, args.newspaper, args.input_folder)

    print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")

    # if args.visualize_model:
    #     print(f"📦 Loading BERTopic model from {args.visualize_model}")
    #     model = BERTopic.load(args.visualize_model)

    #     # Clean model name for filenames (e.g., strip paths, spaces, slashes)
    #     model_id = Path(args.visualize_model).stem
    #     model_id = re.sub(r'\W+', '_', model_id)  # Replace non-alphanumeric with underscores
            
    #     topic_info = model.get_topic_info()
    #     print(topic_info.head())

    #     # Define output directory for visualizations
    #     viz_output_dir = "./results/visualizations/"
    #     os.makedirs(viz_output_dir, exist_ok=True)

    #     # Save visualizations as interactive HTML
    #     print("📊 Saving visualizations...")

    #     #TODO: Fix visualizations
    #     # model.visualize_documents().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))
    #     # model.visualize_hierarchy().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))            
    #     # model.visualize_topics_per_class().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_barchart.html"))
    #     model.visualize_topics().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_overview.html"))
    #     model.visualize_barchart(top_n_topics=20).write_html(os.path.join(viz_output_dir, f"{model_id}_topics_barchart.html"))
    #     model.visualize_heatmap().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_heatmap.html"))
    #     model.visualize_term_rank().write_html(os.path.join(viz_output_dir, f"{model_id}_topics_termrank.html"))
        
    #     print(f"✅ Visualizations saved to {viz_output_dir}")
    # else:
    #     return None