from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, FillMaskPipeline
from transformers import DataCollatorWithPadding
from transformers import AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from typing import Union, List, Dict, Any, Optional, Tuple
from transformers import EarlyStoppingCallback
from transformers import LongformerModel
from pprint import pprint
from sklearn.metrics import f1_score
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import evaluate
import json
import sys
import os
import torch

class TextClassifierDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)  # Convert to float
        return item

class MultiTaskLongformer(nn.Module):
    def __init__(self, model_name, num_tasks=5, num_classes=3):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        # Create 5 separate classification heads
        self.classifiers = nn.ModuleList([
            nn.Linear(self.longformer.config.hidden_size, num_classes)
            for _ in range(num_tasks)
        ])
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Get logits from each classifier
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        logits = torch.stack(logits, dim=1)  # Shape: (batch, 5, 3)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Calculate loss for each task
            losses = []
            for i in range(len(self.classifiers)):
                # Map -1/0/1 to 0/1/2 for CrossEntropyLoss
                task_labels = labels[:, i] + 1  # -1→0, 0→1, 1→2
                task_logits = logits[:, i, :]
                losses.append(loss_fct(task_logits, task_labels.long()))
            loss = torch.stack(losses).mean()
        
        return {"loss": loss, "logits": logits}

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Do text classification in a BERT-like model for long texts.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--input_file", type=str, required=True, help="Path to csv file.")
    parser.add_argument('-sm', '--save_model', type=int, required=False, default=True, help='Saves a pre-trained model.')
    parser.add_argument('-lm', '--load_model', type=bool, required=False, default=False, help='Loads a pre-trained model.')
    parser.add_argument('-mp', '--model_path', type=str, required=True,  help='Path to save/load model.')
    parser.add_argument('-ns', '--number_samples', type=int, required=False, help='Number of samples to use.')
    parser.add_argument('-tm', '--train_model', type=bool, required=False, default=False, help='Flags if model should be trained.')

    return parser.parse_args()

def validate_arguments(args) -> Dict[str, Any]:
    config = {}
    errors = []
    
    if not os.path.isfile(args.input_file):
        errors.append(f"Input file does not exist: {args.input_file}")
    else:
        config["input_file"] = args.input_file

    config["save_model"] = args.save_model
    config["load_model"] = args.load_model
    config["model_path"] = args.model_path
    config["number_samples"] = args.number_samples
    config["train_model"] = args.train_model

    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    return config

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_classes = np.argmax(predictions, axis=2)  # Shape: (batch_size, 5)
    true_classes = (labels + 1).astype(int)  # -1→0, 0→1, 1→2
    accuracy = (predicted_classes == true_classes).mean()
    task_accuracies = []
    task_f1_scores = []
    for i in range(5):
        task_acc = (predicted_classes[:, i] == true_classes[:, i]).mean()
        task_f1 = f1_score(true_classes[:, i], predicted_classes[:, i], average='macro')
        task_accuracies.append(task_acc)
        task_f1_scores.append(task_f1)
    
    overall_f1 = np.mean(task_f1_scores)
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(overall_f1),
        "task_0_acc": float(task_accuracies[0]),
        "task_1_acc": float(task_accuracies[1]),
        "task_2_acc": float(task_accuracies[2]),
        "task_3_acc": float(task_accuracies[3]),
        "task_4_acc": float(task_accuracies[4]),
        "task_0_f1": float(task_f1_scores[0]),
        "task_1_f1": float(task_f1_scores[1]),
        "task_2_f1": float(task_f1_scores[2]),
        "task_3_f1": float(task_f1_scores[3]),
        "task_4_f1": float(task_f1_scores[4]),
        "mean_task_accuracy": float(np.mean(task_accuracies)),
    }

def train(config) -> None:
    df = pd.read_csv(config["input_file"], sep=",", doublequote=True, encoding="utf-8", encoding_errors="strict", header=0, date_format='%Y-%M-%d')
    
    if config['number_samples']:
        df = df.head(config['number_samples'])
        print(f"Using first {config['number_samples']} samples")

    df['date'] = pd.to_datetime(df['date'], format='%Y-%M-%d')
    df['newspaper'] = df['newspaper'].astype('category')
    df['combined'] = df.apply(lambda row: '. '.join(row[['headline', 'content']].dropna().astype(str)), axis=1)

    # train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['newspaper'])
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")

    not_chosen_columns = ['headline', 'content', 'newspaper', 'date', 'combined', 'comentario']
    label_columns = [col for col in df.columns if col not in not_chosen_columns]

    df_labels_train = train_df[label_columns]
    df_labels_test = test_df[label_columns]

    labels_list_train = df_labels_train.values.tolist()
    labels_list_test = df_labels_test.values.tolist()

    train_texts = train_df['combined'].tolist()
    train_labels = labels_list_train

    eval_texts = test_df['combined'].tolist()
    eval_labels = labels_list_test

    val_texts = val_df['combined'].tolist()
    val_labels = val_df[label_columns].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained('mrm8488/longformer-base-4096-spanish', do_lower_case=True)
    
    train_encodings = tokenizer(train_texts, truncation=True, max_length=4096)
    eval_encodings = tokenizer(eval_texts, truncation=True, max_length=4096)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=4096)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = TextClassifierDataset(train_encodings, train_labels)
    eval_dataset = TextClassifierDataset(eval_encodings, eval_labels)
    val_dataset = TextClassifierDataset(val_encodings, val_labels)

    model = MultiTaskLongformer("mrm8488/longformer-base-4096-spanish")

    training_arguments = TrainingArguments(
        output_dir="./results/checkpoints",
        learning_rate=2e-5,
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        metric_for_best_model="macro_f1",  # Changed from eval_loss
        greater_is_better=True, 
        logging_strategy="epoch",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        processing_class=tokenizer,
        data_collator=data_collator,  # Add this!
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    test_results = trainer.evaluate(eval_dataset)

    print("Final test set results:", test_results)

    print(f"💾 Saving model to {config['model_path']}")
    trainer.save_model(config['model_path'])
    tokenizer.save_pretrained(config['model_path'])
    
    results_path = os.path.join(config['model_path'], 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"✅ Model and results saved to {config['model_path']}")

def main() -> None:
    args = parse_arguments()
    config = validate_arguments(args)

    if config['train_model']:
        train(config)
    elif config['load_model']:
        print(f"Loading model from {config['model_path']}")
        model = MultiTaskLongformer.from_pretrained(config['model_path'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    else:
        print("No action specified. Use --train_model or --load_model flags.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    main()