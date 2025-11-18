from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dataclasses import field

@dataclass
class TrainingConfig:
    # Model Configuration
    model_name: str = 'mrm8488/longformer-base-4096-spanish'
    num_tasks: int = 5
    num_classes: int = 3
    random_seed: int = 42
    label_columns: List[str] = field(default_factory=list)

    # Tokenizer Configuration 
    eos_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None

    # Loss Function Configuration
    focal_loss_config : Dict[str, Any] = field(default_factory=dict)
    class_weights_per_task: Dict[str, List[float]] = field(default_factory=dict) 

    # Training Configuration
    learning_rate: float = 2e-5
    head_learning_rate: float = 0.0002149719644275139 # TODO: Implement
    batch_size: int = 16 
    num_epochs: int = 8 
    warmup_ratio: float = 0.05249928641779438
    weight_decay: float = 0.2808836125696343 
    max_grad_norm: float = 2.5348065843037997
    dropout_rate: float = 0.280737187667872

    # Augmentation Configuration
    swap_ratio: float = 0.25
    deletion_prob: float = 0.25
    
    # Cross-Validation
    cross_validate: bool = False
    n_folds: int = 5
    
    # Hyperparameter Tuning
    hyperparameter_tuning: bool = False
    num_trials: int = 10
    
    # Paths
    results_dir: str = "./results/csv/multi_label"
    model_path: str = "/scratch/nicolasal97/gec_extractor/results/models/multi_label"
    input_file: str = ""
    output_dir: str = "/scratch/nicolasal97/multi_label/checkpoints"
    
    # Evaluation
    eval_metric: str = ""
    type_augmentation: str = ""
    number_samples: int = 0
    target_samples_per_class: int = 50
    max_augmentation_per_sample: int = 10
    num_beams: int = 7 
    temperature: float = 1.2

    # Advanced Settings
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1 # TODO: Add to Hyperparameter Tuning
    early_stopping_patience: int = 3 
    save_model: bool = True
    automatic_augmentation: bool = False
