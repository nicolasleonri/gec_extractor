import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Literal, Any
from transformers import (
    MarianMTModel, 
    MarianTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline
)
from scipy.stats import chi2_contingency, entropy
from tqdm import tqdm
import random
import re
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings('ignore')

def analyze_class_distribution(
    df: pd.DataFrame,
    label_columns: List[str],
    split_name: str = 'full') -> Dict[str, Any]:
    """
    Analiza la distribución de clases para cada tarea en un dataset multi-task.
    
    Args:
        df: DataFrame con las etiquetas
        label_columns: Lista de nombres de columnas de etiquetas (tareas)
        split_name: Nombre del split ('train', 'val', 'test', 'full')
    
    Returns:
        Dictionary con análisis detallado por tarea:
        {
            'task_name': {
                'class_counts': {-1: int, 0: int, 1: int},
                'class_percentages': {-1: float, 0: float, 1: float},
                'imbalance_ratio': float,
                'entropy': float,
                'minority_percentage': float,
                'effective_samples': float,
                'severity': str
            },
            'summary': {...}
        }
    """
    results = {}
    total_samples = len(df)
    
    for task in label_columns:
        task_labels = df[task].values
        
        unique, counts = np.unique(task_labels, return_counts=True)
        class_counts = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
        
        for cls in [-1, 0, 1]:
            if cls not in class_counts:
                class_counts[cls] = 0
        
        class_percentages = {
            cls: (count / total_samples) * 100 
            for cls, count in class_counts.items()
        }
        
        counts_array = np.array([class_counts[-1], class_counts[0], class_counts[1]])
        
        max_count = counts_array.max()
        min_count = counts_array[counts_array > 0].min() if (counts_array > 0).any() else 1
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        probs = counts_array / counts_array.sum()
        probs = probs[probs > 0]  # Evitar log(0)
        shannon_entropy = entropy(probs, base=2)
        max_entropy = np.log2(3)  # Máximo para 3 clases
        
        minority_percentage = (min_count / total_samples) * 100
        
        beta = 0.9999
        effective_samples = sum(
            [(1 - beta**count) / (1 - beta) for count in counts_array if count > 0]
        )
        
        if imbalance_ratio < 3:
            severity = 'balanced'
        elif imbalance_ratio < 10:
            severity = 'moderate'
        elif imbalance_ratio < 20:
            severity = 'high'
        else:
            severity = 'severe'
        
        results[task] = {
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'imbalance_ratio': float(imbalance_ratio),
            'entropy': float(shannon_entropy),
            'max_entropy': float(max_entropy),
            'entropy_ratio': float(shannon_entropy / max_entropy),
            'minority_percentage': float(minority_percentage),
            'effective_samples': float(effective_samples),
            'severity': severity,
            'total_samples': total_samples
        }

    all_irs = [results[task]['imbalance_ratio'] for task in label_columns]
    all_entropies = [results[task]['entropy'] for task in label_columns]
    all_minorities = [results[task]['minority_percentage'] for task in label_columns]
    
    summary = {
        'split_name': split_name,
        'total_samples': total_samples,
        'num_tasks': len(label_columns),
        'mean_imbalance_ratio': float(np.mean(all_irs)),
        'max_imbalance_ratio': float(np.max(all_irs)),
        'mean_entropy': float(np.mean(all_entropies)),
        'min_entropy': float(np.min(all_entropies)),
        'mean_minority_percentage': float(np.mean(all_minorities)),
        'min_minority_percentage': float(np.min(all_minorities)),
        'most_imbalanced_task': label_columns[np.argmax(all_irs)],
        'least_imbalanced_task': label_columns[np.argmin(all_irs)],
        'overall_severity': 'severe' if np.max(all_irs) > 20 else 
                           'high' if np.max(all_irs) > 10 else
                           'moderate' if np.max(all_irs) > 3 else 'balanced'
    }
    
    results['summary'] = summary
    
    return results

def analyze_label_combinations(
    df: pd.DataFrame,
    label_columns: List[str],
    rare_threshold: int = 5) -> Dict[str, Any]:
    """
    Analiza las combinaciones de etiquetas multi-task para identificar
    patrones comunes y combinaciones raras.
    
    Args:
        df: DataFrame con las etiquetas
        label_columns: Lista de nombres de columnas de etiquetas
        rare_threshold: Umbral para considerar una combinación como "rara"
    
    Returns:
        Dictionary con análisis de combinaciones:
        {
            'total_unique_combinations': int,
            'total_possible_combinations': int,
            'coverage_percentage': float,
            'combinations': List[Dict],
            'rare_combinations': int,
            'most_common': List[Dict],
            'rarest': List[Dict],
            'extreme_cases': Dict
        }
    """
    total_samples = len(df)
    num_tasks = len(label_columns)
    total_possible = 3 ** num_tasks  # 3 clases por tarea
    
    # Extraer combinaciones
    label_matrix = df[label_columns].values
    combinations = [tuple(row) for row in label_matrix]
    combination_counts = Counter(combinations)
    
    # Analizar cada combinación
    combinations_data = []
    for pattern, count in combination_counts.items():
        percentage = (count / total_samples) * 100
        is_rare = count < rare_threshold or percentage < 1.0
        
        combinations_data.append({
            'pattern': pattern,
            'count': count,
            'percentage': percentage,
            'is_rare': is_rare
        })
    
    # Ordenar por frecuencia
    combinations_data.sort(key=lambda x: x['count'], reverse=True)
    
    # Identificar combinaciones raras
    rare_combinations = sum(1 for c in combinations_data if c['is_rare'])
    
    # Top 10 más comunes y raras
    most_common = combinations_data[:10]
    rarest = sorted(combinations_data, key=lambda x: x['count'])[:10]
    
    # Casos extremos
    all_positive = sum(1 for c in combinations if all(x == 1 for x in c))
    all_negative = sum(1 for c in combinations if all(x == -1 for x in c))
    all_neutral = sum(1 for c in combinations if all(x == 0 for x in c))
    no_positive = sum(1 for c in combinations if all(x != 1 for x in c))
    no_negative = sum(1 for c in combinations if all(x != -1 for x in c))
    
    # Análisis de single-label (solo una tarea positiva)
    single_positive_per_task = {}
    for i, task in enumerate(label_columns):
        single_positive = sum(
            1 for c in combinations 
            if c[i] == 1 and sum(1 for x in c if x == 1) == 1
        )
        single_positive_per_task[task] = single_positive
    
    extreme_cases = {
        'all_positive': {
            'count': all_positive,
            'percentage': (all_positive / total_samples) * 100,
            'pattern': tuple([1] * num_tasks)
        },
        'all_negative': {
            'count': all_negative,
            'percentage': (all_negative / total_samples) * 100,
            'pattern': tuple([-1] * num_tasks)
        },
        'all_neutral': {
            'count': all_neutral,
            'percentage': (all_neutral / total_samples) * 100,
            'pattern': tuple([0] * num_tasks)
        },
        'no_positive_labels': {
            'count': no_positive,
            'percentage': (no_positive / total_samples) * 100
        },
        'no_negative_labels': {
            'count': no_negative,
            'percentage': (no_negative / total_samples) * 100
        },
        'single_positive_per_task': single_positive_per_task
    }
    
    # Métricas de diversidad
    coverage_percentage = (len(combination_counts) / total_possible) * 100
    
    return {
        'total_unique_combinations': len(combination_counts),
        'total_possible_combinations': total_possible,
        'coverage_percentage': float(coverage_percentage),
        'combinations': combinations_data,
        'rare_combinations': rare_combinations,
        'rare_percentage': (rare_combinations / len(combination_counts)) * 100,
        'most_common': most_common,
        'rarest': rarest,
        'extreme_cases': extreme_cases,
        'combination_counts_dict': {str(k): v for k, v in combination_counts.items()}
    }

def compare_splits_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_columns: List[str]) -> Dict[str, Any]:
    """
    Compara las distribuciones de clases entre train/val/test splits usando
    chi-square test y otras métricas estadísticas.
    
    Args:
        train_df: DataFrame de entrenamiento
        val_df: DataFrame de validación
        test_df: DataFrame de test
        label_columns: Lista de nombres de columnas de etiquetas
    
    Returns:
        Dictionary con comparaciones por tarea:
        {
            'task_name': {
                'distributions': {...},
                'chi_square_tests': {...},
                'max_deviation': float,
                'is_consistent': bool,
                'warnings': List[str]
            },
            'overall_consistency': {...}
        }
    """
    results = {}
    all_consistent = True
    all_warnings = []
    
    for task in label_columns:
        # Obtener distribuciones
        train_dist = train_df[task].value_counts(normalize=True).sort_index() * 100
        val_dist = val_df[task].value_counts(normalize=True).sort_index() * 100
        test_dist = test_df[task].value_counts(normalize=True).sort_index() * 100
        
        # Asegurar que todas las clases están presentes
        for cls in [-1, 0, 1]:
            if cls not in train_dist:
                train_dist[cls] = 0.0
            if cls not in val_dist:
                val_dist[cls] = 0.0
            if cls not in test_dist:
                test_dist[cls] = 0.0
        
        train_dist = train_dist.sort_index()
        val_dist = val_dist.sort_index()
        test_dist = test_dist.sort_index()
        
        # Conteos absolutos para chi-square
        train_counts = train_df[task].value_counts().sort_index()
        val_counts = val_df[task].value_counts().sort_index()
        test_counts = test_df[task].value_counts().sort_index()
        
        # Asegurar clases presentes en conteos
        for cls in [-1, 0, 1]:
            if cls not in train_counts:
                train_counts[cls] = 0
            if cls not in val_counts:
                val_counts[cls] = 0
            if cls not in test_counts:
                test_counts[cls] = 0
        
        train_counts = train_counts.sort_index()
        val_counts = val_counts.sort_index()
        test_counts = test_counts.sort_index()
        
        # Chi-square tests
        # Train vs Val
        contingency_train_val = np.array([train_counts.values, val_counts.values])
        try:
            chi2_train_val, p_train_val, dof_tv, _ = chi2_contingency(contingency_train_val)
        except ValueError:
            chi2_train_val, p_train_val = None, None
        
        # Train vs Test
        contingency_train_test = np.array([train_counts.values, test_counts.values])
        try:
            chi2_train_test, p_train_test, dof_tt, _ = chi2_contingency(contingency_train_test)
        except ValueError:
            chi2_train_test, p_train_test = None, None
        
        # Val vs Test
        contingency_val_test = np.array([val_counts.values, test_counts.values])
        try:
            chi2_val_test, p_val_test, dof_vt, _ = chi2_contingency(contingency_val_test)
        except ValueError:
            chi2_val_test, p_val_test = None, None
        
        # Calcular desviación máxima
        deviations = []
        for cls in [-1, 0, 1]:
            dev_tv = abs(train_dist[cls] - val_dist[cls])
            dev_tt = abs(train_dist[cls] - test_dist[cls])
            dev_vt = abs(val_dist[cls] - test_dist[cls])
            deviations.extend([dev_tv, dev_tt, dev_vt])
        
        max_deviation = max(deviations)
        
        # Verificar consistencia
        warnings_list = []
        is_consistent = True
        
        # Check chi-square p-values
        if p_train_val is not None and p_train_val < 0.05:
            warnings_list.append(
                f"Train-Val distributions significantly different (p={p_train_val:.4f})"
            )
            is_consistent = False
        
        if p_train_test is not None and p_train_test < 0.05:
            warnings_list.append(
                f"Train-Test distributions significantly different (p={p_train_test:.4f})"
            )
            is_consistent = False
        
        # Check maximum deviation
        if max_deviation > 5.0:
            warnings_list.append(
                f"High distribution deviation detected ({max_deviation:.2f}%)"
            )
            is_consistent = False
        
        # Check for missing classes
        if 0 in train_counts.values:
            warnings_list.append("Some classes missing in train set")
            is_consistent = False
        if 0 in val_counts.values:
            warnings_list.append("Some classes missing in val set")
            is_consistent = False
        if 0 in test_counts.values:
            warnings_list.append("Some classes missing in test set")
            is_consistent = False
        
        results[task] = {
            'distributions': {
                'train': train_dist.to_dict(),
                'val': val_dist.to_dict(),
                'test': test_dist.to_dict()
            },
            'counts': {
                'train': train_counts.to_dict(),
                'val': val_counts.to_dict(),
                'test': test_counts.to_dict()
            },
            'chi_square_tests': {
                'train_vs_val': {
                    'statistic': float(chi2_train_val) if chi2_train_val is not None else None,
                    'p_value': float(p_train_val) if p_train_val is not None else None,
                    'similar': p_train_val > 0.05 if p_train_val is not None else None
                },
                'train_vs_test': {
                    'statistic': float(chi2_train_test) if chi2_train_test is not None else None,
                    'p_value': float(p_train_test) if p_train_test is not None else None,
                    'similar': p_train_test > 0.05 if p_train_test is not None else None
                },
                'val_vs_test': {
                    'statistic': float(chi2_val_test) if chi2_val_test is not None else None,
                    'p_value': float(p_val_test) if p_val_test is not None else None,
                    'similar': p_val_test > 0.05 if p_val_test is not None else None
                }
            },
            'max_deviation': float(max_deviation),
            'is_consistent': is_consistent,
            'warnings': warnings_list
        }
        
        if not is_consistent:
            all_consistent = False
            all_warnings.extend([f"{task}: {w}" for w in warnings_list])
    
    # Resumen general
    overall_consistency = {
        'all_tasks_consistent': all_consistent,
        'num_inconsistent_tasks': sum(1 for task_data in results.values() 
                                       if not task_data['is_consistent']),
        'max_deviation_across_tasks': max(
            results[task]['max_deviation'] for task in label_columns
        ),
        'all_warnings': all_warnings
    }
    
    results['overall_consistency'] = overall_consistency
    
    return results

def analyze_task_correlations(
    df: pd.DataFrame,
    label_columns: List[str]) -> Dict[str, Any]:
    """
    Analiza correlaciones entre tareas para entender dependencias entre
    perspectivas teóricas.
    
    Args:
        df: DataFrame con las etiquetas
        label_columns: Lista de nombres de columnas de etiquetas
    
    Returns:
        Dictionary con matriz de correlación y análisis:
        {
            'correlation_matrix': np.ndarray,
            'correlation_dict': Dict,
            'strong_correlations': List[Dict],
            'task_independence_score': float,
            'insights': List[str]
        }
    """
    # Calcular matriz de correlación
    label_matrix = df[label_columns].values
    correlation_matrix = np.corrcoef(label_matrix.T)
    
    # Convertir a diccionario para mejor acceso
    correlation_dict = {}
    for i, task_i in enumerate(label_columns):
        correlation_dict[task_i] = {}
        for j, task_j in enumerate(label_columns):
            correlation_dict[task_i][task_j] = float(correlation_matrix[i, j])
    
    # Identificar correlaciones fuertes (>0.5 o <-0.5)
    strong_correlations = []
    for i, task_i in enumerate(label_columns):
        for j, task_j in enumerate(label_columns):
            if i < j:  # Solo mitad superior de la matriz
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.5:
                    interpretation = (
                        'Fuerte correlación positiva' if corr > 0.7 else
                        'Correlación positiva moderada' if corr > 0.5 else
                        'Correlación negativa moderada' if corr < -0.5 else
                        'Fuerte correlación negativa'
                    )
                    
                    strong_correlations.append({
                        'task_pair': (task_i, task_j),
                        'correlation': float(corr),
                        'interpretation': interpretation
                    })
    
    # Score de independencia (promedio de correlaciones absolutas off-diagonal)
    mask = ~np.eye(len(label_columns), dtype=bool)
    task_independence_score = float(1 - np.abs(correlation_matrix[mask]).mean())
    
    # Generar insights
    insights = []
    
    # Insight 1: Tareas más correlacionadas
    if strong_correlations:
        most_correlated = max(strong_correlations, key=lambda x: abs(x['correlation']))
        insights.append(
            f"'{most_correlated['task_pair'][0]}' y '{most_correlated['task_pair'][1]}' "
            f"tienen la correlación más fuerte ({most_correlated['correlation']:.3f})"
        )
    else:
        insights.append("No se detectaron correlaciones fuertes entre tareas (todas <0.5)")
    
    # Insight 2: Independencia general
    if task_independence_score > 0.7:
        insights.append(
            f"Las tareas son en general independientes (score={task_independence_score:.3f}). "
            "Multi-task learning puede ser muy beneficioso."
        )
    elif task_independence_score < 0.3:
        insights.append(
            f"Las tareas están altamente correlacionadas (score={task_independence_score:.3f}). "
            "Considerar task-specific loss weights."
        )
    
    # Insight 3: Correlaciones negativas
    negative_corrs = [c for c in strong_correlations if c['correlation'] < -0.5]
    if negative_corrs:
        insights.append(
            f"Se detectaron {len(negative_corrs)} correlaciones negativas fuertes, "
            "indicando perspectivas teóricas mutuamente excluyentes."
        )
    
    # Insight 4: Tarea más independiente
    avg_abs_corr_per_task = {}
    for i, task in enumerate(label_columns):
        other_corrs = [abs(correlation_matrix[i, j]) for j in range(len(label_columns)) if i != j]
        avg_abs_corr_per_task[task] = np.mean(other_corrs)
    
    most_independent = min(avg_abs_corr_per_task, key=avg_abs_corr_per_task.get)
    insights.append(
        f"'{most_independent}' es la tarea más independiente "
        f"(correlación promedio={avg_abs_corr_per_task[most_independent]:.3f})"
    )
    
    return {
        'correlation_matrix': correlation_matrix.tolist(),
        'correlation_dict': correlation_dict,
        'strong_correlations': strong_correlations,
        'task_independence_score': task_independence_score,
        'avg_abs_correlation_per_task': {k: float(v) for k, v in avg_abs_corr_per_task.items()},
        'insights': insights
    }

def compute_class_weights(
    df: pd.DataFrame,
    label_columns: List[str],
    method: str = 'effective') -> Dict[str, Any]:
    """
    Calcula pesos óptimos para cada clase en cada tarea para usar en loss function.
    
    Args:
        df: DataFrame con las etiquetas
        label_columns: Lista de nombres de columnas de etiquetas
        method: Método de cálculo ('balanced', 'effective', 'focal_inspired')
    
    Returns:
        Dictionary con pesos por tarea:
        {
            'task_name': {
                'weights': {-1: float, 0: float, 1: float},
                'weights_tensor': torch.Tensor,
                'method': str,
                'normalization': str
            },
            'config': {...}
        }
    """
    results = {}
    
    for task in label_columns:
        task_labels = df[task].values
        
        # Contar clases
        unique, counts = np.unique(task_labels, return_counts=True)
        class_counts = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
        
        # Asegurar todas las clases
        for cls in [-1, 0, 1]:
            if cls not in class_counts:
                class_counts[cls] = 1  # Evitar división por 0
        
        n_samples = len(task_labels)
        n_classes = 3
        
        if method == 'balanced':
            # Sklearn-style: n_samples / (n_classes * class_count)
            weights = {
                cls: n_samples / (n_classes * class_counts[cls])
                for cls in [-1, 0, 1]
            }
        
        elif method == 'effective':
            # Effective Number of Samples (Cui et al. 2019)
            beta = 0.9999
            weights = {}
            for cls in [-1, 0, 1]:
                n_cls = class_counts[cls]
                if n_cls == 0:
                    weights[cls] = 1.0
                else:
                    effective_num = (1 - beta**n_cls) / (1 - beta)
                    weights[cls] = 1.0 / effective_num
        
        elif method == 'focal_inspired':
            # Hybrid: suavizar balanced weights
            alpha = 0.25  # Focal loss parameter
            base_weights = {
                cls: n_samples / (n_classes * class_counts[cls])
                for cls in [-1, 0, 1]
            }
            # Aplicar suavizado
            weights = {
                cls: alpha * base_weights[cls] + (1 - alpha)
                for cls in [-1, 0, 1]
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalizar pesos para que sumen a n_classes
        total_weight = sum(weights.values())
        weights_normalized = {
            cls: (weight / total_weight) * n_classes
            for cls, weight in weights.items()
        }
        
        # Crear tensor de PyTorch (orden: -1, 0, 1 → índices 0, 1, 2)
        weights_tensor = torch.tensor(
            [weights_normalized[-1], weights_normalized[0], weights_normalized[1]],
            dtype=torch.float32
        )
        
        results[task] = {
            'weights': weights_normalized,
            'weights_tensor': weights_tensor,
            'weights_array': weights_tensor.numpy().tolist(),
            'method': method,
            'normalization': 'sum_to_num_classes',
            'class_counts': class_counts
        }
    
    # Configuración para uso directo en código
    config = {
        'method': method,
        'usage_example': (
            "# En MultiTaskLongformer.forward():\n"
            "loss_fct = nn.CrossEntropyLoss(weight=class_weights[task_i])\n"
            "# donde class_weights es un dict: {task_name: weights_tensor}"
        ),
        'pytorch_ready': True
    }
    
    results['config'] = config
    
    return results

def identify_critical_samples(
    df: pd.DataFrame,
    label_columns: List[str],
    text_column: str = 'combined_text') -> Dict[str, Any]:
    """
    Identifica muestras críticas o problemáticas para el aprendizaje.
    
    Args:
        df: DataFrame con etiquetas y textos
        label_columns: Lista de nombres de columnas de etiquetas
        text_column: Nombre de columna con texto (para preview)
    
    Returns:
        Dictionary con análisis de casos críticos:
        {
            'extreme_cases': {...},
            'single_positive_per_task': {...},
            'statistics': {...},
            'recommendations': List[str]
        }
    """
    total_samples = len(df)
    label_matrix = df[label_columns].values
    
    # Casos extremos
    all_positive_mask = np.all(label_matrix == 1, axis=1)
    all_negative_mask = np.all(label_matrix == -1, axis=1)
    all_neutral_mask = np.all(label_matrix == 0, axis=1)
    no_positive_mask = ~np.any(label_matrix == 1, axis=1)
    no_negative_mask = ~np.any(label_matrix == -1, axis=1)
    
    all_positive_indices = df[all_positive_mask].index.tolist()
    all_negative_indices = df[all_negative_mask].index.tolist()
    all_neutral_indices = df[all_neutral_mask].index.tolist()
    
    # Obtener previews de texto
    def get_text_preview(indices, max_preview=2):
        if len(indices) == 0:
            return []
        previews = []
        for idx in indices[:max_preview]:
            text = df.loc[idx, text_column] if text_column in df.columns else ""
            preview = text[:100] + "..." if len(text) > 100 else text
            previews.append(preview)
        return previews
    
    extreme_cases = {
        'all_positive': {
            'count': int(all_positive_mask.sum()),
            'percentage': float((all_positive_mask.sum() / total_samples) * 100),
            'indices': all_positive_indices,
            'texts_preview': get_text_preview(all_positive_indices),
            'pattern': tuple([1] * len(label_columns))
        },
        'all_negative': {
            'count': int(all_negative_mask.sum()),
            'percentage': float((all_negative_mask.sum() / total_samples) * 100),
            'indices': all_negative_indices,
            'texts_preview': get_text_preview(all_negative_indices),
            'pattern': tuple([-1] * len(label_columns))
        },
        'all_neutral': {
            'count': int(all_neutral_mask.sum()),
            'percentage': float((all_neutral_mask.sum() / total_samples) * 100),
            'indices': all_neutral_indices,
            'texts_preview': get_text_preview(all_neutral_indices),
            'pattern': tuple([0] * len(label_columns))
        },
        'no_positive_labels': {
            'count': int(no_positive_mask.sum()),
            'percentage': float((no_positive_mask.sum() / total_samples) * 100)
        },
        'no_negative_labels': {
            'count': int(no_negative_mask.sum()),
            'percentage': float((no_negative_mask.sum() / total_samples) * 100)
        }
    }
    
    # Single positive per task (solo una tarea positiva)
    single_positive_per_task = {}
    for i, task in enumerate(label_columns):
        mask = (label_matrix[:, i] == 1) & (np.sum(label_matrix == 1, axis=1) == 1)
        count = mask.sum()
        single_positive_per_task[task] = {
            'count': int(count),
            'percentage': float((count / total_samples) * 100),
            'indices': df[mask].index.tolist()
        }
    
    # Estadísticas generales
    statistics = {
        'samples_with_all_same_label': int(
            all_positive_mask.sum() + all_negative_mask.sum() + all_neutral_mask.sum()
        ),
        'samples_with_mixed_labels': int(
            total_samples - (all_positive_mask.sum() + all_negative_mask.sum() + all_neutral_mask.sum())
        ),
        'avg_positive_labels_per_sample': float(np.mean(np.sum(label_matrix == 1, axis=1))),
        'avg_negative_labels_per_sample': float(np.mean(np.sum(label_matrix == -1, axis=1))),
        'avg_neutral_labels_per_sample': float(np.mean(np.sum(label_matrix == 0, axis=1))),
    }
    
    # Recomendaciones
    recommendations = []
    
    if extreme_cases['all_positive']['count'] < 5:
        recommendations.append(
            f"⚠️ CRÍTICO: Solo {extreme_cases['all_positive']['count']} muestras con todas las etiquetas positivas. "
            "Insuficiente para aprender este patrón. Considerar recolectar más datos."
        )
    
    if extreme_cases['all_negative']['count'] < 5:
        recommendations.append(
            f"⚠️ CRÍTICO: Solo {extreme_cases['all_negative']['count']} muestras con todas las etiquetas negativas. "
            "Insuficiente para aprender este patrón."
        )
    
    for task, data in single_positive_per_task.items():
        if data['count'] < 10:
            recommendations.append(
                f"⚠️ Tarea '{task}': Solo {data['count']} muestras donde es la única etiqueta positiva. "
                "Puede dificultar el aprendizaje de esta perspectiva de forma independiente."
            )
    
    if statistics['samples_with_all_same_label'] > total_samples * 0.5:
        recommendations.append(
            "⚠️ Más del 50% de muestras tienen todas las etiquetas iguales. "
            "Considerar si el problema es realmente multi-task o podría simplificarse."
        )
    
    return {
        'extreme_cases': extreme_cases,
        'single_positive_per_task': single_positive_per_task,
        'statistics': statistics,
        'recommendations': recommendations
    }

def recommend_strategies(
    class_distribution: Dict[str, Any],
    label_combinations: Dict[str, Any],
    split_comparison: Dict[str, Any],
    task_correlations: Dict[str, Any],
    critical_samples: Dict[str, Any],
    config: Any) -> Dict[str, Any]:
    """
    Genera recomendaciones específicas basadas en el análisis completo del dataset.
    
    Args:
        class_distribution: Resultado de analyze_class_distribution()
        label_combinations: Resultado de analyze_label_combinations()
        split_comparison: Resultado de compare_splits_distribution()
        task_correlations: Resultado de analyze_task_correlations()
        critical_samples: Resultado de identify_critical_samples()
        config: TrainingConfig object
    
    Returns:
        Dictionary con recomendaciones y configuraciones sugeridas
    """
    recommendations = {
        'severity_analysis': {},
        'primary_strategy': None,
        'secondary_strategies': [],
        'warnings': [],
        'class_weights_recommended': {},
        'focal_loss_config': None,
        'sampling_strategy': None,
        'cv_strategy': {},
        'augmentation_config': {},
        'evaluation_metrics': [],
        'implementation_notes': []
    }
    
    # Analizar severidad del desbalanceo
    summary = class_distribution['summary']
    max_ir = summary['max_imbalance_ratio']
    mean_ir = summary['mean_imbalance_ratio']
    overall_severity = summary['overall_severity']
    
    recommendations['severity_analysis'] = {
        'overall_severity': overall_severity,
        'max_imbalance_ratio': max_ir,
        'mean_imbalance_ratio': mean_ir,
        'most_problematic_task': summary['most_imbalanced_task']
    }
    
    # Determinar estrategia primaria basada en severidad y tamaño del dataset
    dataset_size = summary['total_samples']
    
    if dataset_size < 500:
        size_category = 'very_small'
    elif dataset_size < 2000:
        size_category = 'small'
    elif dataset_size < 10000:
        size_category = 'medium'
    else:
        size_category = 'large'
    
    # Matriz de decisión
    if overall_severity == 'balanced':
        recommendations['primary_strategy'] = 'no_special_handling'
        recommendations['implementation_notes'].append(
            "Dataset está relativamente balanceado. No se requieren técnicas especiales de balanceo."
        )
    
    elif overall_severity == 'moderate':
        if size_category in ['very_small', 'small']:
            recommendations['primary_strategy'] = 'class_weights_effective'
            recommendations['secondary_strategies'] = [
                'task_specific_augmentation',
                'early_stopping_on_f1'
            ]
        else:
            recommendations['primary_strategy'] = 'class_weights_balanced'
            recommendations['secondary_strategies'] = [
                'oversampling_moderate',
                'stratified_cv'
            ]
    
    elif overall_severity == 'high':
        if size_category == 'very_small':
            recommendations['primary_strategy'] = 'focal_loss'
            recommendations['secondary_strategies'] = [
                'class_weights_effective',
                'aggressive_augmentation',
                'ensemble_models'
            ]
            recommendations['warnings'].append(
                "⚠️ CRÍTICO: Dataset muy pequeño (<500) con desbalanceo alto. "
                "Recolectar más datos es la mejor solución."
            )
        else:
            recommendations['primary_strategy'] = 'focal_loss'
            recommendations['secondary_strategies'] = [
                'class_weights_effective',
                'hybrid_sampling',
                'stratified_cv'
            ]
    
    else:  # severe
        recommendations['primary_strategy'] = 'focal_loss_with_class_weights'
        recommendations['secondary_strategies'] = [
            'aggressive_oversampling',
            'ensemble_models',
            'task_specific_training'
        ]
        recommendations['warnings'].append(
            "🚨 SEVERO: Desbalanceo extremo detectado (IR>20). "
            "Considerar seriamente recolectar más datos de clases minoritarias."
        )
    
    # Configuración de Class Weights
    if recommendations['primary_strategy'] in [
        'class_weights_effective', 'class_weights_balanced', 
        'focal_loss_with_class_weights'
    ]:
        # Calcular weights usando el método apropiado
        method = 'effective' if 'effective' in recommendations['primary_strategy'] else 'balanced'
        recommendations['class_weights_recommended'] = {
            'method': method,
            'note': f"Usar método '{method}' para calcular pesos por tarea",
            'implementation': (
                "weights = compute_class_weights(train_df, label_columns, method='{}')".format(method)
            )
        }
    
    # Configuración de Focal Loss
    if 'focal' in recommendations['primary_strategy'].lower() or \
       'focal_loss' in recommendations['secondary_strategies']:
        # Calcular alpha basado en distribución de clases
        alphas_per_task = {}
        for task in config.label_columns:
            task_dist = class_distribution[task]
            counts = task_dist['class_counts']
            total = sum(counts.values())
            # Alpha inversamente proporcional a frecuencia
            alphas = {
                cls: 1.0 - (count / total) for cls, count in counts.items()
            }
            alphas_per_task[task] = alphas
        
        recommendations['focal_loss_config'] = {
            'gamma': 2.0,  # Standard focal loss parameter
            'alpha_per_task': alphas_per_task,
            'note': "Alpha ajustado por tarea basado en distribución de clases",
            'implementation': (
                "# Implementar FocalLoss en lugar de CrossEntropyLoss\n"
                "# class FocalLoss(nn.Module):\n"
                "#     def __init__(self, alpha, gamma=2.0):\n"
                "#         ...\n"
            )
        }
    
    # Estrategia de Sampling
    if 'sampling' in recommendations['primary_strategy'] or \
       any('sampling' in s for s in recommendations['secondary_strategies']):
        
        if overall_severity in ['high', 'severe']:
            recommendations['sampling_strategy'] = {
                'method': 'hybrid',
                'oversample_factor': 3.0 if overall_severity == 'severe' else 2.0,
                'undersample_factor': 0.7,
                'target_distribution': 'balanced',
                'note': (
                    "Oversample clases minoritarias y undersample mayorías "
                    "para aproximar distribución balanceada"
                )
            }
        else:
            recommendations['sampling_strategy'] = {
                'method': 'moderate_oversample',
                'oversample_factor': 1.5,
                'undersample_factor': 1.0,
                'target_distribution': 'semi_balanced'
            }
    
    # Estrategia de Cross-Validation
    rare_combinations_pct = label_combinations['rare_percentage']
    
    if rare_combinations_pct > 50:
        recommendations['cv_strategy'] = {
            'use_stratified': False,
            'reason': (
                f"{rare_combinations_pct:.1f}% de combinaciones son raras. "
                "Stratified CV puede fallar. Usar KFold estándar."
            ),
            'alternative': 'repeated_random_splits',
            'n_repeats': 3
        }
        recommendations['warnings'].append(
            f"⚠️ {rare_combinations_pct:.1f}% de combinaciones de etiquetas son raras (<1% o <5 samples). "
            "Stratified CV será problemático."
        )
    else:
        recommendations['cv_strategy'] = {
            'use_stratified': True,
            'stratify_on': 'most_imbalanced_task',
            'stratify_task': summary['most_imbalanced_task'],
            'reason': (
                "Suficientes muestras por combinación para hacer stratified CV. "
                f"Estratificar en '{summary['most_imbalanced_task']}' (tarea más desbalanceada)."
            )
        }
    
    # Configuración de Data Augmentation
    tasks_needing_augmentation = []
    for task in config.label_columns:
        task_dist = class_distribution[task]
        if task_dist['minority_percentage'] < 15:  # Clase minoritaria <15%
            tasks_needing_augmentation.append(task)
    
    if tasks_needing_augmentation:
        recommendations['augmentation_config'] = {
            'enabled': True,
            'apply_to_tasks': tasks_needing_augmentation,
            'techniques': ['random_swap', 'random_deletion'],
            'augmentation_probability': 0.3 if overall_severity in ['high', 'severe'] else 0.2,
            'target_classes': [1],  # Solo clases minoritarias positivas
            'augmentation_factor': 2 if overall_severity == 'severe' else 1.5,
            'note': (
                f"Aplicar augmentation a {len(tasks_needing_augmentation)} tareas "
                "con clases minoritarias <15%"
            )
        }
    else:
        recommendations['augmentation_config'] = {
            'enabled': False,
            'reason': "No se detectaron clases minoritarias críticas (<15%)"
        }
    
    # Métricas de Evaluación Recomendadas
    recommendations['evaluation_metrics'] = [
        'f1_macro',  # Principal (sensible a minoritarias)
        'f1_weighted',
        'recall_macro',
        'precision_macro'
    ]
    
    # Agregar métricas por clase para tareas problemáticas
    for task in config.label_columns:
        task_dist = class_distribution[task]
        if task_dist['severity'] in ['high', 'severe']:
            recommendations['evaluation_metrics'].extend([
                f'{task}_f1_per_class',
                f'{task}_recall_per_class',
                f'{task}_precision_per_class'
            ])
    
    # Warnings adicionales basados en análisis
    
    # Check split consistency
    if not split_comparison['overall_consistency']['all_tasks_consistent']:
        recommendations['warnings'].append(
            "⚠️ Distribuciones train/val/test son inconsistentes. "
            "Verificar estrategia de split."
        )
    
    # Check critical samples
    for rec in critical_samples['recommendations']:
        recommendations['warnings'].append(rec)
    
    # Check correlaciones
    if task_correlations['task_independence_score'] < 0.3:
        recommendations['warnings'].append(
            "⚠️ Tareas altamente correlacionadas detectadas. "
            "Considerar usar task-specific loss weights o shared representations más complejas."
        )
        recommendations['implementation_notes'].append(
            "Implementar task_weights en el modelo (ya implementado en MultiTaskLongformer)"
        )
    
    # Notas de implementación específicas
    recommendations['implementation_notes'].extend([
        f"Dataset size: {dataset_size} ({size_category}) - " + 
        ("Priorizar técnicas que no requieran muchos datos" if size_category in ['very_small', 'small'] 
         else "Suficientes datos para técnicas avanzadas"),
        
        f"Objetivo: Maximizar F1-macro",
        
        "Considerar usar early stopping en f1_macro en lugar de accuracy"
    ])
        
    return recommendations


def simple_paraphrase_augmentation(
    df: pd.DataFrame,
    label_columns: List[str],
    text_column: str = 'combined_text',
    target_samples_per_class: int = 50,
    max_augmentation_per_sample: int = 5,
    model_name: str = "milyiyo/paraphraser-spanish-t5-small",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_beams: int = 5,
    temperature: float = 1.2,
    batch_size: int = 4,
    verbose: bool = True) -> pd.DataFrame:
    """
    Args:
        df: DataFrame con textos y etiquetas
        label_columns: Lista de columnas de etiquetas
        text_column: Nombre de columna con texto
        target_samples_per_class: Objetivo de muestras por clase
        max_augmentation_per_sample: Máximo de paráfrasis por muestra original
        model_name: Modelo de HuggingFace a usar
        device: 'cuda' o 'cpu'
        num_beams: Beam search width (3-10 recomendado)
        temperature: >1.0 = más diverso, <1.0 = más conservador
        batch_size: Batch size para generación (ajustar según GPU)
        verbose: Mostrar progreso
    
    Returns:
        DataFrame balanceado con sintéticas
    """
    
    if verbose:
        print("="*70)
        print("🚀 SIMPLE PARAPHRASE AUGMENTATION")
        print(f"   Modelo: {model_name}")
        print(f"   Device: {device}")
        print("="*70)
    
    # Cargar modelo
    print(f"📦 Cargando modelo de parafraseo...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # Analizar necesidades de augmentation
    augmentation_plan = {}
    
    for task in label_columns:
        class_counts = df[task].value_counts().to_dict()
        
        for cls in [-1, 0, 1]:
            current_count = class_counts.get(cls, 0)
            samples_needed = max(0, target_samples_per_class - current_count)
            
            if samples_needed > 0:
                key = (task, cls)
                augmentation_plan[key] = {
                    'current': current_count,
                    'target': target_samples_per_class,
                    'needed': samples_needed
                }
    
    if verbose:
        print(f"📋 Plan de Augmentation:")
        print(f"{'Tarea':<20} {'Clase':<8} {'Actual':<10} {'Target':<10} {'Generar':<10}")
        print("-" * 70)
        
        for (task, cls), plan in augmentation_plan.items():
            print(f"{task:<20} {cls:<8} {plan['current']:<10} "
                  f"{plan['target']:<10} {plan['needed']:<10}")
        
        total_to_generate = sum(p['needed'] for p in augmentation_plan.values())
        print("-" * 70)
        print(f"Total muestras sintéticas a generar: {total_to_generate}")
    
    # Generar muestras sintéticas
    synthetic_samples = []
    
    progress_bar = tqdm(
        augmentation_plan.items(),
        desc="Generando paráfrasis",
        disable=not verbose
    )
    
    for (task, cls), plan in progress_bar:
        # Obtener muestras de esta clase
        class_samples = df[df[task] == cls]
        
        if len(class_samples) == 0:
            if verbose:
                print(f"⚠️ No hay muestras de {task}={cls} para augmentar")
            continue
        
        samples_needed = plan['needed']
        samples_per_original = min(
            max_augmentation_per_sample,
            int(np.ceil(samples_needed / len(class_samples)))
        )
        
        generated_count = 0
        
        # Iterar sobre muestras originales
        for idx, base_sample in class_samples.iterrows():
            if generated_count >= samples_needed:
                break
            
            original_text = base_sample[text_column]
            
            # Generar múltiples paráfrasis con diferentes temperaturas
            for i in range(samples_per_original):
                if generated_count >= samples_needed:
                    break
                
                # Variar temperatura para más diversidad
                temp = temperature + np.random.uniform(-0.2, 0.2)
                temp = max(0.8, min(2.0, temp))  # Clamp entre 0.8 y 2.0
                
                # Generar paráfrasis
                try:
                    inputs = tokenizer(
                        original_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=num_beams,
                            temperature=temp,
                            do_sample=True,
                            top_k=50,
                            top_p=0.95,
                            early_stopping=True,
                            no_repeat_ngram_size=3  # Evitar repeticiones
                        )
                    
                    synthetic_text = tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    
                    # Verificar que no sea idéntico al original
                    if synthetic_text.lower().strip() == original_text.lower().strip():
                        continue
                    
                    # Crear nueva muestra
                    new_sample = base_sample.copy()
                    new_sample[text_column] = synthetic_text
                    new_sample['synthetic'] = True
                    new_sample['synthetic_method'] = 'paraphrase_t5'
                    new_sample['base_sample_id'] = idx
                    new_sample['temperature_used'] = temp
                    
                    synthetic_samples.append(new_sample)
                    generated_count += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Error generando paráfrasis: {e}")
                    continue
        
        progress_bar.set_postfix({
            'task': task,
            'class': cls,
            'generated': generated_count
        })
    
    # Combinar original + sintéticas
    if synthetic_samples:
        df_synthetic = pd.DataFrame(synthetic_samples)
        df_original = df.copy()
        df_original['synthetic'] = False
        df_original['synthetic_method'] = None
        df_original['base_sample_id'] = None
        
        df_balanced = pd.concat([df_original, df_synthetic], ignore_index=True)
    else:
        df_balanced = df.copy()
        df_balanced['synthetic'] = False
    
    # Limpiar memoria
    del model
    torch.cuda.empty_cache()
    
    if verbose:
        print(f"✅ Augmentation completado:")
        print(f"   - Muestras originales: {len(df)}")
        print(f"   - Muestras sintéticas generadas: {len(synthetic_samples)}")
        print(f"   - Total en dataset balanceado: {len(df_balanced)}")
        
        print(f"📊 Distribución final por tarea:")
        for task in label_columns:
            print(f"   {task}:")
            final_dist = df_balanced[task].value_counts().sort_index()
            for cls, count in final_dist.items():
                original_count = (df[task] == cls).sum()
                synthetic_count = count - original_count
                print(f"      Clase {cls:2d}: {count:4d} total "
                      f"({original_count:4d} orig + {synthetic_count:4d} synth)")
    
    print("="*70)
    
    return df_balanced


def validate_synthetic_quality(
    df_original: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    text_column: str = 'combined_text',
    sample_size: int = 10) -> Dict:
    """
    Valida la calidad de las muestras sintéticas generadas.
    
    Args:
        df_original: DataFrame original
        df_synthetic: DataFrame con muestras sintéticas
        text_column: Columna de texto
        sample_size: Número de ejemplos a mostrar
    
    Returns:
        Dictionary con métricas de calidad y ejemplos
    """
    print("="*70)
    print("🔍 VALIDACIÓN DE CALIDAD DE MUESTRAS SINTÉTICAS")
    print("="*70)
    
    synthetic_only = df_synthetic[df_synthetic['synthetic'] == True]
    
    # Métricas básicas
    print(f"📊 Estadísticas:")
    print(f"   - Total sintéticas: {len(synthetic_only)}")
    print(f"   - Métodos usados:")
    
    method_counts = synthetic_only['synthetic_method'].value_counts()
    for method, count in method_counts.items():
        print(f"      • {method}: {count} ({count/len(synthetic_only)*100:.1f}%)")
    
    # Calcular similitud promedio (longitud de texto como proxy simple)
    synthetic_lengths = synthetic_only[text_column].str.len()
    original_lengths = df_original[text_column].str.len()
    
    print(f"📏 Longitud de textos:")
    print(f"   - Original promedio: {original_lengths.mean():.0f} caracteres")
    print(f"   - Sintético promedio: {synthetic_lengths.mean():.0f} caracteres")
    print(f"   - Diferencia: {abs(synthetic_lengths.mean() - original_lengths.mean()):.0f} caracteres")
    
    # Mostrar ejemplos
    print(f"📝 Ejemplos de muestras sintéticas (primeras {sample_size}):")
    print("-" * 70)
    
    for idx, row in synthetic_only.head(sample_size).iterrows():
        base_id = row['base_sample_id']
        original = df_original.loc[base_id, text_column]
        synthetic = row[text_column]
        method = row['synthetic_method']
        
        print(f"Ejemplo {idx + 1} - Método: {method}")
        print(f"Original:  {original[:150]}...")
        print(f"Sintético: {synthetic[:150]}...")
        print("-" * 70)
    
    return {
        'total_synthetic': len(synthetic_only),
        'method_distribution': method_counts.to_dict(),
        'avg_length_original': original_lengths.mean(),
        'avg_length_synthetic': synthetic_lengths.mean()
    }