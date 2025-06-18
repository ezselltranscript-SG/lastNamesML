"""
Evaluation module for surname correction project.
Implements metrics and evaluation functions for surname correction models.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os


def evaluate_model(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Evaluate model predictions using standard classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_surname_correction(original_surnames: List[str], 
                               corrected_surnames: List[str], 
                               true_surnames: List[str]) -> Dict[str, float]:
    """
    Evaluate surname correction performance.
    
    Args:
        original_surnames: Original (potentially misspelled) surnames
        corrected_surnames: Model-corrected surnames
        true_surnames: True correct surnames
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Count corrections
    total = len(original_surnames)
    needed_correction = sum(1 for orig, true in zip(original_surnames, true_surnames) 
                           if orig.lower() != true.lower())
    
    # Count correct predictions
    correct_before = sum(1 for orig, true in zip(original_surnames, true_surnames) 
                        if orig.lower() == true.lower())
    correct_after = sum(1 for corr, true in zip(corrected_surnames, true_surnames) 
                       if corr.lower() == true.lower())
    
    # Calculate improvement
    improvement = correct_after - correct_before
    
    # Calculate metrics
    accuracy_before = correct_before / total if total > 0 else 0
    accuracy_after = correct_after / total if total > 0 else 0
    
    correction_rate = improvement / needed_correction if needed_correction > 0 else 0
    
    return {
        'total_samples': total,
        'needed_correction': needed_correction,
        'correct_before': correct_before,
        'correct_after': correct_after,
        'improvement': improvement,
        'accuracy_before': accuracy_before,
        'accuracy_after': accuracy_after,
        'correction_rate': correction_rate
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         save_path: str = None) -> None:
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except ImportError:
        print("Matplotlib or seaborn not available for plotting")


def plot_metrics(metrics_list: List[Dict[str, float]], 
                model_names: List[str], 
                save_path: str = None) -> None:
    """
    Plot comparison of metrics across different models.
    
    Args:
        metrics_list: List of metric dictionaries for each model
        model_names: Names of the models
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Extract metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set width of bars
        bar_width = 0.2
        index = np.arange(len(model_names))
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            values = [m.get(metric, 0) for m in metrics_list]
            ax.bar(index + i * bar_width, values, bar_width, label=metric)
        
        # Add labels and legend
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")


def cross_validate(model_class: Any, X: Any, y: Any, 
                   n_splits: int = 5, **model_kwargs) -> Dict[str, float]:
    """
    Perform cross-validation for model evaluation.
    
    Args:
        model_class: Model class to instantiate
        X: Feature matrix (numpy array or list)
        y: Target labels (numpy array or list)
        n_splits: Number of cross-validation splits
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Dictionary of average evaluation metrics
    """
    from sklearn.model_selection import KFold
    import numpy as np
    
    # Convert inputs to numpy arrays if they aren't already
    X_array = np.array(X) if not isinstance(X, np.ndarray) else X
    y_array = np.array(y) if not isinstance(y, np.ndarray) else y
    
    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics
    all_metrics = []
    
    # Perform cross-validation
    for train_idx, test_idx in kf.split(X_array):
        # Split data
        X_train = [X[i] for i in train_idx] if isinstance(X, list) else X_array[train_idx]
        X_test = [X[i] for i in test_idx] if isinstance(X, list) else X_array[test_idx]
        y_train = [y[i] for i in train_idx] if isinstance(y, list) else y_array[train_idx]
        y_test = [y[i] for i in test_idx] if isinstance(y, list) else y_array[test_idx]
        
        # Train model
        model = model_class(**model_kwargs)
        model.train(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        all_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = sum(m[metric] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics


def generate_evaluation_report(metrics: Dict[str, float], 
                              model_name: str, 
                              output_path: str = None) -> str:
    """
    Generate a text report of evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        output_path: Path to save the report (optional)
        
    Returns:
        Report text
    """
    # Format the report
    report = f"Evaluation Report for {model_name}\n"
    report += "=" * 50 + "\n\n"
    
    # Add metrics
    for metric, value in metrics.items():
        report += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


def evaluate_correction_examples(original_surnames: List[str],
                               corrected_surnames: List[str],
                               true_surnames: List[str],
                               n_examples: int = 10) -> List[Dict[str, str]]:
    """
    Generate examples of surname corrections for qualitative evaluation.
    
    Args:
        original_surnames: Original (potentially misspelled) surnames
        corrected_surnames: Model-corrected surnames
        true_surnames: True correct surnames
        n_examples: Number of examples to generate
        
    Returns:
        List of example dictionaries
    """
    examples = []
    
    # Track indices of different types of examples
    correct_corrections = []
    incorrect_corrections = []
    unnecessary_corrections = []
    missed_corrections = []
    
    for i, (orig, corr, true) in enumerate(zip(original_surnames, corrected_surnames, true_surnames)):
        orig_lower = orig.lower()
        corr_lower = corr.lower()
        true_lower = true.lower()
        
        if orig_lower != true_lower and corr_lower == true_lower:
            # Correct correction
            correct_corrections.append(i)
        elif orig_lower != true_lower and corr_lower != true_lower:
            # Incorrect correction
            incorrect_corrections.append(i)
        elif orig_lower == true_lower and corr_lower != true_lower:
            # Unnecessary correction
            unnecessary_corrections.append(i)
        elif orig_lower != true_lower and corr_lower == orig_lower:
            # Missed correction
            missed_corrections.append(i)
    
    # Sample examples from each category
    categories = [
        ('Correct Correction', correct_corrections),
        ('Incorrect Correction', incorrect_corrections),
        ('Unnecessary Correction', unnecessary_corrections),
        ('Missed Correction', missed_corrections)
    ]
    
    import random
    
    for category_name, indices in categories:
        # Sample up to n_examples/4 from each category
        sample_size = min(n_examples // 4, len(indices))
        if sample_size == 0:
            continue
            
        sampled_indices = random.sample(indices, sample_size)
        
        for idx in sampled_indices:
            examples.append({
                'category': category_name,
                'original': original_surnames[idx],
                'corrected': corrected_surnames[idx],
                'true': true_surnames[idx]
            })
    
    return examples
