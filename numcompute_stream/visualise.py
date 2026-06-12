import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence

def _show_or_save(save_path: Optional[str]) -> None:
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metric_over_time(metric_values: List[float], title: str, ylabel: str, save_path: Optional[str] = None):
    """   
    Plot a single metric
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metric_values, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(title)
    plt.xlabel("Chunk Index")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    _show_or_save(save_path)

def compare_models(metric1: List[float], metric2: List[float], labels: List[str], save_path: Optional[str] = None):
    """
    Compare two models on streaming metrics
    """
    if len(labels) < 2:
        raise ValueError("labels must contain at least two names.")
        
    plt.figure(figsize=(10, 5))
    plt.plot(metric1, marker='o', label=labels[0], linewidth=2)
    plt.plot(metric2, marker='s', label=labels[1], linestyle='--', linewidth=2)
    plt.title("Model Comparison")
    plt.xlabel("Chunk Index")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    _show_or_save(save_path)

def plot_predictions_vs_ground_truth(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None):
    """
    Visualise predictions vs. actuals on the latest chunk
    """
    plt.figure(figsize=(12, 4))
    indices = np.arange(len(y_true))
    
    plt.scatter(indices, y_true, color='blue', label='Ground Truth', alpha=0.6, s=50)
    plt.scatter(indices, y_pred, color='red', marker='x', label='Predictions', alpha=0.8, s=50)
    
    plt.title("Predictions vs Ground Truth (Latest Chunk)")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _show_or_save(save_path)

def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str = "Confusion Matrix", save_path: Optional[str] = None):

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    _show_or_save(save_path)
