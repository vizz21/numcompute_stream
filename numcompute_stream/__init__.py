from .pipeline import Pipeline
from .tree import StreamingDecisionTreeClassifier
from .ensemble import RandomForestClassifier, EnsembleClassifier
from .stream import StreamTrainer
from .preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
from .metrics import accuracy, confusion_matrix, precision, recall, f1, mse, roc_curve, auc, StreamingAccuracy, StreamingConfusionMatrix
from .stats import mean, variance, welford, histogram, quantile, StreamingMeanVariance
from .io import load_csv
from .utils import get_logger
from .visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix
)

__all__ = [
    "Pipeline",
    "StreamingDecisionTreeClassifier",
    "RandomForestClassifier",
    "EnsembleClassifier",
    "StreamTrainer",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    "roc_curve",
    "auc",
    "StreamingAccuracy",
    "StreamingConfusionMatrix",
    "mean",
    "variance",
    "welford",
    "histogram",
    "quantile",
    "StreamingMeanVariance",
    "load_csv",
    "get_logger",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_confusion_matrix",
]
