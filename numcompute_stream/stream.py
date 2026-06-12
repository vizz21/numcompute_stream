import numpy as np
from typing import Any, Optional, Dict

from .metrics import StreamingAccuracy, StreamingConfusionMatrix


class StreamTrainer:
    """
    Manages a streaming Pipeline: incrementally fits it chunk-by-chunk,
    logs per-chunk metrics, sample counts, and memory footprint
    """

    def __init__(self, pipeline: Any, metrics: Optional[Dict[str, Any]] = None):
        self.pipeline = pipeline
        self.metrics: Dict[str, Any] = metrics if metrics is not None else {
            "accuracy": StreamingAccuracy(),
            "confusion_matrix": StreamingConfusionMatrix(),
        }
        self.log: Dict[str, Any] = {
            "chunk_id": [],
            "n_samples": [],
            "memory_bytes": [],      
            "metrics": {m: [] for m in self.metrics},
        }
        self.total_samples_processed: int = 0


    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be NumPy arrays.")
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")


    def update(self, X_chunk: np.ndarray, y_chunk: np.ndarray,
               chunk_id: Optional[int] = None) -> None:
        if chunk_id is None:
            chunk_id = len(self.log["chunk_id"])
        self.fit_chunk(X_chunk, y_chunk, chunk_id)

    def fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray,
                  chunk_id: int) -> None:

        self._validate_data(X_chunk, y_chunk)

        chunk_bytes = X_chunk.nbytes + y_chunk.nbytes

        self.pipeline.partial_fit(X_chunk, y_chunk)

        y_pred_chunk = self.pipeline.predict(X_chunk)

        for m_name, metric_obj in self.metrics.items():
            metric_obj.update(y_chunk, y_pred_chunk)
            self.log["metrics"][m_name].append(metric_obj.result())

        self.total_samples_processed += X_chunk.shape[0]
        self.log["chunk_id"].append(chunk_id)
        self.log["n_samples"].append(X_chunk.shape[0])
        self.log["memory_bytes"].append(chunk_bytes)   # FIX

    def score_chunk(self, X_chunk: np.ndarray,
                    y_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Score one chunk without modifying accumulated value
        """
        self._validate_data(X_chunk, y_chunk)

        final_step = self.pipeline.steps[-1][1]
        fitted = getattr(final_step, "classes_", None) is not None or \
                 getattr(final_step, "root", None) is not None
        if not fitted:
            raise ValueError("Model not fitted yet. Call fit_chunk first.")

        y_pred_chunk = self.pipeline.predict(X_chunk)

        current_scores: Dict[str, Any] = {}
        for m_name, metric_obj in self.metrics.items():
            temp = type(metric_obj)()
            temp.update(y_chunk, y_pred_chunk)
            current_scores[m_name] = temp.result()
        return current_scores

    def get_log(self) -> Dict[str, Any]:
        return self.log

    def reset_metrics(self) -> None:
        for metric_obj in self.metrics.values():
            metric_obj.reset()

    def get_metrics(self) -> Dict[str, Any]:
        return {m: obj.result() for m, obj in self.metrics.items()}