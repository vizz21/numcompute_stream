from typing import Optional, Sequence, Tuple, List
import numpy as np


def _validate_1d_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name_pred: str = "y_pred",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate two 1-D arrays with matching shapes.

    Returns
    -------
    (y_true, y_pred) as np.ndarray

    Raises
    ------
    ValueError for 1-D, empty, or length-mismatch violations.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and {name_pred} must have the same length "
            f"({yt.shape[0]} vs {yp.shape[0]})."
        )
    return yt, yp



def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _validate_1d_pair(y_true, y_pred)
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence] = None,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    FIX: vectorised via np.add.at — no Python element loop.

    Entry (i, j) = count where true label is labels[i] and
    predicted label is labels[j].
    """
    yt, yp = _validate_1d_pair(y_true, y_pred)

    if labels is None:
        label_values = np.unique(np.concatenate([yt, yp]))
    else:
        label_values = np.asarray(labels)
        if label_values.ndim != 1 or label_values.size == 0:
            raise ValueError("labels must be a non-empty 1-D sequence.")

    n = label_values.size
    matrix = np.zeros((n, n), dtype=int)

    label_to_idx = {lbl: i for i, lbl in enumerate(label_values.tolist())}

    yt_idx = np.array([label_to_idx.get(v, -1) for v in yt.tolist()])
    yp_idx = np.array([label_to_idx.get(v, -1) for v in yp.tolist()])

    valid = (yt_idx >= 0) & (yp_idx >= 0)
    np.add.at(matrix, (yt_idx[valid], yp_idx[valid]), 1)
    return matrix


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label=1,
    zero_division: float = 0.0,
) -> float:
    yt, yp = _validate_1d_pair(y_true, y_pred)
    tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
    fp = int(np.sum((yt != pos_label) & (yp == pos_label)))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else float(zero_division)


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label=1,
    zero_division: float = 0.0,
) -> float:
    yt, yp = _validate_1d_pair(y_true, y_pred)
    tp = int(np.sum((yt == pos_label) & (yp == pos_label)))
    fn = int(np.sum((yt == pos_label) & (yp != pos_label)))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float(zero_division)


def f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label=1,
    zero_division: float = 0.0,
) -> float:
    p = precision(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    r = recall(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0 else float(zero_division)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _validate_1d_pair(y_true, y_pred)
    diff = yt.astype(float) - yp.astype(float)
    return float(np.mean(diff ** 2))


def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label=1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt, ys = _validate_1d_pair(y_true, y_score, name_pred="y_score")
    ys = ys.astype(float)
    y_pos = yt == pos_label
    positives = int(np.sum(y_pos))
    negatives = int(yt.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("roc_curve requires both positive and negative samples.")

    order = np.argsort(-ys, kind="stable")
    ys_sorted = ys[order]
    y_sorted = y_pos[order]

    distinct_idx = np.where(np.diff(ys_sorted))[0]
    threshold_idx = np.r_[distinct_idx, yt.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (threshold_idx + 1) - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, ys_sorted[threshold_idx]]

    tpr = tps / positives
    fpr = fps / negatives
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    x_arr, y_arr = _validate_1d_pair(x, y, name_pred="y")
    if x_arr.size < 2:
        raise ValueError("x and y must contain at least two points.")
    order = np.argsort(x_arr, kind="stable")
    return float(np.trapezoid(y_arr[order].astype(float),
                              x_arr[order].astype(float)))



class StreamingMetric:
    """
    Base class for metrics that update incrementally chunk-by-chunk
    """

    def __init__(self, window_size: Optional[int] = None) -> None:
        self.window_size = window_size

    def update(self, y_true_chunk: np.ndarray,
               y_pred_chunk: np.ndarray) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def result(self):
        raise NotImplementedError




class StreamingAccuracy(StreamingMetric):
    """
    Cumulative accuracy over all chunks
    """

    def __init__(self, window_size: Optional[int] = None) -> None:
        super().__init__(window_size)
        self._correct: int = 0
        self._total: int = 0
        self._window_correct: List[int] = []
        self._window_total: List[int] = []

    def update(self, y_true_chunk: np.ndarray,
               y_pred_chunk: np.ndarray) -> None:
        yt, yp = _validate_1d_pair(y_true_chunk, y_pred_chunk)
        chunk_correct = int(np.sum(yt == yp))
        chunk_total = len(yt)

        if self.window_size is not None:
            self._window_correct.append(chunk_correct)
            self._window_total.append(chunk_total)

            while sum(self._window_total) > self.window_size:
                self._window_correct.pop(0)
                self._window_total.pop(0)
        else:
            self._correct += chunk_correct
            self._total += chunk_total

    def reset(self) -> None:
        self._correct = 0
        self._total = 0
        self._window_correct = []
        self._window_total = []

    def result(self) -> float:
        if self.window_size is not None:
            total = sum(self._window_total)
            return sum(self._window_correct) / total if total > 0 else 0.0
        return self._correct / self._total if self._total > 0 else 0.0



class StreamingConfusionMatrix(StreamingMetric):
    """
    Confusion matrix accumulated over chunks
    """

    def __init__(self, labels: Optional[Sequence] = None,
                 window_size: Optional[int] = None) -> None:
        super().__init__(window_size)
        self._user_labels = labels is not None
        self.labels: Optional[np.ndarray] = (
            np.asarray(labels) if labels is not None else None
        )

        if self.labels is not None:
            n = len(self.labels)
            self._matrix: Optional[np.ndarray] = np.zeros((n, n), dtype=int)
        else:
            self._matrix = None

    def _ensure_matrix(self, seen_labels: np.ndarray) -> None:
        if self.labels is None:
            self.labels = seen_labels
            self._matrix = np.zeros(
                (len(self.labels), len(self.labels)), dtype=int
            )
            return

        existing_set = set(self.labels.tolist())
        new_only = [v for v in seen_labels.tolist() if v not in existing_set]
        if not new_only:
            return  

        expanded = np.concatenate([self.labels, np.asarray(new_only)])
        new_size = len(expanded)
        new_matrix = np.zeros((new_size, new_size), dtype=int)
        old_size = len(self.labels)
        if self._matrix is not None:
            new_matrix[:old_size, :old_size] = self._matrix
        self.labels = expanded
        self._matrix = new_matrix

    def update(self, y_true_chunk: np.ndarray,
               y_pred_chunk: np.ndarray) -> None:
        yt, yp = _validate_1d_pair(y_true_chunk, y_pred_chunk)
        seen = np.unique(np.concatenate([yt, yp]))
        self._ensure_matrix(seen)

        label_to_idx = {lbl: i for i, lbl in enumerate(self.labels.tolist())}
        yt_idx = np.array([label_to_idx.get(v, -1) for v in yt.tolist()])
        yp_idx = np.array([label_to_idx.get(v, -1) for v in yp.tolist()])
        valid = (yt_idx >= 0) & (yp_idx >= 0)
        np.add.at(self._matrix, (yt_idx[valid], yp_idx[valid]), 1)

    def reset(self) -> None:
        self._matrix = None

        if not self._user_labels:
            self.labels = None

    def result(self) -> np.ndarray:
        if self._matrix is None:
            return np.array([], dtype=int)
        return self._matrix.copy()


class StreamingPrecision(StreamingMetric):
    """
    Cumulative precision 
    """

    def __init__(self, pos_label=1, zero_division: float = 0.0,
                 window_size: Optional[int] = None) -> None:
        super().__init__(window_size)
        self.pos_label = pos_label
        self.zero_division = zero_division
        self._tp: int = 0
        self._fp: int = 0
        self._window: List[Tuple[int, int]] = []  
    def update(self, y_true_chunk: np.ndarray,
               y_pred_chunk: np.ndarray) -> None:
        yt, yp = _validate_1d_pair(y_true_chunk, y_pred_chunk)
        tp = int(np.sum((yt == self.pos_label) & (yp == self.pos_label)))
        fp = int(np.sum((yt != self.pos_label) & (yp == self.pos_label)))

        if self.window_size is not None:
            self._window.append((tp, fp))
            while sum(t + f for t, f in self._window) > self.window_size:
                self._window.pop(0)
        else:
            self._tp += tp
            self._fp += fp

    def reset(self) -> None:
        self._tp = 0
        self._fp = 0
        self._window = []

    def result(self) -> float:
        if self.window_size is not None:
            tp = sum(t for t, _ in self._window)
            fp = sum(f for _, f in self._window)
        else:
            tp, fp = self._tp, self._fp
        denom = tp + fp
        return float(tp / denom) if denom > 0 else float(self.zero_division)


class StreamingRecall(StreamingMetric):
    """
    Cumulative recall
    """

    def __init__(self, pos_label=1, zero_division: float = 0.0,
                 window_size: Optional[int] = None) -> None:
        super().__init__(window_size)
        self.pos_label = pos_label
        self.zero_division = zero_division
        self._tp: int = 0
        self._fn: int = 0
        self._window: List[Tuple[int, int]] = [] 

    def update(self, y_true_chunk: np.ndarray,
               y_pred_chunk: np.ndarray) -> None:
        yt, yp = _validate_1d_pair(y_true_chunk, y_pred_chunk)
        tp = int(np.sum((yt == self.pos_label) & (yp == self.pos_label)))
        fn = int(np.sum((yt == self.pos_label) & (yp != self.pos_label)))

        if self.window_size is not None:
            self._window.append((tp, fn))
            while sum(t + f for t, f in self._window) > self.window_size:
                self._window.pop(0)
        else:
            self._tp += tp
            self._fn += fn

    def reset(self) -> None:
        self._tp = 0
        self._fn = 0
        self._window = []

    def result(self) -> float:
        if self.window_size is not None:
            tp = sum(t for t, _ in self._window)
            fn = sum(f for _, f in self._window)
        else:
            tp, fn = self._tp, self._fn
        denom = tp + fn
        return float(tp / denom) if denom > 0 else float(self.zero_division)


class StreamingF1(StreamingMetric):
    """
    Cumulative F1
    """

    def __init__(self, pos_label=1, zero_division: float = 0.0,
                 window_size: Optional[int] = None) -> None:
        super().__init__(window_size)
        self._prec = StreamingPrecision(
            pos_label=pos_label, zero_division=zero_division,
            window_size=window_size
        )
        self._rec = StreamingRecall(
            pos_label=pos_label, zero_division=zero_division,
            window_size=window_size
        )
        self.zero_division = zero_division

    def update(self, y_true_chunk: np.ndarray,
               y_pred_chunk: np.ndarray) -> None:
        self._prec.update(y_true_chunk, y_pred_chunk)
        self._rec.update(y_true_chunk, y_pred_chunk)

    def reset(self) -> None:
        self._prec.reset()
        self._rec.reset()

    def result(self) -> float:
        p = self._prec.result()
        r = self._rec.result()
        denom = p + r
        return float(2.0 * p * r / denom) if denom > 0 else float(self.zero_division)


__all__ = [
    "accuracy", "confusion_matrix", "precision", "recall", "f1",
    "mse", "roc_curve", "auc",
    "StreamingAccuracy", "StreamingConfusionMatrix",
    "StreamingPrecision", "StreamingRecall", "StreamingF1",
]