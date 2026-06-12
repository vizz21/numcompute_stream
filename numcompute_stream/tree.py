import numpy as np
from collections import Counter
from typing import Optional


class Node:
    def __init__(self, feature_idx=None, threshold=None,
                 left=None, right=None, value=None,
                 is_leaf=False, n_samples=0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf
        self.stats = Counter() 
        self.n_samples = n_samples

    @property
    def prediction(self):
        if self.is_leaf:
            return self.value
        if self.stats:
            return self.stats.most_common(1)[0][0]
        return None

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(value={self.value}, n_samples={self.n_samples})"
        return (f"Node(feature={self.feature_idx}, "
                f"threshold={self.threshold}, n_samples={self.n_samples})")


class StreamingDecisionTreeClassifier:
    """
    Depth-limited decision tree classifier supporting Gini or entropy splits
    """

    _DEFAULT_MAX_STORED = 5_000

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = "gini",
                 max_features=None,
                 max_stored_samples: int = _DEFAULT_MAX_STORED):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.max_stored_samples = max_stored_samples

        self.root: Optional[Node] = None
        self.n_features: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self._X_seen: Optional[np.ndarray] = None
        self._y_seen: Optional[np.ndarray] = None



    def _gini(self, y: np.ndarray, classes: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        counts = np.array([np.sum(y == c) for c in classes])
        p = counts / y.size
        return float(1.0 - np.sum(p ** 2))

    def _entropy(self, y: np.ndarray, classes: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        counts = np.array([np.sum(y == c) for c in classes])
        p = counts / y.size
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    def _impurity(self, y: np.ndarray, classes: np.ndarray) -> float:
        if self.criterion == "gini":
            return self._gini(y, classes)
        elif self.criterion == "entropy":
            return self._entropy(y, classes)
        raise ValueError(f"Unknown criterion: {self.criterion!r}")


    def _n_features_to_try(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        if isinstance(mf, int):
            return min(mf, n_features)
        raise ValueError(f"Unknown max_features value: {mf!r}")


    def _best_split_for_feature(self, col: np.ndarray, y: np.ndarray,
                                 classes: np.ndarray):
        order = np.argsort(col, kind="stable")
        col_sorted = col[order]
        y_sorted = y[order]
        n = len(y)

        change_pts = np.where(np.diff(col_sorted))[0] 
        if change_pts.size == 0:
            return None, float("inf")

        best_thr = None
        best_imp = float("inf")

        left_counts = np.zeros(len(classes), dtype=float)

        total_counts = np.array([np.sum(y == c) for c in classes], dtype=float)

        prev_idx = 0
        for split_idx in change_pts:
            for i in range(prev_idx, split_idx + 1):
                cls_pos = np.searchsorted(classes, y_sorted[i])
                if cls_pos < len(classes) and classes[cls_pos] == y_sorted[i]:
                    left_counts[cls_pos] += 1
            prev_idx = split_idx + 1

            n_left = split_idx + 1
            n_right = n - n_left

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            right_counts = total_counts - left_counts

            if self.criterion == "gini":
                p_l = left_counts / n_left
                p_r = right_counts / n_right
                imp_l = 1.0 - float(np.sum(p_l ** 2))
                imp_r = 1.0 - float(np.sum(p_r ** 2))
            else: 
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_l = np.where(left_counts > 0, left_counts / n_left, 0.0)
                    p_r = np.where(right_counts > 0, right_counts / n_right, 0.0)
                    imp_l = float(-np.sum(np.where(p_l > 0, p_l * np.log2(p_l), 0.0)))
                    imp_r = float(-np.sum(np.where(p_r > 0, p_r * np.log2(p_r), 0.0)))

            w_imp = (n_left / n) * imp_l + (n_right / n) * imp_r
            if w_imp < best_imp:
                best_imp = w_imp
                best_thr = float(col_sorted[split_idx])

        return best_thr, best_imp

    def _find_best_split(self, X: np.ndarray, y: np.ndarray,
                          classes: np.ndarray):
        n_features = X.shape[1]
        k = self._n_features_to_try(n_features)
        feature_indices = np.random.choice(n_features, k, replace=False)

        best_feat = None
        best_thr = None
        best_imp = float("inf")

        for fi in feature_indices:
            thr, imp = self._best_split_for_feature(X[:, fi], y, classes)
            if thr is not None and imp < best_imp:
                best_imp = imp
                best_feat = fi
                best_thr = thr

        return best_feat, best_thr, best_imp

    def _predict_leaf(self, y: np.ndarray):
        if y.size == 0:
            return None
        return Counter(y.tolist()).most_common(1)[0][0]

    def _build_tree(self, X: np.ndarray, y: np.ndarray,
                    depth: int, classes: np.ndarray) -> Node:
        n_samples = X.shape[0]
        unique_y = np.unique(y)

        if (depth >= self.max_depth
                or n_samples < self.min_samples_split
                or len(unique_y) == 1):
            node = Node(value=self._predict_leaf(y),
                        is_leaf=True, n_samples=n_samples)
            node.stats.update(y.tolist())
            return node

        feat, thr, _ = self._find_best_split(X, y, classes)

        if feat is None:
            node = Node(value=self._predict_leaf(y),
                        is_leaf=True, n_samples=n_samples)
            node.stats.update(y.tolist())
            return node

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, classes)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, classes)

        node = Node(feature_idx=feat, threshold=thr,
                    left=left_child, right=right_child, n_samples=n_samples)
        node.stats.update(y.tolist())
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StreamingDecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y, depth=0, classes=self.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if self.root is None:
            raise ValueError("Model not fitted yet.")

        n = X.shape[0]
        predictions = np.empty(n, dtype=object)
        self._predict_vectorised(X, np.arange(n), self.root, predictions)
        return predictions

    def _predict_vectorised(self, X: np.ndarray, indices: np.ndarray,
                             node: Node, out: np.ndarray) -> None:
        if node.is_leaf or indices.size == 0:
            out[indices] = node.value
            return
        left_mask = X[indices, node.feature_idx] <= node.threshold
        self._predict_vectorised(X, indices[left_mask], node.left, out)
        self._predict_vectorised(X, indices[~left_mask], node.right, out)

    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes: Optional[np.ndarray] = None
                    ) -> "StreamingDecisionTreeClassifier":
        if classes is not None:
            self.classes_ = np.unique(
                np.concatenate([self.classes_, classes])
                if self.classes_ is not None else classes
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.size == 0:
            raise ValueError("Input data X cannot be empty.")
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self._X_seen is None:
            self._X_seen = X
            self._y_seen = y
        else:
            self._X_seen = np.vstack((self._X_seen, X))
            self._y_seen = np.concatenate((self._y_seen, y))

        if self._X_seen.shape[0] > self.max_stored_samples:
            self._X_seen = self._X_seen[-self.max_stored_samples:]
            self._y_seen = self._y_seen[-self.max_stored_samples:]

        self.fit(self._X_seen, self._y_seen)
        return self