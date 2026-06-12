import numpy as np
from typing import List, Optional
from .tree import StreamingDecisionTreeClassifier


class EnsembleClassifier:
    def partial_fit(self, X, y, classes=None):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

class RandomForestClassifier(EnsembleClassifier):
    """
    Random Forest for streaming data. 
    Trains n_estimators trees, each on a bootstrap sample and random feature subset of each incoming chunk
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 criterion: str = "gini", max_features="sqrt",
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)

        self.trees: List[StreamingDecisionTreeClassifier] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.feature_subsets_: Optional[List[np.ndarray]] = None



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

    def _init_trees_and_subsets(self, n_features: int) -> None:
        """Initialise trees and feature-subset arrays on first call."""
        self.n_features_ = n_features
        k = self._n_features_to_try(n_features)
        self.feature_subsets_ = [
            self._rng.choice(n_features, k, replace=False)
            for _ in range(self.n_estimators)
        ]
        self.trees = [
            StreamingDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            )
            for _ in range(self.n_estimators)
        ]


    def partial_fit(self, X: np.ndarray, y: np.ndarray,
                    classes: Optional[np.ndarray] = None
                    ) -> "RandomForestClassifier":
        """
        Incrementally fit each tree on a bootstrap sample of the chunk.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")


        chunk_classes = np.unique(y) if classes is None else np.unique(classes)
        if self.classes_ is None:
            self.classes_ = chunk_classes
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, chunk_classes]))


        if self.n_features_ is None:
            self._init_trees_and_subsets(X.shape[1])
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        n = X.shape[0]
        for i, tree in enumerate(self.trees):

            idx = self._rng.choice(n, n, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            X_sub = X_boot[:, self.feature_subsets_[i]]

            tree.classes_ = self.classes_
            tree.partial_fit(X_sub, y_boot)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Majority-vote prediction.

        FIX: vectorised majority vote using np.apply_along_axis +
        np.bincount instead of a Python Counter loop.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if not self.trees:
            raise ValueError("RandomForestClassifier not fitted yet.")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )


        all_preds = np.array([
            tree.predict(X[:, self.feature_subsets_[i]])
            for i, tree in enumerate(self.trees)
        ])  
        n_samples = X.shape[0]
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        int_preds = np.vectorize(lambda v: class_to_idx.get(v, 0))(all_preds)

        winner_idx = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=len(self.classes_)).argmax(),
            axis=0,
            arr=int_preds,
        )
        return self.classes_[winner_idx]

    def fit(self, X: np.ndarray, y: np.ndarray,
            classes: Optional[np.ndarray] = None) -> "RandomForestClassifier":
        """Full fit: resets state then delegates to partial_fit"""
        self.trees = []
        self.classes_ = None
        self.n_features_ = None
        self.feature_subsets_ = None
        self._rng = np.random.default_rng(self.random_state)
        self.partial_fit(X, y, classes=classes)
        return self