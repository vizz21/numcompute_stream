from typing import Any, List, Sequence, Tuple


class Pipeline:
    """
    Chains preprocessing steps and a final estimator.
    Supports both batch (fit/predict) and streaming (partial_fit) workflows.
    """

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("steps must contain at least one (name, object) pair.")
        self.steps: List[Tuple[str, Any]] = list(steps)
        self._validate_steps()

    def _validate_steps(self) -> None:
        seen: set = set()
        for i, (name, step) in enumerate(self.steps):
            if not isinstance(name, str) or not name:
                raise ValueError("Each step name must be a non-empty string.")
            if name in seen:
                raise ValueError(f"Duplicate step name: {name!r}.")
            seen.add(name)
            if not hasattr(step, "fit"):
                raise ValueError(f"Step {name!r} must implement fit().")
            is_last = i == len(self.steps) - 1
            if not is_last and not hasattr(step, "transform"):
                raise ValueError(
                    f"Non-final step {name!r} must implement transform()."
                )
            if is_last and not (hasattr(step, "transform")
                                or hasattr(step, "predict")):
                raise ValueError(
                    f"Final step {name!r} must implement transform() or predict()."
                )



    def _transform_all_but_last(self, X: Any) -> Any:
        x = X
        for _, step in self.steps[:-1]:
            x = step.transform(x)
        return x


    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        x = X
        for i, (_, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last:
                step.fit(x, y) if y is not None else step.fit(x)
            else:
                if hasattr(step, "fit_transform"):
                    x = step.fit_transform(x)
                else:
                    step.fit(x)
                    x = step.transform(x)
        return self

    def transform(self, X: Any) -> Any:
        x = X
        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last and not hasattr(step, "transform"):
                raise ValueError(
                    f"Final step {name!r} does not implement transform(); "
                    "use predict() instead."
                )
            x = step.transform(x)
        return x

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        self.fit(X, y=y)
        return self.transform(X)

    def predict(self, X: Any) -> Any:
        x = self._transform_all_but_last(X)
        name, final = self.steps[-1]
        if not hasattr(final, "predict"):
            raise ValueError(
                f"Final step {name!r} does not implement predict()."
            )
        return final.predict(x)


    def partial_fit(self, X: Any, y: Any = None) -> "Pipeline":
        """
        Incrementally fit every step on one chunk
        """
        x = X
        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1

            if hasattr(step, "partial_fit"):
                if is_last:
                    step.partial_fit(x, y) if y is not None else step.partial_fit(x)
                else:
                    step.partial_fit(x)
            elif hasattr(step, "fit"):
                if is_last:
                    step.fit(x, y) if y is not None else step.fit(x)
                else:
                    step.fit(x)
            else:
                raise ValueError(
                    f"Step {name!r} must implement partial_fit() or fit()."
                )

            if not is_last:
                if hasattr(step, "partial_transform"):
                    x = step.partial_transform(x)
                elif hasattr(step, "transform"):
                    x = step.transform(x)
                else:
                    raise ValueError(
                        f"Non-final step {name!r} must implement "
                        "transform() or partial_transform() for streaming."
                    )
        return self


__all__ = ["Pipeline"]