from typing import Callable, Literal, Optional

import numpy as np


def _as_1d_array(x: np.ndarray, name: str = "x") -> np.ndarray:
    """
    Convert input to a 1D NumPy array of floats and validate it.

    Parameters
    x : np.ndarray
        Input array-like object.
    name : str, optional
        Name used in error messages.

    Returns
    np.ndarray
        A 1D NumPy array of shape (n,) with dtype=float.

    Raises
    ValueError
        If the input is not 1-dimensional or is empty.

    Time Complexity
    O(n), where n is the number of elements in x.

    Space Complexity
    O(n), due to array conversion.
    """

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _validate_h(h: float) -> float:
    """
    Validate and normalize a finite-difference step size.

    Parameters
    h : float
        Step size.

    Returns
    float
        Validated positive finite float.

    Raises
    ValueError
        If h is not finite or is non-positive.

    Time Complexity
    O(1)

    Space Complexity
    O(1)
    """
    h_val = float(h)
    if not np.isfinite(h_val) or h_val <= 0.0:
        raise ValueError("h must be a positive finite float.")
    return h_val


def _as_1d_output(y: np.ndarray, name: str = "output") -> np.ndarray:
    """
    Ensure function output is a non-empty 1D NumPy array.

    Parameters
    y : np.ndarray
        Function output.
    name : str, optional
        Name used in error messages.

    Returns
    np.ndarray
        A 1D NumPy array of shape (m,), where m >= 1.

    Raises
    ValueError
        If the output is not scalar or 1D, or is empty.

    Time Complexity
    O(m), where m is the number of output elements.

    Space Complexity
    O(m)
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as_scalar(y: np.ndarray, name: str = "f(x)") -> float:
    """
    Convert input to a scalar float.

    Parameters
    y : np.ndarray
        Function output.
    name : str, optional
        Name used in error messages.

    Returns
    float
        Scalar value.

    Raises
    ValueError
        If the input is not scalar-valued.

    Time Complexity
    O(1)

    Space Complexity
    O(1)
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.size == 1:
        return float(arr[0])
    raise ValueError(f"{name} must be scalar-valued.")


def grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
    method: Literal["central", "forward"] = "central",
) -> np.ndarray:
    """
    Estimate the gradient of a scalar-valued function via finite differences.

    Parameters
    f : Callable[[np.ndarray], float]
    x : np.ndarray
        Input point of shape (n,).
    h : float, optional
        Step size for finite differences.
    method : {"central", "forward"}, optional
        Finite difference scheme:
        - "central": (f(x+h) - f(x-h)) / (2h)
        - "forward": (f(x+h) - f(x)) / h

    Returns
    np.ndarray
        Gradient vector of shape (n,).

    Raises
    ValueError
        If x is not 1D, h is invalid, method is unsupported,
        or f does not return a scalar.

    Time Complexity
    O(n * T_f), where n is dimension of x and T_f is cost of evaluating f.
    Central differences require ~2n evaluations; forward requires ~n+1.

    Space Complexity
    O(n), for storing the gradient and temporary vectors.
    """

    x_arr = _as_1d_array(x)
    h_val = _validate_h(h)

    if method not in {"central", "forward"}:
        raise ValueError("method must be one of: 'central', 'forward'.")

    g = np.empty_like(x_arr, dtype=float)
    if method == "forward":
        fx = _as_scalar(f(x_arr), name="f(x)")

    for i in range(x_arr.size):
        x_plus = x_arr.copy()
        x_plus[i] += h_val

        if method == "central":
            x_minus = x_arr.copy()
            x_minus[i] -= h_val
            f_plus = _as_scalar(f(x_plus), name="f(x + h e_i)")
            f_minus = _as_scalar(f(x_minus), name="f(x - h e_i)")
            g[i] = (f_plus - f_minus) / (2.0 * h_val)
        else:
            f_plus = _as_scalar(f(x_plus), name="f(x + h e_i)")
            g[i] = (f_plus - fx) / h_val

    return g


def jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-5,
    method: Literal["central", "forward"] = "central",
) -> np.ndarray:
    """
    Estimate the Jacobian of a vector-valued function via finite differences.

    Parameters
    F : Callable[[np.ndarray], np.ndarray]
    x : np.ndarray
    h : float, optional
        Step size for finite differences.
    method : {"central", "forward"}, optional
        Finite difference scheme.

    Returns
    np.ndarray
        Jacobian matrix of shape (m, n), where m is output dimension.

    Raises
    ValueError
        If input/output shapes are invalid, method is unsupported,
        or F returns inconsistent output dimensions.

    Time Complexity
    O(n * T_F * k), where k = 2 for central and k = 1 for forward,
    and T_F is the cost of evaluating F.

    Space Complexity
    O(mn), for the Jacobian matrix.
    """
    x_arr = _as_1d_array(x)
    h_val = _validate_h(h)

    if method not in {"central", "forward"}:
        raise ValueError("method must be one of: 'central', 'forward'.")

    Fx = _as_1d_output(F(x_arr), name="F(x)")
    m = Fx.size
    n = x_arr.size
    J = np.empty((m, n), dtype=float)

    for i in range(n):
        x_plus = x_arr.copy()
        x_plus[i] += h_val

        if method == "central":
            x_minus = x_arr.copy()
            x_minus[i] -= h_val
            f_plus = _as_1d_output(F(x_plus), name="F(x + h e_i)")
            f_minus = _as_1d_output(F(x_minus), name="F(x - h e_i)")
            if f_plus.size != m or f_minus.size != m:
                raise ValueError("F must return outputs with consistent dimension.")
            J[:, i] = (f_plus - f_minus) / (2.0 * h_val)
        else:
            f_plus = _as_1d_output(F(x_plus), name="F(x + h e_i)")
            if f_plus.size != m:
                raise ValueError("F must return outputs with consistent dimension.")
            J[:, i] = (f_plus - Fx) / h_val

    return J


def line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    grad_x: Optional[np.ndarray] = None,
    alpha0: float = 1.0,
    c: float = 1e-4,
    tau: float = 0.5,
    max_iter: int = 50,
) -> float:
    """
    Perform Armijo backtracking line search.

    Parameters
    f : Callable[[np.ndarray], float]
    x : np.ndarray
        Current point of shape (n,).
    direction : np.ndarray
        Descent direction of shape (n,).
    grad_x : np.ndarray, optional
        Gradient at x of shape (n,). If None, it is estimated via `grad`.
    alpha0 : float, optional
        Initial step size.
    c : float, optional
        Armijo condition constant in (0, 1).
    tau : float, optional
        Step reduction factor in (0, 1).
    max_iter : int, optional
        Maximum number of backtracking iterations.

    Returns
    float
        Accepted step size. Returns 0.0 if no valid step is found.

    Raises
    ValueError
        If shapes mismatch, parameters are invalid, or direction is not
        a descent direction.

    Time Complexity
    O(max_iter * T_f + T_grad), where T_f is cost of evaluating f and
    T_grad is cost of gradient computation (if needed).

    Space Complexity
    O(n), for temporary vectors.
    """
    x_arr = _as_1d_array(x)
    p = _as_1d_array(direction, name="direction")
    if p.shape != x_arr.shape:
        raise ValueError("direction must have the same shape as x.")

    alpha = float(alpha0)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha0 must be a positive finite float.")
    if not (0.0 < c < 1.0):
        raise ValueError("c must be in (0, 1).")
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1).")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    fx = _as_scalar(f(x_arr), name="f(x)")

    if grad_x is None:
        g = grad(f, x_arr)
    else:
        g = _as_1d_array(grad_x, name="grad_x")
        if g.shape != x_arr.shape:
            raise ValueError("grad_x must have the same shape as x.")

    directional_derivative = float(np.dot(g, p))
    if directional_derivative >= 0.0:
        raise ValueError(
            "direction must be a descent direction (grad_x dot direction < 0)."
        )

    for _ in range(max_iter):
        candidate = x_arr + alpha * p
        f_candidate = _as_scalar(f(candidate), name="f(x + alpha * direction)")
        if f_candidate <= fx + c * alpha * directional_derivative:
            return alpha
        alpha *= tau

    return 0.0
