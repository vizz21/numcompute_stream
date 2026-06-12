from pathlib import Path
import numpy as np
import pytest
from numcompute_stream.optim import grad, jacobian, line_search


def test_grad_quadratic_central() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])
    g = grad(f, x)
    assert np.allclose(g, [2.0, 4.0, 6.0], atol=1e-4)

def test_grad_quadratic_forward() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0, 3.0])
    g = grad(f, x, method="forward")
    assert np.allclose(g, [2.0, 4.0, 6.0], atol=1e-3)

def test_grad_invalid_method_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="method"):
        grad(f, x, method="backward")

def test_grad_empty_array_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    with pytest.raises(ValueError, match="non-empty"):
        grad(f, np.array([]))

def test_grad_2d_array_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    with pytest.raises(ValueError, match="1D"):
        grad(f, np.array([[1.0, 2.0]]))

def test_grad_invalid_h_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        grad(f, x, h=-1e-5)

def test_jacobian_shape() -> None:
    F = lambda x: np.array([x[0] ** 2, x[1] ** 2])
    x = np.array([1.0, 2.0])
    J = jacobian(F, x)
    assert J.shape == (2, 2)

def test_jacobian_known_values() -> None:
    F = lambda x: np.array([x[0] ** 2, x[1] ** 2])
    x = np.array([1.0, 2.0])
    J = jacobian(F, x)
    expected = np.array([[2.0, 0.0],
                         [0.0, 4.0]])
    assert np.allclose(J, expected, atol=1e-4)


def test_jacobian_invalid_method_raises() -> None:
    F = lambda x: np.array([x[0] ** 2])
    x = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="method"):
        jacobian(F, x, method="backward")

def test_jacobian_empty_array_raises() -> None:
    F = lambda x: np.array([x[0] ** 2])
    with pytest.raises(ValueError, match="non-empty"):
        jacobian(F, np.array([]))

def test_line_search_returns_positive_step() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    direction = np.array([-1.0, -2.0])
    alpha = line_search(f, x, direction)
    assert alpha > 0.0


def test_line_search_non_descent_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    direction = np.array([1.0, 2.0])  
    with pytest.raises(ValueError, match="descent"):
        line_search(f, x, direction)


def test_line_search_invalid_alpha_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    direction = np.array([-1.0, -2.0])
    with pytest.raises(ValueError, match="alpha0"):
        line_search(f, x, direction, alpha0=-1.0)

def test_line_search_shape_mismatch_raises() -> None:
    f = lambda x: np.sum(x ** 2)
    x = np.array([1.0, 2.0])
    direction = np.array([-1.0, -2.0, -3.0])  # wrong shape!
    with pytest.raises(ValueError, match="shape"):
        line_search(f, x, direction)