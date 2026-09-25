"""Polynomial and piecewise interpolation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _check(xs: Sequence[float], ys: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("xs and ys must be 1-D sequences of equal length")
    if len(np.unique(x)) != len(x):
        raise ValueError("interpolation nodes must be distinct")
    return x, y


def lagrange(xs: Sequence[float], ys: Sequence[float], t: float | np.ndarray) -> float | np.ndarray:
    """Evaluate the Lagrange interpolating polynomial at t."""
    x, y = _check(xs, ys)
    t_arr = np.asarray(t, dtype=float)
    result = np.zeros_like(t_arr)
    for j in range(len(x)):
        basis = np.ones_like(t_arr)
        for m in range(len(x)):
            if m != j:
                basis *= (t_arr - x[m]) / (x[j] - x[m])
        result += y[j] * basis
    return float(result) if result.ndim == 0 else result


def newton_divided_differences(xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
    """Return the Newton form coefficients f[x0], f[x0,x1], ..., f[x0..xn]."""
    x, y = _check(xs, ys)
    coef = y.copy()
    n = len(x)
    for level in range(1, n):
        coef[level:] = (coef[level:] - coef[level - 1 : -1]) / (x[level:] - x[: n - level])
    return coef


def newton_eval(xs: Sequence[float], coef: np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
    """Evaluate a Newton-form polynomial with Horner's scheme."""
    x = np.asarray(xs, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    result = np.full_like(t_arr, coef[-1])
    for k in range(len(coef) - 2, -1, -1):
        result = result * (t_arr - x[k]) + coef[k]
    return float(result) if result.ndim == 0 else result


def linear_spline(xs: Sequence[float], ys: Sequence[float], t: float | np.ndarray) -> float | np.ndarray:
    """Piecewise linear interpolation. Nodes are sorted automatically."""
    x, y = _check(xs, ys)
    order = np.argsort(x)
    x, y = x[order], y[order]
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr < x[0]) or np.any(t_arr > x[-1]):
        raise ValueError("t is outside the interpolation range")
    idx = np.clip(np.searchsorted(x, t_arr) - 1, 0, len(x) - 2)
    w = (t_arr - x[idx]) / (x[idx + 1] - x[idx])
    result = (1 - w) * y[idx] + w * y[idx + 1]
    return float(result) if result.ndim == 0 else result
