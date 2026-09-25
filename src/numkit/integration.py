"""Numerical quadrature on a closed interval [a, b]."""

from __future__ import annotations

from typing import Callable

import numpy as np

Func = Callable[[np.ndarray], np.ndarray]


def trapezoid(f: Func, a: float, b: float, n: int = 100) -> float:
    """Composite trapezoidal rule. Error O(h^2)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2))


def simpson(f: Func, a: float, b: float, n: int = 100) -> float:
    """Composite Simpson 1/3 rule. n must be even. Error O(h^4)."""
    if n < 2 or n % 2:
        raise ValueError("n must be an even integer >= 2")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1]))


def romberg(f: Func, a: float, b: float, levels: int = 6) -> float:
    """Romberg integration: Richardson extrapolation of the trapezoidal rule."""
    r = np.zeros((levels, levels))
    for i in range(levels):
        r[i, 0] = trapezoid(f, a, b, 2**i)
        for j in range(1, i + 1):
            r[i, j] = r[i, j - 1] + (r[i, j - 1] - r[i - 1, j - 1]) / (4**j - 1)
    return float(r[levels - 1, levels - 1])


def gauss_legendre(f: Func, a: float, b: float, points: int = 5) -> float:
    """Gauss-Legendre quadrature. Exact for polynomials of degree <= 2*points - 1."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    return float(0.5 * (b - a) * np.dot(weights, f(x)))
