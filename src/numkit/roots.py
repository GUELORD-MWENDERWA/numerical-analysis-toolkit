"""Root-finding methods for scalar equations f(x) = 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Func = Callable[[float], float]


@dataclass
class RootResult:
    root: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def bisection(f: Func, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> RootResult:
    """Bisection method. Requires f(a) and f(b) to have opposite signs.

    Converges linearly; the error bound halves at every iteration.
    """
    fa, fb = f(a), f(b)
    if fa == 0:
        return RootResult(a, 0, True, [a])
    if fb == 0:
        return RootResult(b, 0, True, [b])
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history: list[float] = []
    for i in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        history.append(m)
        if fm == 0 or (b - a) / 2 < tol:
            return RootResult(m, i, True, history)
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return RootResult((a + b) / 2, max_iter, False, history)


def newton(f: Func, df: Func, x0: float, tol: float = 1e-12, max_iter: int = 100) -> RootResult:
    """Newton-Raphson method. Quadratic convergence near a simple root."""
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        d = df(x)
        if d == 0:
            raise ZeroDivisionError(f"zero derivative at x={x}")
        x_new = x - f(x) / d
        history.append(x_new)
        if abs(x_new - x) < tol:
            return RootResult(x_new, i, True, history)
        x = x_new
    return RootResult(x, max_iter, False, history)


def secant(f: Func, x0: float, x1: float, tol: float = 1e-12, max_iter: int = 100) -> RootResult:
    """Secant method. Superlinear convergence (order about 1.618) without a derivative."""
    history = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for i in range(1, max_iter + 1):
        if f1 == f0:
            raise ZeroDivisionError("flat secant: f(x0) == f(x1)")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            return RootResult(x2, i, True, history)
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    return RootResult(x1, max_iter, False, history)


def fixed_point(g: Func, x0: float, tol: float = 1e-12, max_iter: int = 500) -> RootResult:
    """Fixed-point iteration x_{n+1} = g(x_n). Converges when |g'(x*)| < 1."""
    x = x0
    history = [x]
    for i in range(1, max_iter + 1):
        x_new = g(x)
        history.append(x_new)
        if abs(x_new - x) < tol:
            return RootResult(x_new, i, True, history)
        x = x_new
    return RootResult(x, max_iter, False, history)
