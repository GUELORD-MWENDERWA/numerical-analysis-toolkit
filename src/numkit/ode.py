"""Explicit one-step solvers for initial value problems y' = f(t, y), y(t0) = y0.

All solvers accept scalar or vector states and return (t, y) arrays.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def _grid(t0: float, t1: float, h: float) -> np.ndarray:
    if h <= 0:
        raise ValueError("step h must be positive")
    n = int(round((t1 - t0) / h))
    return np.linspace(t0, t0 + n * h, n + 1)


def _solve(step, f: RHS, y0, t0: float, t1: float, h: float):
    t = _grid(t0, t1, h)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = step(f, t[i], y[i], h)
    return t, y


def _euler_step(f, t, y, h):
    return y + h * np.asarray(f(t, y))


def _heun_step(f, t, y, h):
    k1 = np.asarray(f(t, y))
    k2 = np.asarray(f(t + h, y + h * k1))
    return y + h / 2 * (k1 + k2)


def _rk4_step(f, t, y, h):
    k1 = np.asarray(f(t, y))
    k2 = np.asarray(f(t + h / 2, y + h / 2 * k1))
    k3 = np.asarray(f(t + h / 2, y + h / 2 * k2))
    k4 = np.asarray(f(t + h, y + h * k3))
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler(f: RHS, y0, t0: float, t1: float, h: float):
    """Forward Euler. Global error O(h)."""
    return _solve(_euler_step, f, y0, t0, t1, h)


def heun(f: RHS, y0, t0: float, t1: float, h: float):
    """Heun (improved Euler, RK2). Global error O(h^2)."""
    return _solve(_heun_step, f, y0, t0, t1, h)


def rk4(f: RHS, y0, t0: float, t1: float, h: float):
    """Classical fourth-order Runge-Kutta. Global error O(h^4)."""
    return _solve(_rk4_step, f, y0, t0, t1, h)
