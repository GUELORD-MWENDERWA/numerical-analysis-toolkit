"""Direct and iterative solvers for linear systems A x = b."""

from __future__ import annotations

import numpy as np


def gauss_solve(a, b) -> np.ndarray:
    """Gaussian elimination with partial pivoting, then back substitution."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    n = len(b)
    if a.shape != (n, n):
        raise ValueError("A must be square and match b")
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(a[k:, k])))
        if a[p, k] == 0:
            raise np.linalg.LinAlgError("matrix is singular")
        if p != k:
            a[[k, p]] = a[[p, k]]
            b[[k, p]] = b[[p, k]]
        factors = a[k + 1 :, k] / a[k, k]
        a[k + 1 :, k:] -= np.outer(factors, a[k, k:])
        b[k + 1 :] -= factors * b[k]
    if a[n - 1, n - 1] == 0:
        raise np.linalg.LinAlgError("matrix is singular")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - a[i, i + 1 :] @ x[i + 1 :]) / a[i, i]
    return x


def lu_decompose(a) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU factorisation with partial pivoting: P A = L U."""
    a = np.array(a, dtype=float)
    n = a.shape[0]
    perm = np.eye(n)
    lower = np.eye(n)
    upper = a.copy()
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(upper[k:, k])))
        if upper[p, k] == 0:
            raise np.linalg.LinAlgError("matrix is singular")
        if p != k:
            upper[[k, p], k:] = upper[[p, k], k:]
            perm[[k, p]] = perm[[p, k]]
            lower[[k, p], :k] = lower[[p, k], :k]
        for i in range(k + 1, n):
            lower[i, k] = upper[i, k] / upper[k, k]
            upper[i, k:] -= lower[i, k] * upper[k, k:]
    return perm, lower, upper


def lu_solve(perm, lower, upper, b) -> np.ndarray:
    """Solve A x = b from a factorisation returned by lu_decompose."""
    pb = perm @ np.asarray(b, dtype=float).reshape(-1)
    n = len(pb)
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - lower[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - upper[i, i + 1 :] @ x[i + 1 :]) / upper[i, i]
    return x


def jacobi(a, b, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> tuple[np.ndarray, int]:
    """Jacobi iteration. Converges for strictly diagonally dominant matrices."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float)
    d = np.diag(a)
    if np.any(d == 0):
        raise ValueError("zero on the diagonal")
    r = a - np.diagflat(d)
    for i in range(1, max_iter + 1):
        x_new = (b - r @ x) / d
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, i
        x = x_new
    raise RuntimeError(f"Jacobi did not converge in {max_iter} iterations")


def gauss_seidel(a, b, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> tuple[np.ndarray, int]:
    """Gauss-Seidel iteration. Usually converges about twice as fast as Jacobi."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()
    n = len(b)
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = a[i, :i] @ x[:i] + a[i, i + 1 :] @ x_old[i + 1 :]
            x[i] = (b[i] - s) / a[i, i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, it
    raise RuntimeError(f"Gauss-Seidel did not converge in {max_iter} iterations")
