# numkit: Numerical Analysis Toolkit

![tests](https://github.com/GUELORD-MWENDERWA/numerical-analysis-toolkit/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![license](https://img.shields.io/badge/license-MIT-green)

Classical numerical analysis methods implemented from first principles in Python and NumPy, with tests that check both correctness and the expected convergence behaviour of each method.

The library follows the syllabus of a university course in numerical methods (*Méthodes et Analyse Numérique*, L2 Computer Science and AI) and is meant to be read as much as used: every function is short, documented with its convergence order, and free of hidden calls to SciPy.

## Contents

| Module | Methods | Notes |
| --- | --- | --- |
| `roots` | bisection, Newton-Raphson, secant, fixed-point iteration | Each returns a `RootResult` with the iteration history for convergence plots |
| `interpolation` | Lagrange, Newton divided differences (Horner evaluation), piecewise linear | Vectorised evaluation |
| `integration` | composite trapezoid, composite Simpson, Romberg, Gauss-Legendre | Gauss-Legendre is exact up to degree 2n-1 |
| `ode` | forward Euler, Heun (RK2), classical RK4 | Scalar and vector systems |
| `linalg` | Gaussian elimination with partial pivoting, LU (PA = LU), Jacobi, Gauss-Seidel | Singular-matrix detection |

## Installation

```bash
git clone https://github.com/GUELORD-MWENDERWA/numerical-analysis-toolkit.git
cd numerical-analysis-toolkit
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
import numkit as nk

# Root of x^3 - 2x - 5
r = nk.newton(lambda x: x**3 - 2*x - 5, lambda x: 3*x**2 - 2, x0=2.0)
print(r.root, r.iterations)            # 2.0945514815423265 5

# Integral of sin on [0, pi]
nk.simpson(np.sin, 0, np.pi, n=50)     # 2.00000017325...

# Harmonic oscillator y'' + y = 0 as a first-order system
t, y = nk.rk4(lambda t, y: np.array([y[1], -y[0]]), [1.0, 0.0], 0, 10, h=0.01)

# Linear system
nk.gauss_solve([[4, 1], [2, 3]], [1, 2])
```

### Convergence comparison

`examples/convergence_study.py` solves the same equation with three root finders:

```
bisection  root=2.094551481542 iterations=40
secant     root=2.094551481542 iterations=7
newton     root=2.094551481542 iterations=5
```

This illustrates linear, superlinear (about 1.618) and quadratic convergence.

## Testing

```bash
pytest
```

The suite checks results against analytical solutions and asserts the ordering of errors between methods of different order (for example, RK4 error < Heun error < Euler error at the same step size).

## Design choices

- Pure NumPy, no SciPy: the point is to see the algorithm.
- Iterative methods expose `tol` and `max_iter` and report whether they converged instead of silently returning a wrong value.
- Direct solvers pivot to stay stable on matrices with small leading entries.

## Roadmap

- Cubic splines and least-squares fitting
- Adaptive step size (RK45) and implicit Euler for stiff problems
- Power iteration for dominant eigenvalues

## License

MIT. See [LICENSE](LICENSE).
