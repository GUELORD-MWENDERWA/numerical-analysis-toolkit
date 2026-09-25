"""Compare how fast bisection, secant and Newton converge on the same equation."""

import numkit as nk


def f(x):
    return x**3 - 2 * x - 5


def df(x):
    return 3 * x**2 - 2


for name, result in (
    ("bisection", nk.bisection(f, 2, 3, tol=1e-12)),
    ("secant", nk.secant(f, 2, 3)),
    ("newton", nk.newton(f, df, 2)),
):
    print(f"{name:10s} root={result.root:.12f} iterations={result.iterations}")
