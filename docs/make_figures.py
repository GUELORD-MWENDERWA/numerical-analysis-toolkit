"""Regenerate the figures in docs/images from the library itself.

    pip install -e . matplotlib
    python docs/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import numkit as nk

OUT = Path(__file__).resolve().parent / "images"
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def root_convergence() -> None:
    f = lambda x: x**3 - 2 * x - 5
    df = lambda x: 3 * x**2 - 2
    exact = nk.newton(f, df, 2, tol=1e-15).root
    runs = {
        "bisection (linear)": nk.bisection(f, 2, 3, tol=1e-14),
        "secant (order 1.618)": nk.secant(f, 2, 3, tol=1e-14),
        "Newton (quadratic)": nk.newton(f, df, 2, tol=1e-14),
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, r in runs.items():
        err = np.abs(np.array(r.history) - exact)
        k = np.flatnonzero(err > 0)  # an exact hit has no place on a log scale
        ax.semilogy(k, err[k], "o-", ms=3.5, label=f"{name}, {r.iterations} iterations")
    ax.set(xlabel="iteration", ylabel="|x_k - x*|", title="Root of x^3 - 2x - 5 = 0: error per iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "root_convergence.png")


def ode_accuracy() -> None:
    f = lambda t, y: -2 * y + np.sin(t)
    exact = lambda t: (2 * np.sin(t) - np.cos(t)) / 5 + 6 / 5 * np.exp(-2 * t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    tt = np.linspace(0, 5, 400)
    a1.plot(tt, exact(tt), "k", lw=2, label="exact")
    for name, method in (("Euler", nk.euler), ("Heun", nk.heun), ("RK4", nk.rk4)):
        t, y = method(f, 1.0, 0, 5, 0.25)
        a1.plot(t, y[:, 0], "o--", ms=3, label=f"{name}, h = 0.25")
        hs = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
        errs = [abs(method(f, 1.0, 0, 5, h)[1][-1, 0] - exact(5)) for h in hs]
        a2.loglog(hs, errs, "o-", label=name)
    a1.set(xlabel="t", ylabel="y", title="y' = -2y + sin t, y(0) = 1")
    a1.legend()
    a2.set(xlabel="step h", ylabel="global error at t = 5", title="Observed order: slopes 1, 2 and 4")
    a2.set_xticks(hs, [f"{h:g}" for h in hs], minor=False)
    a2.minorticks_off()
    a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ode_accuracy.png")


def interpolation() -> None:
    runge = lambda x: 1 / (1 + 25 * x**2)
    t = np.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, runge(t), "k", lw=2, label="f(x) = 1 / (1 + 25x^2)")
    xe = np.linspace(-1, 1, 11)
    xc = np.cos((2 * np.arange(11) + 1) * np.pi / 22)
    ax.plot(t, nk.lagrange(xe, runge(xe), t), label="Lagrange, 11 equispaced nodes")
    ax.plot(t, nk.lagrange(xc, runge(xc), t), label="Lagrange, 11 Chebyshev nodes")
    ax.plot(t, nk.linear_spline(xe, runge(xe), t), "--", label="linear spline")
    ax.set(ylim=(-1.0, 1.6), xlabel="x", title="Runge phenomenon and how node placement removes it")
    ax.legend(loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "interpolation_runge.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    root_convergence()
    ode_accuracy()
    interpolation()
