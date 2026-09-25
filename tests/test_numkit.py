import math

import numpy as np
import pytest

import numkit as nk


def f(x):
    return x**3 - 2 * x - 5  # classic Newton example, root ~ 2.0945514815


ROOT = 2.0945514815423265


def test_bisection():
    r = nk.bisection(f, 2, 3, tol=1e-12)
    assert r.converged and abs(r.root - ROOT) < 1e-10


def test_bisection_requires_sign_change():
    with pytest.raises(ValueError):
        nk.bisection(f, 3, 4)


def test_newton_and_secant_converge_faster_than_bisection():
    rn = nk.newton(f, lambda x: 3 * x**2 - 2, 2.0)
    rs = nk.secant(f, 2.0, 3.0)
    rb = nk.bisection(f, 2, 3, tol=1e-12)
    assert abs(rn.root - ROOT) < 1e-12 and abs(rs.root - ROOT) < 1e-12
    assert rn.iterations < rs.iterations < rb.iterations


def test_fixed_point_cosine():
    r = nk.fixed_point(math.cos, 1.0)
    assert r.converged and abs(r.root - math.cos(r.root)) < 1e-10


def test_interpolation_reproduces_polynomial():
    xs = [0, 1, 2, 3]
    ys = [x**3 - x + 1 for x in xs]
    coef = nk.newton_divided_differences(xs, ys)
    for t in (0.5, 1.7, 2.9):
        expected = t**3 - t + 1
        assert nk.lagrange(xs, ys, t) == pytest.approx(expected)
        assert nk.newton_eval(xs, coef, t) == pytest.approx(expected)


def test_linear_spline():
    assert nk.linear_spline([0, 2, 1], [0, 4, 1], 1.5) == pytest.approx(2.5)


def test_quadrature_rules():
    exact = 2.0  # integral of sin on [0, pi]
    assert nk.trapezoid(np.sin, 0, math.pi, 200) == pytest.approx(exact, abs=1e-4)
    assert nk.simpson(np.sin, 0, math.pi, 50) == pytest.approx(exact, abs=1e-6)
    assert nk.romberg(np.sin, 0, math.pi) == pytest.approx(exact, abs=1e-9)
    assert nk.gauss_legendre(np.sin, 0, math.pi, 8) == pytest.approx(exact, abs=1e-9)


def test_gauss_legendre_is_exact_for_polynomials():
    assert nk.gauss_legendre(lambda x: x**5 + x**2, -1, 2, 3) == pytest.approx(2**6 / 6 - 1 / 6 + 3)


def test_ode_orders():
    rhs = lambda t, y: -2 * y  # y = exp(-2t)
    exact = math.exp(-2)
    errors = {}
    for name, solver in (("euler", nk.euler), ("heun", nk.heun), ("rk4", nk.rk4)):
        _, y = solver(rhs, 1.0, 0.0, 1.0, 0.05)
        errors[name] = abs(y[-1, 0] - exact)
    assert errors["rk4"] < errors["heun"] < errors["euler"]
    assert errors["rk4"] < 1e-6


def test_ode_system_harmonic_oscillator():
    rhs = lambda t, y: np.array([y[1], -y[0]])
    t, y = nk.rk4(rhs, [1.0, 0.0], 0.0, 2 * math.pi, 0.01)
    assert y[-1, 0] == pytest.approx(math.cos(t[-1]), abs=1e-8)


A = np.array([[10.0, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
B = np.array([6.0, 25, -11, 15])
X = np.array([1.0, 2, -1, 1])


def test_direct_solvers():
    np.testing.assert_allclose(nk.gauss_solve(A, B), X)
    p, l, u = nk.lu_decompose(A)
    np.testing.assert_allclose(p @ A, l @ u)
    np.testing.assert_allclose(nk.lu_solve(p, l, u, B), X)


def test_iterative_solvers():
    xj, nj = nk.jacobi(A, B)
    xg, ng = nk.gauss_seidel(A, B)
    np.testing.assert_allclose(xj, X, atol=1e-8)
    np.testing.assert_allclose(xg, X, atol=1e-8)
    assert ng < nj


def test_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        nk.gauss_solve([[1, 2], [2, 4]], [1, 2])
