"""numkit: numerical analysis methods implemented from first principles."""

from .roots import bisection, newton, secant, fixed_point, RootResult
from .interpolation import lagrange, newton_divided_differences, newton_eval, linear_spline
from .integration import trapezoid, simpson, romberg, gauss_legendre
from .ode import euler, heun, rk4
from .linalg import gauss_solve, lu_decompose, lu_solve, jacobi, gauss_seidel

__all__ = [
    "RootResult", "bisection", "newton", "secant", "fixed_point",
    "lagrange", "newton_divided_differences", "newton_eval", "linear_spline",
    "trapezoid", "simpson", "romberg", "gauss_legendre",
    "euler", "heun", "rk4",
    "gauss_solve", "lu_decompose", "lu_solve", "jacobi", "gauss_seidel",
]
__version__ = "0.1.0"
