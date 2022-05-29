"""Tests for the API boundary."""
import jax
import jax.numpy as jnp
import pytest_cases

from tornadox import ivp_examples, solve
from tornadox.solvers import ek1


def case_ivp_vanderpol_nonstiff():

    f, tspan, u0 = ivp_examples.vanderpol(stiffness_constant=1.0)
    return f, tspan, u0


@pytest_cases.parametrize("num_derivatives", [3, 6])
def case_solver_ek1(num_derivatives):

    # Assume that the test-IVPs are two-dimensional.
    # When this changes, update here.
    ode_dimension = 2

    return ek1.ek1_terminal_value(
        ode_dimension=ode_dimension, num_derivatives=num_derivatives
    )


@pytest_cases.parametrize_with_cases(argnames=("ivp",), cases=".", prefix="case_ivp_")
@pytest_cases.parametrize_with_cases(
    argnames=("solver",), cases=".", prefix="case_solver_"
)
@pytest_cases.parametrize("atol, rtol", ((1e-6, 1e-3),))
def test_solve_ivp_terminal_value(ivp, solver, atol, rtol):
    f, tspan, u0 = ivp
    df = jax.jacfwd(f)
    f, df = jax.jit(f), jax.jit(df)

    t, (m, c_sqrtm), _ = solve.solve_ivp_for_terminal_value(
        f=f, df=df, tspan=tspan, u0=u0, solver=solver, atol=atol, rtol=rtol
    )
    assert m.ndim >= 1
    assert c_sqrtm.ndim >= 2

    assert jnp.allclose(t, tspan[1]), t - tspan[1]
