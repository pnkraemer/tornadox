"""Tests for solve_saveat()."""
import jax
import jax.numpy as jnp
import pytest_cases
import scipy.integrate

from tornadox import ivp_examples, solve
from tornadox.blocks import markov
from tornadox.solvers import ek1


def case_ivp_vanderpol_nonstiff():
    f, tspan, u0 = ivp_examples.vanderpol(stiffness_constant=0.1)
    return f, tspan, u0


@pytest_cases.parametrize("num_derivatives", [3, 6])
def case_solver_ek1(num_derivatives):

    # Assume that the test-IVPs are two-dimensional.
    # When this changes, update here.
    ode_dimension = 2

    return ek1.ek1_saveat(ode_dimension=ode_dimension, num_derivatives=num_derivatives)


@pytest_cases.parametrize_with_cases(argnames=("ivp",), cases=".", prefix="case_ivp_")
@pytest_cases.parametrize_with_cases(
    argnames=("solver",), cases=".", prefix="case_solver_"
)
@pytest_cases.parametrize("atol, rtol", ((1e-6, 1e-3),))
def test_solve_ivp_saveat(ivp, solver, atol, rtol):
    f, tspan, u0 = ivp
    df = jax.jacfwd(f)
    f, df = jax.jit(f), jax.jit(df)

    t0, t1 = tspan
    saveat = jnp.array([t0, 0.75 * (t0 + t1), t1])

    solution = solve.solve_ivp_saveat(
        f=f, df=df, saveat=saveat, u0=u0, solver=solver, atol=atol, rtol=rtol
    )
    # Each of those is of the form ((rv, backward_model), t)
    state_terminal, solution_stacked, state_initial = solution

    # Uhmmm, each output (terminal, stacked, initial) looks so innocent,
    # but they are _packed_ full of information.
    # Long list of assertions incoming...
    # This probably implies that the output needs some refactoring,
    # but for now I am happy with it, because it does the job.

    # Assert the validity of terminal_state
    # Asserting shape[0] > 4 implies that the array is _not_
    # a batch of arrays, where each element of the batch corresponds
    # to a "saveat" location.
    t_terminal, rv_terminal, bw_terminal, _ = state_terminal
    m_terminal, c_sqrtm_terminal = rv_terminal  # random variable
    a_terminal, (b_terminal, d_sqrtm_terminal) = bw_terminal  # backward model
    assert m_terminal.shape[0] > saveat.shape[0]
    assert c_sqrtm_terminal.shape[0] > saveat.shape[0]
    assert a_terminal.shape[0] > saveat.shape[0]
    assert b_terminal.shape[0] > saveat.shape[0]
    assert d_sqrtm_terminal.shape[0] > saveat.shape[0]
    assert t_terminal.shape == ()
    assert jnp.allclose(t_terminal, saveat[-1])

    # Assert the validity of the solution_stacked objects.
    # Note how for N saveat() locations, they have leading dimension N-1:
    # the "zeroth" element is the initial state, which is returned separately,
    # mostly due to scan() semantics (but also because it is often not needed).
    t_full, rv_full, bw_full, _ = solution_stacked
    m_full, c_sqrtm_full = rv_full
    a_full, (b_full, d_sqrtm_full) = bw_full
    assert m_full.shape[0] == saveat.shape[0] - 1
    assert c_sqrtm_full.shape[0] == saveat.shape[0] - 1
    assert a_full.shape[0] == saveat.shape[0] - 1
    assert b_full.shape[0] == saveat.shape[0] - 1
    assert d_sqrtm_full.shape[0] == saveat.shape[0] - 1
    assert t_full.shape[0] == saveat.shape[0] - 1
    assert jnp.allclose(t_full, saveat[1:])
    # Common issue when forgetting to reset the backward transition
    assert not jnp.allclose(m_full[0, 0], u0[0])
    # The terminal values must coincide
    assert jnp.allclose(m_full[-1, :], m_terminal)

    # Assert the validity of the initial state object.
    # This is the same procedure as for the terminal state stuff.
    t_initial, rv_initial, bw_initial, stats = state_initial
    m_initial, c_sqrtm_initial = rv_initial  # random variable
    a_initial, (b_initial, d_sqrtm_initial) = bw_initial  # backward model
    assert m_initial.shape[0] > saveat.shape[0]
    assert c_sqrtm_initial.shape[0] > saveat.shape[0]
    assert a_initial.shape[0] > saveat.shape[0]
    assert b_initial.shape[0] > saveat.shape[0]
    assert d_sqrtm_initial.shape[0] > saveat.shape[0]
    assert t_initial.shape == ()
    assert jnp.allclose(t_initial, saveat[0])
    # The initial value must work.
    # We reshape, because some solvers return (n,d) means.
    # The reshaping order ("F", "C") does not matter, because
    # we select the zeroth element.
    assert jnp.allclose(m_initial.reshape((-1,))[0], u0[0])

    # Let's also look at some backward marginals
    (m0, c_sqrtm0), (ms, c_sqrtms) = markov.marginalize(
        m0=m_terminal,
        c_sqrtm0=c_sqrtm_terminal,
        A=a_full,
        b=b_full,
        Q_sqrtm=d_sqrtm_full,
        reverse=True,
    )
    assert ms.shape[0] == saveat.shape[0] - 1
    assert c_sqrtms.shape[0] == saveat.shape[0] - 1
    assert m0.shape[0] > saveat.shape[0]
    assert c_sqrtm0.shape[0] > saveat.shape[0]
    # Assert reverse=True.
    # If reverse=False, the initial values would be wrong:
    assert jnp.allclose(m0.reshape((-1,))[0], u0[0])
    assert jnp.allclose(ms[0].reshape((-1,))[0], u0[0])

    # Test the values by comparing to scipy:
    scipy_solution = scipy.integrate.solve_ivp(
        fun=lambda _, y: f(y),
        t_span=tspan,
        y0=u0,
        method="LSODA",
        jac=lambda _, y: df(y),
        t_eval=saveat,
    )
    # atol and rtol are kinda sharp,
    # (chosen as strict as possible for solutions that satisfy the "eye-test")
    # so if you add new solvers they might be reconsidered.
    assert jnp.allclose(
        ms.reshape((2, -1))[:, 0], scipy_solution.y.T[:-1, 0], atol=1e-2, rtol=1e-4
    )
