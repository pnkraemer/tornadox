import numpy as np
import pytest
import jax.numpy as jnp
from scipy.integrate import solve_ivp


import tornado
from tornado.ek0 import vec_trick_mul_full, vec_trick_mul_right, EK0, ReferenceEK0


def test_vec_trick_mul_full():
    for i in range(100):
        d1, d2, d3, d4 = np.random.randint(1, 10, size=4)
        K1, K2 = np.random.rand(d1, d2), np.random.rand(d3, d4)
        v = np.random.rand(d2 * d4)
        assert np.allclose(vec_trick_mul_full(K1, K2, v), np.kron(K1, K2) @ v)


def test_vec_trick_mul_right():
    for i in range(100):
        d = np.random.randint(1, 10)
        q = np.random.randint(1, 10)
        K2 = np.random.rand(q + 1, q + 1)
        v = np.random.rand(d * (q + 1))
        assert np.allclose(vec_trick_mul_right(K2, v), np.kron(np.eye(d), K2) @ v)


@pytest.fixture
def ivp():
    return tornado.ivp.vanderpol(t0=0.0, tmax=0.5, stiffness_constant=1.0)


@pytest.fixture
def scipy_solution(ivp):
    return solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0, rtol=1e-8, atol=1e-8)


def test_reference_ek0_constant_steps(ivp, scipy_solution):
    scipy_final_t = scipy_solution.t[-1]
    scipy_final_y = scipy_solution.y[:, -1]

    constant_steps = tornado.step.ConstantSteps(0.01)
    solver = ReferenceEK0(steprule=constant_steps, solver_order=4)
    for state in solver.solution_generator(ivp=ivp):
        pass

    assert jnp.allclose(scipy_final_t, state.t)
    assert jnp.allclose(scipy_final_y, state.y, rtol=1e-3, atol=1e-3)


def test_ek0_constant_steps(ivp, scipy_solution):
    scipy_final_t = scipy_solution.t[-1]
    scipy_final_y = scipy_solution.y[:, -1]

    constant_steps = tornado.step.ConstantSteps(0.01)
    solver = EK0(steprule=constant_steps, solver_order=4)
    for state in solver.solution_generator(ivp=ivp):
        pass

    assert jnp.allclose(scipy_final_t, state.t)
    assert jnp.allclose(scipy_final_y, state.y, rtol=1e-3, atol=1e-3)


def test_ek0_adaptive_steps(ivp, scipy_solution):
    scipy_final_t = scipy_solution.t[-1]
    scipy_final_y = scipy_solution.y[:, -1]

    srule = tornado.step.AdaptiveSteps(first_dt=0.01, abstol=1e-6, reltol=1e-3)
    solver = EK0(steprule=srule, solver_order=4)
    for state in solver.solution_generator(ivp=ivp):
        pass

    assert jnp.allclose(scipy_final_t, state.t)
    assert jnp.allclose(scipy_final_y, state.y, rtol=1e-3, atol=1e-3)
