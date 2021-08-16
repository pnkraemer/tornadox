import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp


import tornado
from tornado.ek0 import vec_trick_mul_full, vec_trick_mul_right, EK0


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


def test_ek0_constant_steps():
    """Assert the solver returns a similar solution to SciPy"""

    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=0.5, stiffness_constant=1.0)
    scipy_sol = solve_ivp(
        ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0, rtol=1e-8, atol=1e-8
    )
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]

    # dt = jnp.mean(jnp.diff(scipy_sol.t))
    dt = 0.01

    constant_steps = tornado.step.ConstantSteps(dt)
    solver = EK0(steprule=constant_steps, solver_order=4)
    sol_gen = solver.solution_generator(ivp=ivp)
    for state in sol_gen:
        pass

    final_t_ek0 = state.t
    final_y_ek0 = state.y
    assert jnp.allclose(final_t_scipy, final_t_ek0)
    assert jnp.allclose(final_y_scipy, final_y_ek0, rtol=1e-3, atol=1e-3)
