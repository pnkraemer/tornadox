"""Tests for the EK1 implementation."""

from scipy.integrate import solve_ivp

import tornado


def test_result():
    """Assert the solver returns a similar solution to SciPy."""

    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=1.0, stiffness_constant=1.0)
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t = scipy_sol.t[-1]
    final_y = scipy_sol.y[:, -1]

    steps = tornado.step.ConstantSteps(0.1)
    ek1 = tornado.ek1.EK1(num_derivatives=4, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        print(state)
    assert False
