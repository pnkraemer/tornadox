import dataclasses
import jax.numpy as jnp

import tornado
from tornado.odesolver import ODESolver


@dataclasses.dataclass
class EK0State:
    ivp: tornado.ivp.InitialValueProblem
    y: jnp.array
    t: float
    error_estimate: jnp.array
    reference_state: jnp.array
    Y: tornado.rv.MultivariateNormal


class EK0(ODESolver):
    def initialize(self, ivp):
        self.d = ivp.dimension
        self.q = self.solver_order
        self.iwp = tornado.iwp.IntegratedWienerTransition(
            wiener_process_dimension=self.d, num_derivatives=self.q
        )
        self.A, self.Ql = self.iwp.preconditioned_discretize_1d
        # A_full, Q_full = iwp.preconditioned_discretize

        Y0_full = tornado.taylor_mode.TaylorModeInitialization()(ivp, self.iwp)
        Y0_kron = tornado.rv.MultivariateNormal(
            Y0_full.mean, jnp.zeros((self.q + 1, self.q + 1))
        )

        self.e0 = self.iwp.make_projection_matrix_1d(0)
        self.e1 = self.iwp.make_projection_matrix_1d(1)

        return EK0State(
            ivp=ivp,
            y=ivp.y0,
            t=ivp.t0,
            error_estimate=None,
            reference_state=ivp.y0,
            Y=Y0_kron,
        )

    def attempt_step(self, state, dt):
        print("Setup:")
        Y = state.Y
        m, Cl = Y.mean, Y.cov_cholesky
        C = Cl @ Cl.T
        A, Ql = self.A, self.Ql
        ic(Y)

        t_new = state.t + dt

        print("Preconditioners:")
        P, PI = self.iwp.nordsieck_preconditioner_1d(dt)
        m, Cl = vec_trick_mul(PI, m), PI @ Cl
        C = Cl @ Cl.T
        # ic(P, PI)

        print("Predict:")
        # Predict mean
        mp = vec_trick_mul(A, m)
        # Predict cov
        Cp = A @ C @ A.T + Ql @ Ql.T
        Clp = tornado.sqrt.propagate_cholesky_factor(A @ Cl, Ql)
        assert (Cp == Clp @ Clp.T).all()
        # ic(mp, Clp)

        print("Measure:")
        z = state.ivp.f(t_new, mp)
        e1 = self.e1
        _S = e1 @ P @ Cp @ P @ e1.T
        Sl = e1 @ P @ Clp
        S = (Sl @ Sl.T)[0]
        ic(z, S, _S)
        assert jnp.allclose(_S, S)

        """
        Notes to self:

        - Something is not working with the preconditioning yet
        - vec_trick_mul is not happy with K@v yet
        - I should look up the squareroot implementation once more and do it as suggested there
        - Completely ignored so far: diffusion, error estimation, proper square-root stuff, efficient code
        """

        print("Update:")
        K = Cp @ e1.T / S
        _m_new = m - jnp.kron(jnp.eye(self.d), K) @ z
        ic(K, mp, z, vec_trick_mul(P, _m_new))
        m_new = m - vec_trick_mul(K, z)

        raise Exception

        y = state.y + dt * state.ivp.f(state.t, state.y)
        t = state.t + dt
        return EK0State(ivp=state.ivp, y=y, t=t, error_estimate=None, reference_state=y)


def vec_trick_mul(M, v):
    """Use the vec trick to compute M@v more efficiently"""
    d, d = M.shape
    D = len(v)

    V = v.reshape(d, D // d, order="F")
    return (M @ V).reshape(D, order="F")


if __name__ == "__main__":
    from icecream import ic

    print("EK0 development")

    print("Problem setup")
    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=1.5)

    print("Solver setup")
    constant_steps = tornado.step.ConstantSteps(dt=0.001)
    solver_order = 2
    solver = EK0(steprule=constant_steps, solver_order=solver_order)
    ic(solver)

    print("Solve")
    gen_sol = solver.solution_generator(ivp)
    for i, state in enumerate(gen_sol):
        ic(i)
        ic(state.t)
        ic(state.y)
