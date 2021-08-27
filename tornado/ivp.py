"""Initial value problems and examples."""


import dataclasses
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp


@dataclasses.dataclass
class InitialValueProblem:
    """Initial value problems."""

    f: Callable[[float, jnp.ndarray], jnp.ndarray]
    t0: float
    tmax: float
    y0: Union[float, jnp.ndarray]
    df: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None

    @property
    def dimension(self):
        if jnp.isscalar(self.y0):
            return 1
        return self.y0.shape[0]


def vanderpol(t0=0.0, tmax=30, y0=None, stiffness_constant=1e1):

    if y0 is None:
        y0 = jnp.array([2.0, 0.0])

    def vanderpol_rhs(Y):
        return jnp.array([Y[1], stiffness_constant * (1.0 - Y[0] ** 2) * Y[1] - Y[0]])

    df = jax.jacfwd(vanderpol_rhs)

    def rhs(t, y):
        return vanderpol_rhs(Y=y)

    def jac(t, y):
        return df(y)

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def threebody(tmax=17.0652165601579625588917206249):
    def threebody_rhs(Y):
        # defining the ODE:
        # assume Y = [y1,y2,y1',y2']
        mu = 0.012277471  # a constant (standardised moon mass)
        mp = 1 - mu
        D1 = ((Y[0] + mu) ** 2 + Y[1] ** 2) ** (3 / 2)
        D2 = ((Y[0] - mp) ** 2 + Y[1] ** 2) ** (3 / 2)
        y1p = Y[0] + 2 * Y[3] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        y2p = Y[1] - 2 * Y[2] - mp * Y[1] / D1 - mu * Y[1] / D2
        return jnp.array([Y[2], Y[3], y1p, y2p])

    df = jax.jacfwd(threebody_rhs)

    def rhs(t, y):
        return threebody_rhs(Y=y)

    def jac(t, y):
        return df(y)

    y0 = jnp.array([0.994, 0, 0, -2.00158510637908252240537862224])
    t0 = 0.0
    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)


def brusselator(N=20, t0=0.0, tmax=10.0):
    """Brusselator as in https://uk.mathworks.com/help/matlab/math/solve-stiff-odes.html.
    N=20 is the same default as in Matlab.
    """
    alpha = 1.0 / 50.0
    const = alpha * (N + 1) ** 2
    weights = jnp.array([1.0, -2.0, 1.0])

    def brusselator_rhs(y):
        """Evaluate the Brusselator RHS via jnp.convolve, which is equivalent to multiplication with a banded matrix."""
        u, v = y[:N], y[N:]

        # Compute (1, -2, 1)-weighted average with boundary behaviour as in the Matlab link above.
        u_pad = jnp.array([1.0])
        v_pad = jnp.array([3.0])
        u_ = jnp.concatenate([u_pad, u, u_pad])
        v_ = jnp.concatenate([v_pad, v, v_pad])
        conv_u = jnp.convolve(u_, weights, mode="valid")
        conv_v = jnp.convolve(v_, weights, mode="valid")

        u_new = 1.0 + u ** 2 * v - 4 * u + const * conv_u
        v_new = 3 * u - u ** 2 * v + const * conv_v
        return jnp.concatenate([u_new, v_new])

    df = jax.jacfwd(brusselator_rhs)

    def rhs(_, y):
        dy = brusselator_rhs(y)
        return dy

    def jac(_, y):
        df_ = df(y)
        return df_

    u0 = jnp.arange(1, N + 1) / N + 1
    v0 = 3.0 * jnp.ones(N)
    y0 = jnp.concatenate([u0, v0])

    return InitialValueProblem(f=rhs, t0=t0, tmax=tmax, y0=y0, df=jac)
