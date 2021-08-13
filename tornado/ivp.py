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
