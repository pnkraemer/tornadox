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
    df: Callable[[float, jnp.ndarray], jnp.ndarray]

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
