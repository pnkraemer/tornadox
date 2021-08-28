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

    y0 = y0 or jnp.array([2.0, 0.0])

    @jax.jit
    def f_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([Y[1], mu * (1.0 - Y[0] ** 2) * Y[1] - Y[0]])

    df_vanderpol = jax.jit(jax.jacfwd(f_vanderpol, argnums=1))

    return InitialValueProblem(f=f_vanderpol, t0=t0, tmax=tmax, y0=y0, df=df_vanderpol)


def threebody(tmax=17.0652165601579625588917206249):
    @jax.jit
    def f_threebody(_, Y):
        # defining the ODE:
        # assume Y = [y1,y2,y1',y2']
        mu = 0.012277471  # a constant (standardised moon mass)
        mp = 1 - mu
        D1 = ((Y[0] + mu) ** 2 + Y[1] ** 2) ** (3 / 2)
        D2 = ((Y[0] - mp) ** 2 + Y[1] ** 2) ** (3 / 2)
        y1p = Y[0] + 2 * Y[3] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        y2p = Y[1] - 2 * Y[2] - mp * Y[1] / D1 - mu * Y[1] / D2
        return jnp.array([Y[2], Y[3], y1p, y2p])

    df_threebody = jax.jit(jax.jacfwd(f_threebody, argnums=1))

    y0 = jnp.array([0.994, 0, 0, -2.00158510637908252240537862224])
    t0 = 0.0
    return InitialValueProblem(f=f_threebody, t0=t0, tmax=tmax, y0=y0, df=df_threebody)


def brusselator(N=20, t0=0.0, tmax=10.0):
    """Brusselator as in https://uk.mathworks.com/help/matlab/math/solve-stiff-odes.html.
    N=20 is the same default as in Matlab.
    """
    alpha = 1.0 / 50.0
    const = alpha * (N + 1) ** 2
    weights = jnp.array([1.0, -2.0, 1.0])

    @jax.jit
    def f_brusselator(_, y, n=N, w=weights, c=const):
        """Evaluate the Brusselator RHS via jnp.convolve, which is equivalent to multiplication with a banded matrix."""
        u, v = y[:n], y[n:]

        # Compute (1, -2, 1)-weighted average with boundary behaviour as in the Matlab link above.
        u_pad = jnp.array([1.0])
        v_pad = jnp.array([3.0])
        u_ = jnp.concatenate([u_pad, u, u_pad])
        v_ = jnp.concatenate([v_pad, v, v_pad])
        conv_u = jnp.convolve(u_, w, mode="valid")
        conv_v = jnp.convolve(v_, w, mode="valid")

        u_new = 1.0 + u ** 2 * v - 4 * u + c * conv_u
        v_new = 3 * u - u ** 2 * v + c * conv_v
        return jnp.concatenate([u_new, v_new])

    df_brusselator = jax.jit(jax.jacfwd(f_brusselator, argnums=1))

    u0 = jnp.arange(1, N + 1) / N + 1
    v0 = 3.0 * jnp.ones(N)
    y0 = jnp.concatenate([u0, v0])

    return InitialValueProblem(
        f=f_brusselator, t0=t0, tmax=tmax, y0=y0, df=df_brusselator
    )


def lorenz96(t0=0.0, tmax=30.0, y0=None, num_variables=10, forcing=8.0):
    """Lorenz 96 system in JAX implementation."""

    y0 = y0 or _lorenz96_chaotic_y0(forcing, num_variables)

    @jax.jit
    def f_lorenz96(_, y, c=forcing):
        A = jnp.roll(y, shift=-1)
        B = jnp.roll(y, shift=2)
        C = jnp.roll(y, shift=1)
        D = y
        return (A - B) * C - D + c

    df_lorenz96 = jax.jit(jax.jacfwd(f_lorenz96, argnums=1))

    return InitialValueProblem(f=f_lorenz96, t0=t0, tmax=tmax, y0=y0, df=df_lorenz96)


# The loop version is useful, because Taylor mode cannot handle jnp.roll...
def lorenz96_loop(t0=0.0, tmax=30.0, y0=None, num_variables=10, forcing=8.0):
    """Lorenz 96 system in JAX implementation, where the RHS is implemented in a loop."""

    y0 = y0 or _lorenz96_chaotic_y0(forcing, num_variables)

    @jax.jit
    def f_lorenz96_loop(_, x, c=forcing):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        N = x.shape[0]
        d = jnp.zeros(N)

        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            A = x[(i + 1) % N] - x[i - 2]
            B = x[i - 1]
            C = x[i]
            val = A * B - C + c
            d = d.at[i].set(val)
        return d

    df_lorenz96_loop = jax.jit(jax.jacfwd(f_lorenz96_loop, argnums=1))

    return InitialValueProblem(
        f=f_lorenz96_loop, t0=t0, tmax=tmax, y0=y0, df=df_lorenz96_loop
    )


def _lorenz96_chaotic_y0(forcing, num_variables):

    y0_equilibrium = jnp.ones(num_variables) * forcing

    # Slightly perturb the equilibrium initval to create chaotic behaviour
    y0 = y0_equilibrium.at[0].set(y0_equilibrium[0] + 0.01)
    return y0
