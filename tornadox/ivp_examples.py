"""Initial value problem examples."""

from functools import partial

import jax
import jax.numpy as jnp


def logistic(*, p=None, t0=0.0, tmax=1.0):
    if p is None:
        p = jnp.array([1.0, 1.0])

    @jax.jit
    def f_logistic(u, /):
        return p[0] * u * (1.0 - p[1] * u)

    u0 = jnp.array([0.1])

    return f_logistic, (t0, tmax), u0


def vanderpol(*, t0=0.0, tmax=6.3, u0=None, stiffness_constant=1.0):

    if u0 is None:
        u0 = jnp.array([2.0, 0.0])

    @jax.jit
    def f_vanderpol(u, /):
        return jnp.array([u[1], stiffness_constant * ((1.0 - u[0] ** 2) * u[1] - u[0])])

    return f_vanderpol, (t0, tmax), u0


def vanderpol_second_order(*, t0=0.0, tmax=6.3, u0=None, stiffness_constant=1.0):

    if u0 is None:
        u0 = jnp.array([2.0])
        du0 = jnp.array([0.0])

    @jax.jit
    def f_vanderpol(u, du, /):
        return stiffness_constant * ((1.0 - u**2) * du - u)

    return f_vanderpol, (t0, tmax), (u0, du0)


def threebody(
    *, t0=0.0, tmax=17.0652165601579625588917206249, standardised_moon_mass=0.012277471
):

    # Some shorthand
    mu = standardised_moon_mass
    mp = 1.0 - mu

    @jax.jit
    def f_threebody(Y, /):
        # defining the ODE:
        # assume Y = [u[0],u[1],u[0]',u[1]']
        D1 = jnp.linalg.norm(jnp.array([Y[0] + mu, Y[1]])) ** 3.0
        D2 = jnp.linalg.norm(jnp.array([Y[0] - mp, Y[1]])) ** 3.0
        du0 = Y[0] + 2 * Y[3] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        du1 = Y[1] - 2 * Y[2] - mp * Y[1] / D1 - mu * Y[1] / D2
        return jnp.array([Y[2], Y[3], du0, du1])

    u0 = jnp.array([0.994, 0, 0, -2.00158510637908252240537862224])
    return f_threebody, (t0, tmax), u0


def threebody_second_order(
    *, t0=0.0, tmax=17.0652165601579625588917206249, standardised_moon_mass=0.012277471
):

    # Some shorthand
    mu = standardised_moon_mass
    mp = 1.0 - mu

    @jax.jit
    def f_threebody(Y, dY, /):
        # defining the ODE:
        # assume Y = [u[0],u[1],u[0]',u[1]']
        D1 = jnp.linalg.norm(jnp.array([Y[0] + mu, Y[1]])) ** 3.0
        D2 = jnp.linalg.norm(jnp.array([Y[0] - mp, Y[1]])) ** 3.0
        du0p = Y[0] + 2 * dY[1] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        du1p = Y[1] - 2 * dY[0] - mp * Y[1] / D1 - mu * Y[1] / D2
        return jnp.array([du0p, du1p])

    u0 = jnp.array([0.994, 0])
    du0 = jnp.array([0, -2.00158510637908252240537862224])
    return f_threebody, (t0, tmax), (u0, du0)


def pleiades(*, t0=0.0, tmax=3.0):

    # fmt: off
    u0 = jnp.array(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on

    @jax.jit
    def f_pleiades(u, /):
        """Following the PLEI definition in Hairer I."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        dx = u[14:21]  # x′
        dy = u[21:28]  # y′
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)
        mj = jnp.arange(1, 8)[None, :]
        ddx = jnp.sum(jnp.nan_to_num(mj * (xj - xi) / rij), axis=1)
        ddy = jnp.sum(jnp.nan_to_num(mj * (yj - yi) / rij), axis=1)
        return jnp.concatenate((dx, dy, ddx, ddy))

    return f_pleiades, (t0, tmax), u0


def lorenz96(*, t0=0.0, tmax=30.0, u0=None, num_variables=10, forcing=8.0):

    if u0 is None:
        u0 = _lorenz96_chaotic_u0(forcing=forcing, num_variables=num_variables)

    @jax.jit
    def f_lorenz96(y, /):
        A = jnp.roll(y, shift=-1)
        B = jnp.roll(y, shift=2)
        C = jnp.roll(y, shift=1)
        D = y
        return (A - B) * C - D + forcing

    return f_lorenz96, (t0, tmax), u0


@partial(jax.jit, static_argnames=("num_variables",))
def _lorenz96_chaotic_u0(*, forcing, num_variables):

    u0_equilibrium = jnp.ones(num_variables) * forcing

    # Slightly perturb the equilibrium initval to create chaotic behaviour
    u0 = u0_equilibrium.at[0].set(u0_equilibrium[0] + 0.01)
    return u0


def rigidbody(*, t0=0.0, tmax=20.0, u0=None, params=(-2.0, 1.25, -0.5)):
    r"""Rigid body dynamics without external forces."""

    if u0 is None:
        u0 = jnp.array([1.0, 0.0, 0.9])

    p1, p2, p3 = params

    @jax.jit
    def f_rigidbody(y, /):
        return jnp.array([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])

    return f_rigidbody, (t0, tmax), u0


def fitzhughnagumo(*, t0=0.0, tmax=20.0, u0=None, params=(0.2, 0.2, 3.0, 1.0)):
    r"""FitzHugh-Nagumo model."""

    if u0 is None:
        u0 = jnp.array([1.0, -1.0])

    a, b, c, d = params

    @jax.jit
    def f_fhn(u, /):
        return jnp.array(
            [u[0] - u[0] ** 3.0 / 3.0 - u[1] + a, (u[0] + b - c * u[1]) / d]
        )

    return f_fhn, (t0, tmax), u0


def lotkavolterra(*, t0=0.0, tmax=20.0, u0=None, params=(0.5, 0.05, 0.5, 0.05)):
    if u0 is None:
        u0 = jnp.array([20.0, 20.0])

    a, b, c, d = params

    @jax.jit
    def f_lv(y, /):
        return jnp.array([a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]])

    return f_lv, (t0, tmax), u0


def seir(*, t0=0.0, tmax=200.0, u0=None, params=(0.3, 0.3, 0.1)):

    if u0 is None:
        u0 = jnp.array([998.0, 1.0, 1.0, 0.0])

    params = params + (jnp.sum(u0),)
    alpha, beta, gamma, population_count = params

    @jax.jit
    def f_seir(u, /):
        du0_next = -beta * u[0] * u[2] / population_count
        du1_next = beta * u[0] * u[2] / population_count - alpha * u[1]
        du2_next = alpha * u[1] - gamma * u[2]
        du3_next = gamma * u[2]
        return jnp.array([du0_next, du1_next, du2_next, du3_next])

    return f_seir, (t0, tmax), u0


def sir(*, t0=0.0, tmax=200.0, u0=None, params=(0.3, 0.1)):

    if u0 is None:
        u0 = jnp.array([998.0, 1.0, 1.0, 0.0])

    params = params + (jnp.sum(u0),)
    beta, gamma, population_count = params

    @jax.jit
    def f_sir(u, /):
        du0_next = -beta * u[0] * u[1] / population_count
        du1_next = beta * u[0] * u[1] / population_count - gamma * u[1]
        du2_next = gamma * u[1]
        return jnp.array([du0_next, du1_next, du2_next])

    return f_sir, (t0, tmax), u0


def sird(*, t0=0.0, tmax=200.0, u0=None, params=(0.005, 0.3, 0.1)):

    if u0 is None:
        u0 = jnp.array([998.0, 1.0, 1.0, 0.0])

    params = params + (jnp.sum(u0),)
    eta, beta, gamma, population_count = params

    @jax.jit
    def f_sird(u, /):
        du0_next = -beta * u[0] * u[1] / population_count
        du1_next = beta * u[0] * u[1] / population_count - gamma * u[1] - eta * u[1]
        du2_next = gamma * u[1]
        du3_next = eta * u[1]
        return jnp.array([du0_next, du1_next, du2_next, du3_next])

    return f_sird, (t0, tmax), u0


def lorenz63(*, t0=0.0, tmax=20.0, u0=None, params=(10.0, 28.0, 8.0 / 3.0)):

    if u0 is None:
        u0 = jnp.array([0.0, 1.0, 1.05])

    a, b, c = params

    @jax.jit
    def f_lorenz63(u, /):
        return jnp.array(
            [a * (u[1] - u[0]), u[0] * (b - u[2]) - u[1], u[0] * u[1] - c * u[2]]
        )

    return f_lorenz63, (t0, tmax), u0


def hires(*, t0=0.0, tmax=321.8122, u0=None):
    @jax.jit
    def f_hires(u, /):
        du1 = -1.71 * u[0] + 0.43 * u[1] + 8.32 * u[2] + 0.0007
        du2 = 1.71 * u[0] - 8.75 * u[1]
        du3 = -10.03 * u[2] + 0.43 * u[3] + 0.035 * u[4]
        du4 = 8.32 * u[1] + 1.71 * u[2] - 1.12 * u[3]
        du5 = -1.745 * u[4] + 0.43 * u[5] + 0.43 * u[6]
        du6 = (
            -280.0 * u[5] * u[7] + 0.69 * u[3] + 1.71 * u[4] - 0.43 * u[5] + 0.69 * u[6]
        )
        du7 = 280.0 * u[5] * u[7] - 1.81 * u[6]
        du8 = -280.0 * u[5] * u[7] + 1.81 * u[6]
        return jnp.array([du1, du2, du3, du4, du5, du6, du7, du8])

    if u0 is None:
        u0 = jnp.array([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])
    return f_hires, (t0, tmax), u0


def rober(*, t0=0.0, tmax=1e5, u0=None, params=(0.04, 3e7, 1e4)):

    k1, k2, k3 = params

    @jax.jit
    def f_rober(u, /):
        du0 = -k1 * u[0] + k3 * u[1] * u[2]
        du1 = k1 * u[0] - k2 * u[1] ** 2 - k3 * u[1] * u[2]
        du2 = k2 * u[1] ** 2
        return jnp.array([du0, du1, du2])

    if u0 is None:
        u0 = jnp.array([1.0, 0.0, 0.0])

    return f_rober, (t0, tmax), u0
