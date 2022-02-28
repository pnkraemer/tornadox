"""Initial value problems and examples."""

from collections import namedtuple

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d


class InitialValueProblem(
    namedtuple(
        "_InitialValueProblem", "f t0 tmax y0 df df_diagonal", defaults=(None, None)
    )
):
    """Initial value problems."""

    @property
    def dimension(self):
        if jnp.isscalar(self.y0):
            return 1
        return self.y0.shape[0]

    @property
    def t_span(self):
        return self.t0, self.tmax


def vanderpol(t0=0.0, tmax=30, y0=None, stiffness_constant=1e1):

    y0 = y0 or jnp.array([2.0, 0.0])

    @jax.jit
    def f_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([Y[1], mu * (1.0 - Y[0] ** 2) * Y[1] - Y[0]])

    df_vanderpol = jax.jit(jax.jacfwd(f_vanderpol, argnums=1))

    @jax.jit
    def df_diagonal_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([0.0, mu * (1.0 - Y[0] ** 2)])

    return InitialValueProblem(
        f=f_vanderpol,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_vanderpol,
        df_diagonal=df_diagonal_vanderpol,
    )


def vanderpol_julia(t0=0.0, tmax=6.3, y0=None, stiffness_constant=1e1):

    y0 = y0 or jnp.array([2.0, 0.0])

    @jax.jit
    def f_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([Y[1], mu * ((1.0 - Y[0] ** 2) * Y[1] - Y[0])])

    df_vanderpol = jax.jit(jax.jacfwd(f_vanderpol, argnums=1))

    @jax.jit
    def df_diagonal_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([0.0, mu * (1.0 - Y[0] ** 2)])

    return InitialValueProblem(
        f=f_vanderpol,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_vanderpol,
        df_diagonal=df_diagonal_vanderpol,
    )


def burgers_1d(t0=0.0, tmax=10.0, y0=None, bbox=None, dx=0.02, diffusion_param=0.02):

    if bbox is None:
        bbox = [0.0, 1.0]

    mesh = jnp.arange(bbox[0], bbox[1] + dx, step=dx)

    if y0 is None:
        y0 = jnp.exp(-500.0 * (mesh - (0.6 * (bbox[1] - bbox[0]))) ** 2)

    @jax.jit
    def f_burgers_1d(_, x):
        nonlinear_convection = x[1:-1] * (x[1:-1] - x[:-2]) / dx
        diffusion = diffusion_param * (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dx**2)
        interior = jnp.array(-nonlinear_convection + diffusion).reshape(-1)
        # Set "cyclic" boundary conditions
        boundary = jnp.array(
            -x[0] * (x[0] - x[-2]) / dx
            + diffusion_param * (x[1] - 2.0 * x[0] + x[-2]) / (dx**2)
        ).reshape(-1)
        return jnp.concatenate((boundary, interior, boundary))

    df_burgers_1d = jax.jit(jax.jacfwd(f_burgers_1d, argnums=1))

    return InitialValueProblem(
        f=f_burgers_1d,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_burgers_1d,
        df_diagonal=lambda t, x: jnp.diag(df_burgers_1d(t, x)),
    )


def wave_1d(t0=0.0, tmax=20.0, y0=None, bbox=None, dx=0.02, diffusion_param=0.01):

    if bbox is None:
        bbox = [0.0, 1.0]

    mesh = jnp.arange(bbox[0], bbox[1] + dx, step=dx)

    if y0 is None:
        y0 = jnp.exp(-70.0 * (mesh - (0.6 * (bbox[1] - bbox[0]))) ** 2)
        y0_dot = jnp.zeros_like(y0)
        Y0 = jnp.concatenate((y0, y0_dot))

    @jax.jit
    def f_wave_1d(_, x):
        _x, _dx = jnp.split(x, 2)
        interior = diffusion_param * (_x[2:] - 2.0 * _x[1:-1] + _x[:-2]) / (dx**2)
        boundaries = (jnp.array(_x[0]).reshape(-1), jnp.array(_x[-1]).reshape(-1))
        _ddx = jnp.concatenate((boundaries[0], interior, boundaries[1]))
        new_dx = jnp.concatenate((jnp.zeros(1), _dx[1:-1], jnp.zeros(1)))
        return jnp.concatenate((new_dx, _ddx))

    df_wave_1d = jax.jit(jax.jacfwd(f_wave_1d, argnums=1))

    return InitialValueProblem(
        f=f_wave_1d,
        t0=t0,
        tmax=tmax,
        y0=Y0,
        df=df_wave_1d,
        df_diagonal=lambda t, x: jnp.diag(df_wave_1d(t, x)),
    )


def wave_2d(t0=0.0, tmax=20.0, y0=None, bbox=None, dx=0.02, diffusion_param=0.01):

    if bbox is None:
        bbox = jnp.array([[0.0, 0.0], [1.0, 1.0]])

    ny, nx = int((bbox[1, 0] - bbox[0, 0]) / dx), int((bbox[1, 1] - bbox[0, 1]) / dx)
    if y0 is None:

        dy = 0.05

        X = jnp.linspace(bbox[0, 0], bbox[1, 0], endpoint=True, num=nx)
        Y = jnp.linspace(bbox[0, 1], bbox[1, 1], endpoint=True, num=ny)
        X, Y = jnp.meshgrid(X, Y, indexing="ij")
        XY = jnp.dstack((X, Y))
        Y0 = jnp.exp(
            -50.0 * jnp.linalg.norm(XY - jnp.array([0.5, 0.5]), axis=-1) ** 2
        ).reshape(-1)
        dy0 = jnp.zeros_like(Y0)
        y0 = jnp.concatenate((Y0, dy0))

    center, top, bottom, left, right = _shifted_indices_on_vectorized_2d_grid(nx, ny)

    @jax.jit
    def f_wave_2d(_, x):
        _x, _dx = jnp.split(x, 2)
        interior = diffusion_param * _laplace_2d(
            _x, center, top, bottom, left, right, dx
        )

        _ddx = jax.ops.index_update(jnp.zeros_like(_x), center, interior)
        new_dx = jax.ops.index_update(jnp.zeros_like(_dx), center, _dx[center])
        return jnp.concatenate((new_dx, _ddx))

    df_wave_2d = jax.jit(jax.jacfwd(f_wave_2d, argnums=1))

    return InitialValueProblem(
        f=f_wave_2d,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_wave_2d,
        df_diagonal=lambda t, x: jnp.diag(df_wave_2d(t, x)),
    )


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

    @jax.jit
    def df_diagonal_threebody(*args, **kwargs):
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    y0 = jnp.array([0.994, 0, 0, -2.00158510637908252240537862224])
    t0 = 0.0
    return InitialValueProblem(
        f=f_threebody,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_threebody,
        df_diagonal=df_diagonal_threebody,
    )


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

        u_new = 1.0 + u**2 * v - 4 * u + c * conv_u
        v_new = 3 * u - u**2 * v + c * conv_v
        return jnp.concatenate([u_new, v_new])

    df_brusselator = jax.jit(jax.jacfwd(f_brusselator, argnums=1))

    @jax.jit
    def df_diagonal_brusselator(_, y, n=N, c=const):
        u, v = y[:n], y[n:]
        u_new = 2 * u * v - 4 - 2 * c
        v_new = -(u**2) - 2 * c
        concat = jnp.concatenate([u_new, v_new])
        return concat

    u0 = jnp.arange(1, N + 1) / N + 1
    v0 = 3.0 * jnp.ones(N)
    y0 = jnp.concatenate([u0, v0])

    return InitialValueProblem(
        f=f_brusselator,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_brusselator,
        df_diagonal=df_diagonal_brusselator,
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

    @jax.jit
    def df_diagonal_lorenz96(_, y, c=forcing):
        return -1.0 * jnp.ones(y.shape[0])

    return InitialValueProblem(
        f=f_lorenz96,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_lorenz96,
        df_diagonal=df_diagonal_lorenz96,
    )


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

    @jax.jit
    def df_diagonal_lorenz96_loop(_, y, c=forcing):
        return -1.0 * jnp.ones(y.shape[0])

    return InitialValueProblem(
        f=f_lorenz96_loop,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df_lorenz96_loop,
        df_diagonal=df_diagonal_lorenz96_loop,
    )


def _lorenz96_chaotic_y0(forcing, num_variables):

    y0_equilibrium = jnp.ones(num_variables) * forcing

    # Slightly perturb the equilibrium initval to create chaotic behaviour
    y0 = y0_equilibrium.at[0].set(y0_equilibrium[0] + 0.01)
    return y0


def pleiades(t0=0.0, tmax=3.0):
    y0 = jnp.array(
        [
            3.0,
            3.0,
            -1.0,
            -3.0,
            2.0,
            -2.0,
            2.0,
            3.0,
            -3.0,
            2.0,
            0,
            0,
            -4.0,
            4.0,
            0,
            0,
            0,
            0,
            0,
            1.75,
            -1.5,
            0,
            0,
            0,
            -1.25,
            1,
            0,
            0,
        ]
    )

    @jax.jit
    def f(t, u):
        """Following the PLEI definition in Hairer I"""
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

    df = jax.jit(jax.jacfwd(f, argnums=1))

    @jax.jit
    def df_diagonal(t, y):
        return jnp.diagonal(df(t, y))

    return InitialValueProblem(
        f=f,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df,
        df_diagonal=df_diagonal,
    )


def fhn_2d(
    t0=0.0,
    tmax=20.0,
    y0=None,
    bbox=None,
    dx=0.02,
    a=2.8e-4,
    b=5e-3,
    k=-0.005,
    tau=0.1,
    prng_key=None,
):
    """Source: https://ipython-books.github.io/124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns/"""

    if bbox is None:
        bbox = [[0.0, 0.0], [1.0, 1.0]]

    ny, nx = int((bbox[1][0] - bbox[0][0]) / dx), int((bbox[1][1] - bbox[0][1]) / dx)
    if ny != nx:
        raise ValueError(
            f"The diagonal Jacobian only works for quadratic spatial grids. Got {bbox}."
        )

    if y0 is None:
        prng_key = prng_key or jax.random.PRNGKey(seed=2)
        y0 = jax.random.uniform(prng_key, shape=(2 * ny * nx,))

    @jax.jit
    def fhn_2d(_, x):
        u, v = jnp.split(x, 2)
        du = _laplace_2d(u.reshape((nx, ny)), dx=dx).reshape((-1,))
        dv = _laplace_2d(v.reshape((nx, ny)), dx=dx).reshape((-1,))
        u_new = a * du + u - u**3 - v + k
        v_new = (b * dv + u - v) / tau
        return jnp.concatenate((u_new, v_new))

    dfhn_2d = jax.jit(jax.jacfwd(fhn_2d, argnums=1))

    # The diagonal entries of the Jacobian are iether
    # 2. (for corner nodes), 3. (for boundary nodes), or 4. (for interior nodes)
    df_diag_interior = 4.0 * jnp.ones((nx - 2, ny - 2))
    df_diag_with_bdry = jnp.pad(
        df_diag_interior, pad_width=1, mode="constant", constant_values=3.0
    )
    df_diag = (
        df_diag_with_bdry.at[0, 0]
        .set(2.0)
        .at[0, -1]
        .set(2.0)
        .at[-1, 0]
        .set(2.0)
        .at[-1, -1]
        .set(2.0)
    )
    dlaplace = -1.0 * df_diag.reshape((-1,)) / (dx**2)

    @jax.jit
    def df_diag(_, x):

        u, v = jnp.split(x, 2)

        d_u = a * dlaplace + 1.0 - 3.0 * u**2
        d_v = (b * dlaplace - 1.0) / tau
        return jnp.concatenate((d_u, d_v))

    return InitialValueProblem(
        f=fhn_2d,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=dfhn_2d,
        df_diagonal=df_diag,
    )


@jax.jit
def _laplace_2d(grid, dx):
    """2D Laplace operator on a vectorized 2d grid."""

    # Set the boundary values to the nearest interior node
    # This enforces Neumann conditions.
    padded_grid = jnp.pad(grid, pad_width=1, mode="edge")

    # Laplacian via convolve2d()
    kernel = (
        1
        / (dx**2)
        * jnp.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    grid = convolve2d(padded_grid, kernel, mode="same")
    return grid[1:-1, 1:-1]
