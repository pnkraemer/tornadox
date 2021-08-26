"""ENKF solvers."""

import dataclasses
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from tornado import init, ivp, iwp, linops, odesolver, rv


@dataclasses.dataclass(frozen=False)
class StateEnsemble:
    ivp: ivp.InitialValueProblem
    t: float
    samples: jnp.ndarray  # shape = [d * (nu + 1), N]

    @cached_property
    def ensemble_size(self):
        return self.samples.shape[1]

    @cached_property
    def dim(self):
        return self.samples.shape[0]

    def __getitem__(self, key):
        return self.samples.__getitem__(key)

    def __matmul__(self, other):
        if isinstance(other, jnp.ndarray):
            return StateEnsemble(ivp=self.ivp, t=self.t, samples=self.samples @ other)
        elif isinstance(other, StateEnsemble):
            return StateEnsemble(
                ivp=self.ivp, t=self.t, samples=self.samples @ other.samples
            )
        else:
            raise TypeError(f"Got type {type(other)}.")

    def __add__(self, other):
        if isinstance(other, jnp.ndarray):
            return StateEnsemble(ivp=self.ivp, t=self.t, samples=self.samples + other)
        elif isinstance(other, StateEnsemble):
            return StateEnsemble(
                ivp=self.ivp, t=self.t, samples=self.samples + other.samples
            )
        else:
            raise TypeError(f"Got type {type(other)}.")

    def __sub__(self, other):
        if isinstance(other, jnp.ndarray):
            return StateEnsemble(ivp=self.ivp, t=self.t, samples=self.samples - other)
        elif isinstance(other, StateEnsemble):
            return StateEnsemble(
                ivp=self.ivp, t=self.t, samples=self.samples - other.samples
            )
        else:
            raise TypeError(f"Got type {type(other)}.")

    def __rmatmul__(self, other):
        if isinstance(other, jnp.ndarray):
            return StateEnsemble(ivp=self.ivp, t=self.t, samples=other @ self.samples)
        elif isinstance(other, StateEnsemble):
            return StateEnsemble(
                ivp=self.ivp, t=self.t, samples=other.samples @ self.samples
            )
        else:
            raise TypeError(f"Got type {type(other)}.")

    def __radd__(self, other):
        if isinstance(other, jnp.ndarray):
            return StateEnsemble(ivp=self.ivp, t=self.t, samples=other + self.samples)
        elif isinstance(other, StateEnsemble):
            return StateEnsemble(
                ivp=self.ivp, t=self.t, samples=other.samples + self.samples
            )
        else:
            raise TypeError(f"Got type {type(other)}.")

    def __rsub__(self, other):
        if isinstance(other, jnp.ndarray):
            return StateEnsemble(ivp=self.ivp, t=self.t, samples=other - self.samples)
        elif isinstance(other, StateEnsemble):
            return StateEnsemble(
                ivp=self.ivp, t=self.t, samples=other.samples - self.samples
            )
        else:
            raise TypeError(f"Got type {type(other)}.")

    @property
    def mean(self):
        return (
            jnp.mean(self.samples, 1, keepdims=True)
            if self.samples is not None
            else None
        )

    @property
    def sample_cov(self):
        centered = self.samples - self.mean
        return centered @ centered.T / (self.ensemble_size - 1)

    @property
    def cov_sqrtm(self):
        return jnp.linalg.cholesky(self.sample_cov)


class EnK0(odesolver.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None
        self.ensemble_size = kwargs["ensemble_size"]

        self.rng = jax.random.PRNGKey(1)

    def initialize(self, ivp):
        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )

        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        mean = extended_dy0.reshape((-1, 1), order="F")

        init_states = jnp.repeat(
            mean, self.ensemble_size, axis=1
        )  #  shape = [d * (nu + 1), N]
        assert init_states.shape == (
            ivp.dimension * (self.num_derivatives + 1),
            self.ensemble_size,
        )
        assert jnp.allclose(init_states[:, 0], mean.squeeze())
        assert jnp.allclose(init_states[:, -1], mean.squeeze())
        return StateEnsemble(ivp=ivp, t=ivp.t0, samples=init_states)

    def attempt_step(self, ensemble, dt, verbose=False):
        # [Setup]
        A, Ql = self.iwp.non_preconditioned_discretize(dt)

        # [Predict]
        w = jax.random.multivariate_normal(
            self.rng,
            mean=jnp.zeros((ensemble.dim,)),
            cov=Ql @ Ql.T,
            shape=(ensemble.ensemble_size,),
        ).T

        _, self.rng = jax.random.split(self.rng)

        pred_samples = A @ ensemble + w

        z = self.E1 @ pred_samples - ensemble.ivp.f(
            ensemble.t + dt, self.E0 @ pred_samples
        )
        H = self.E1

        # Estimate Kalman gain
        # via Eq. (11) in https://www.math.umd.edu/~slud/RITF17/enkf-tutorial.pdf
        sample_cov = pred_samples.sample_cov
        CHT = sample_cov @ H.T
        to_invert = H @ CHT
        gain_times_z = CHT @ jnp.linalg.solve(to_invert, z.samples)

        # Update
        updated_samples = pred_samples - gain_times_z
        updated_samples.t = updated_samples.t + dt

        return updated_samples
