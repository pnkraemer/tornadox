import dataclasses
import functools

import jax.numpy as jnp
import scipy.linalg
import scipy.special


@dataclasses.dataclass
class IntegratedWienerTransition:

    wiener_process_dimension: int
    num_derivatives: int

    @functools.cached_property
    def preconditioned_transition_matrix(self):
        """Preconditioned state transition matrix."""
        state_transition_1d = jnp.flip(
            jnp.array(
                scipy.linalg.pascal(self.num_derivatives + 1, kind="lower", exact=False)
            )
        )
        return jnp.kron(jnp.eye(self.wiener_process_dimension), state_transition_1d)

    @functools.cached_property
    def preconditioned_cholesky_process_noise(self):
        """Preconditioned cholesky factor of the process noise covariance matrix."""
        process_noise_1d = jnp.flip(
            jnp.array(scipy.linalg.hilbert(self.num_derivatives + 1))
        )
        process_noise_cholesky_1d = jnp.linalg.cholesky(process_noise_1d)

        return jnp.kron(
            jnp.eye(self.wiener_process_dimension), process_noise_cholesky_1d
        )

    def nordsieck_preconditioner(self, dt):
        """Create Nordsieck precondition matrix and its inverse.

        Returns
        -------
        nordsieck_precond: jax array
            Nordsieck preconditioning matrix
        nordsieck_precond_inv: jax array
            Inverse Nordsieck preconditioning matrix
        """
        powers = jnp.arange(self.num_derivatives, -1, -1)
        scales = jnp.array(scipy.special.factorial(powers))
        powers = powers + 0.5

        scaling_vector = (jnp.abs(dt) ** powers) / scales
        scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales

        nordsieck_precond = jnp.kron(
            jnp.eye(self.wiener_process_dimension), jnp.diag(scaling_vector)
        )
        nordsieck_procond_inv = jnp.kron(
            jnp.eye(self.wiener_process_dimension), jnp.diag(scaling_vector_inv)
        )
        return nordsieck_precond, nordsieck_procond_inv

    def discretize(self, dt):
        """Non-preconditioned system matrices. Mainly for testing and debugging.

        Returns
        -------
        state_trans_mat: jax array
            Non-preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            Non-preconditioned cholesky factor of the process noise covariance matrix.
        """

        nordsieck_precond, nordsieck_precond_inv = self.nordsieck_preconditioner(dt)

        state_trans_mat = (
            nordsieck_precond
            @ self.preconditioned_transition_matrix
            @ nordsieck_precond_inv
        )

        proc_noise_cov_cholesky = (
            nordsieck_precond @ self.preconditioned_cholesky_process_noise
        )

        return (state_trans_mat, proc_noise_cov_cholesky)
