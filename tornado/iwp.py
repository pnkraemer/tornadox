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
    def preconditioned_discretize_1d(self):
        """Preconditioned system matrices for one dimension.

        Returns
        -------
        state_trans_mat: jax array
            1D preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            1D preconditioned cholesky factor of the process noise covariance matrix.
        """
        A_1d = jnp.flip(
            jnp.array(
                scipy.linalg.pascal(self.num_derivatives + 1, kind="lower", exact=False)
            )
        )
        Q_1d = jnp.flip(jnp.array(scipy.linalg.hilbert(self.num_derivatives + 1)))
        return A_1d, jnp.linalg.cholesky(Q_1d)

    @functools.cached_property
    def preconditioned_discretize(self):
        """Preconditioned system matrices.

        Returns
        -------
        state_trans_mat: jax array
            Preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            Preconditioned cholesky factor of the process noise covariance matrix.
        """
        A_1d, L_Q1d = self.preconditioned_discretize_1d
        id_factor = jnp.eye(self.wiener_process_dimension)
        A = jnp.kron(
            id_factor,
            A_1d,
        )
        L_Q = jnp.kron(
            id_factor,
            L_Q1d,
        )
        return A, L_Q

    def nordsieck_preconditioner_1d(self, dt):
        """Create matrix for 1-D Nordsieck preconditioner and its inverse.

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

        nordsieck_precond_1d = jnp.diag(scaling_vector)
        nordsieck_procond_inv_1d = jnp.diag(scaling_vector_inv)

        return nordsieck_precond_1d, nordsieck_procond_inv_1d

    def nordsieck_preconditioner(self, dt):
        """Create matrix for Nordsieck preconditioner and its inverse.

        Returns
        -------
        nordsieck_precond: jax array
            Nordsieck preconditioning matrix
        nordsieck_precond_inv: jax array
            Inverse Nordsieck preconditioning matrix
        """
        (
            nordsieck_precond_1d,
            nordsieck_procond_inv_1d,
        ) = self.nordsieck_preconditioner_1d(dt)
        id_factor = jnp.eye(self.wiener_process_dimension)
        return (
            jnp.kron(id_factor, nordsieck_precond_1d),
            jnp.kron(id_factor, nordsieck_procond_inv_1d),
        )

    def non_preconditioned_discretize(self, dt):
        """Non-preconditioned system matrices. Mainly for testing and debugging.

        Returns
        -------
        state_trans_mat: jax array
            Non-preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            Non-preconditioned cholesky factor of the process noise covariance matrix.
        """

        nordsieck_precond, nordsieck_precond_inv = self.nordsieck_preconditioner(dt)
        (
            precond_state_trans_mat,
            precond_proc_noise_chol,
        ) = self.preconditioned_discretize

        state_trans_mat = (
            nordsieck_precond @ precond_state_trans_mat @ nordsieck_precond_inv
        )

        proc_noise_cov_cholesky = nordsieck_precond @ precond_proc_noise_chol

        return (state_trans_mat, proc_noise_cov_cholesky)

    def projection_matrix(self, derivative_to_project_onto):
        """Creates a projection matrix kron(I_d, e_p)"""
        I_d = jnp.eye(self.wiener_process_dimension)
        return jnp.kron(I_d, self.projection_matrix_1d(derivative_to_project_onto))

    def projection_matrix_1d(self, derivative_to_project_onto):
        """Creates a projection matrix e_p"""
        return jnp.eye(1, self.num_derivatives + 1, derivative_to_project_onto)

    @property
    def state_dimension(self):
        return self.wiener_process_dimension * (self.num_derivatives + 1)

    # Reordering functionality

    def reorder_state_from_derivative_to_coordinate(self, state):
        d, dim = self.wiener_process_dimension, self.state_dimension
        stride = lambda i: jnp.arange(start=i, stop=dim, step=d).reshape((-1, 1))
        indices = jnp.vstack([stride(i) for i in range(d)])[:, 0]
        return state[indices]

    def reorder_state_from_coordinate_to_derivative(self, state):
        n, dim = self.num_derivatives, self.state_dimension
        stride = lambda i: jnp.arange(start=i, stop=dim, step=(n + 1)).reshape((-1, 1))
        indices = jnp.vstack([stride(i) for i in range(n + 1)])[:, 0]
        return state[indices]
