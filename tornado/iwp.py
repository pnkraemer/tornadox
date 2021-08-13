import dataclasses
import functools

import jax.numpy as jnp
import scipy.linalg


@dataclasses.dataclass
class IntegratedWienerTransition:

    wiener_process_dimension: int
    num_derivatives: int

    @functools.cached_property
    def preconditioned_transition_matrix(self):
        state_transition_1d = jnp.flip(
            jnp.array(
                scipy.linalg.pascal(self.num_derivatives + 1, kind="lower", exact=False)
            )
        )
        return jnp.kron(jnp.eye(self.wiener_process_dimension), state_transition_1d)

    @functools.cached_property
    def cholesky_process_noise(self):
        process_noise_1d = jnp.flip(
            jnp.array(scipy.linalg.hilbert(self.num_derivatives + 1))
        )
        process_noise_cholesky_1d = jnp.linalg.cholesky(process_noise_1d)

        return jnp.kron(
            jnp.eye(self.wiener_process_dimension), process_noise_cholesky_1d
        )
