import jax.numpy as jnp


def make_projmat(self, d, q, derivative_to_project_onto):
    """Creates a projection matrix kron(I_d, e_p)"""
    I_d = jnp.eye(d)
    e_p = jnp.eye(1, q + 1, derivative_to_project_onto)
    return jnp.kron(I_d, e_p)
