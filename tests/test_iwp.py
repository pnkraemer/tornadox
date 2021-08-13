import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def iwp():
    return tornado.IntegratedWienerTransition(
        num_derivatives=1, wiener_process_dimension=1
    )
