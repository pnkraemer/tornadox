"""Tests."""

from jax.config import config

# Run all tests with double precision.
# This is needed for high orders and/or stiff equations.
config.update("jax_enable_x64", True)

# We want to fail early with a readable trace for NaNs.
config.update("jax_debug_nans", True)
