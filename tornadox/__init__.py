"""Collect all modules into the tornadox.* namespace"""

from jax.config import config

from . import ek0, ek1, init, ivp, iwp, kalman, linops, odefilter, rv, sqrt, step

config.update("jax_enable_x64", True)
