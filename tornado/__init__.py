"""Collect all modules into the tornado.* namespace"""

from jax.config import config

from . import ek0, ek1, init, ivp, ivpsolve, iwp, linops, odesolver, rv, sqrt, step

config.update("jax_enable_x64", True)
