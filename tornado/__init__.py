"""Collect all modules into the tornado.* namespace"""

from jax.config import config

from . import ek1, ivp, iwp, odesolver, rv, sqrt, step, taylor_mode

config.update("jax_enable_x64", True)
