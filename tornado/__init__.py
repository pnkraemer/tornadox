"""Collect all modules into the tornado.* namespace"""

from jax.config import config

from . import ivp, iwp, odesolver, rv, step, taylor_mode

config.update("jax_enable_x64", True)
