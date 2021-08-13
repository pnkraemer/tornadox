"""Tests for stepsize selection."""


import tornado

def test_propose_firststep():

    ivp = tornado.ivp.vanderpol()

    dt = tornado.step.propose_firststep(ivp)
    assert dt > 0
    
