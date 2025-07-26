"""Unit-tests for Roots functionality in chebpy/core/chebfun.py"""

import pytest
import numpy as np

from chebpy import chebfun
from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences

from ..utilities import infnorm

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
eps = DefaultPreferences.eps


@pytest.fixture
def root_functions():
    f1 = Chebfun.initfun_adaptive(lambda x: cos(4 * pi * x), np.linspace(-10, 10, 101))
    f2 = Chebfun.initfun_adaptive(lambda x: sin(2 * pi * x), np.linspace(-1, 1, 5))
    f3 = Chebfun.initfun_adaptive(lambda x: sin(4 * pi * x), np.linspace(-10, 10, 101))
    return f1, f2, f3


def test_empty():
    rts = Chebfun.initempty().roots()
    assert isinstance(rts, np.ndarray)
    assert rts.size == 0


def test_multiple_pieces(root_functions):
    f1, _, _ = root_functions
    rts = f1.roots()
    assert rts.size == 80
    assert infnorm(rts - np.arange(-9.875, 10, 0.25)) <= 10 * eps


# check we don't get repeated roots at breakpoints
def test_breakpoint_roots_1(root_functions):
    _, f2, _ = root_functions
    rts = f2.roots()
    assert rts.size == 5
    assert infnorm(rts - f2.breakpoints) <= eps


# check we don't get repeated roots at breakpoints
def test_breakpoint_roots_2(root_functions):
    _, _, f3 = root_functions
    rts = f3.roots()
    assert rts.size == 81
    assert infnorm(rts - np.arange(-10, 10.25, 0.25)) <= 1e1 * eps


def test_roots_cache():
    # check that a _cache property is created containing the stored roots
    ff = chebfun(sin, np.linspace(-10, 10, 13))
    assert not hasattr(ff, "_cache")
    ff.roots()
    assert hasattr(ff, "_cache")
    assert ff.roots.__name__ in ff._cache.keys()
