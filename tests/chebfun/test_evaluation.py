"""Unit-tests for Evaluation functionality in chebpy/core/chebfun.py"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences

from ..utilities import infnorm

# aliases
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps


@pytest.fixture
def chebfun_functions():
    f0 = Chebfun.initempty()
    f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])
    return f0, f1, f2

def test__call__empty_chebfun(chebfun_functions):
    f0, _, _ = chebfun_functions
    assert f0(np.linspace(-1, 1, 100)).size == 0

def test__call__empty_array(chebfun_functions):
    f0, f1, f2 = chebfun_functions
    assert f0(np.array([])).size == 0
    assert f1(np.array([])).size == 0
    assert f2(np.array([])).size == 0

def test__call__point_evaluation(chebfun_functions):
    _, f1, _ = chebfun_functions
    # check we get back a scalar for scalar input
    assert np.isscalar(f1(0.1))

def test__call__singleton(chebfun_functions):
    _, f1, _ = chebfun_functions
    # check that the output is the same for the following inputs:
    # np.array(x), np.array([x]), [x]
    a = f1(np.array(0.1))
    b = f1(np.array([0.1]))
    c = f1([0.1])
    assert a.size == 1
    assert b.size == 1
    assert c.size == 1
    assert np.equal(a, b).all()
    assert np.equal(b, c).all()
    assert np.equal(a, c).all()

def test__call__breakpoints(chebfun_functions):
    _, f1, f2 = chebfun_functions
    # check we get the values at the breakpoints back
    x1 = f1.breakpoints
    x2 = f2.breakpoints
    assert np.equal(f1(x1), [1, 1]).all()
    assert np.equal(f2(x2), [1, 0, 1, 4]).all()

def test__call__outside_interval(chebfun_functions):
    _, f1, f2 = chebfun_functions
    # check we are able to evaluate the Chebfun outside the
    # interval of definition
    x = np.linspace(-3, 3, 100)
    assert np.isfinite(f1(x)).all()
    assert np.isfinite(f2(x)).all()

def test__call__general_evaluation():
    def f(x):
        return sin(4 * x) + exp(cos(14 * x)) - 1.4

    npts = 50000
    dom1 = [-1, 1]
    dom2 = [-1, 0, 1]
    dom3 = [-2, -0.3, 1.2]
    ff1 = Chebfun.initfun_adaptive(f, dom1)
    ff2 = Chebfun.initfun_adaptive(f, dom2)
    ff3 = Chebfun.initfun_adaptive(f, dom3)
    x1 = np.linspace(dom1[0], dom1[-1], npts)
    x2 = np.linspace(dom2[0], dom2[-1], npts)
    x3 = np.linspace(dom3[0], dom3[-1], npts)
    assert infnorm(f(x1) - ff1(x1)) <= 5e1 * eps
    assert infnorm(f(x2) - ff2(x2)) <= 2e1 * eps
    assert infnorm(f(x3) - ff3(x3)) <= 5e1 * eps
