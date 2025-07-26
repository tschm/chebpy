"""Unit-tests for Calculus functionality in chebpy/core/chebfun.py"""

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
def calculus_functions():
    def f(x):
        return sin(4 * x - 1.4)

    def g(x):
        return exp(x)

    df = lambda x: 4 * cos(4 * x - 1.4)
    If = lambda x: -0.25 * cos(4 * x - 1.4)
    f1 = Chebfun.initfun_adaptive(f, [-1, 1])
    f2 = Chebfun.initfun_adaptive(f, [-3, 0, 1])
    f3 = Chebfun.initfun_adaptive(f, [-2, -0.3, 1.2])
    f4 = Chebfun.initfun_adaptive(f, np.linspace(-1, 1, 11))
    g1 = Chebfun.initfun_adaptive(g, [-1, 1])
    g2 = Chebfun.initfun_adaptive(g, [-3, 0, 1])
    g3 = Chebfun.initfun_adaptive(g, [-2, -0.3, 1.2])
    g4 = Chebfun.initfun_adaptive(g, np.linspace(-1, 1, 11))

    return {
        'df': df,
        'If': If,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f4': f4,
        'g1': g1,
        'g2': g2,
        'g3': g3,
        'g4': g4
    }

def test_sum(calculus_functions):
    f1 = calculus_functions['f1']
    f2 = calculus_functions['f2']
    f3 = calculus_functions['f3']
    f4 = calculus_functions['f4']
    assert abs(f1.sum() - 0.372895407327895) <= 2 * eps
    assert abs(f2.sum() - 0.382270459230604) <= 2 * eps
    assert abs(f3.sum() - (-0.008223712363936)) <= 2 * eps
    assert abs(f4.sum() - 0.372895407327895) <= 2 * eps

def test_diff(calculus_functions):
    f1 = calculus_functions['f1']
    f2 = calculus_functions['f2']
    f3 = calculus_functions['f3']
    f4 = calculus_functions['f4']
    df = calculus_functions['df']
    xx = np.linspace(-5, 5, 10000)
    for f in [f1, f2, f3, f4]:
        a, b = f.support
        x = xx[(xx > a) & (xx < b)]
        assert infnorm(f.diff()(x) - df(x)) <= 2e3 * eps

def test_cumsum(calculus_functions):
    f1 = calculus_functions['f1']
    f2 = calculus_functions['f2']
    f3 = calculus_functions['f3']
    f4 = calculus_functions['f4']
    If = calculus_functions['If']
    xx = np.linspace(-5, 5, 10000)
    for f in [f1, f2, f3, f4]:
        a, b = f.support
        x = xx[(xx > a) & (xx < b)]
        fa = If(a)
        assert infnorm(f.cumsum()(x) - If(x) + fa) <= 3 * eps

def test_sum_empty():
    f = Chebfun.initempty()
    assert f.sum() == 0.0

def test_cumsum_empty():
    If = Chebfun.initempty().cumsum()
    assert isinstance(If, Chebfun)
    assert If.isempty

def test_diff_empty():
    df = Chebfun.initempty().diff()
    assert isinstance(df, Chebfun)
    assert df.isempty

def test_dot(calculus_functions):
    f1 = calculus_functions['f1']
    f2 = calculus_functions['f2']
    f3 = calculus_functions['f3']
    f4 = calculus_functions['f4']
    g1 = calculus_functions['g1']
    g2 = calculus_functions['g2']
    g3 = calculus_functions['g3']
    g4 = calculus_functions['g4']
    assert f1.dot(g1) - 0.66870683499839867 <= 3 * eps
    assert f2.dot(g2) - 0.64053327987194342 <= 3 * eps
    assert f3.dot(g3) - 0.67372257930409951 <= 3 * eps
    assert f4.dot(g4) - 0.66870683499839922 <= 3 * eps
    # different partitions of same interval
    assert f1.dot(g4) - 0.66870683499839867 <= 3 * eps
    assert g1.dot(f4) - 0.66870683499839867 <= 3 * eps

def test_dot_commute(calculus_functions):
    f1 = calculus_functions['f1']
    f2 = calculus_functions['f2']
    f3 = calculus_functions['f3']
    f4 = calculus_functions['f4']
    g1 = calculus_functions['g1']
    g2 = calculus_functions['g2']
    g3 = calculus_functions['g3']
    g4 = calculus_functions['g4']
    assert g1.dot(f1) - 0.66870683499839867 <= 3 * eps
    assert g2.dot(f2) - 0.64053327987194342 <= 3 * eps
    assert g3.dot(f3) - 0.67372257930409951 <= 3 * eps
    assert g4.dot(f4) - 0.66870683499839922 <= 3 * eps

def test_dot_empty(calculus_functions):
    f1 = calculus_functions['f1']
    emptyfun = Chebfun.initempty()
    assert f1.dot(emptyfun) == 0
    assert emptyfun.dot(f1) == 0
