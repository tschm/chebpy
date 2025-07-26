"""Unit-tests for chebpy/core/chebfun.py - Construction class"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from ..utilities import infnorm
from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import Interval
from chebpy.core.exceptions import (
    IntervalGap,
    IntervalOverlap,
    InvalidDomain,
    BadFunLengthArgument,
)

# aliases
eps = DefaultPreferences.eps


@pytest.fixture
def construction_functions():
    f = lambda x: np.exp(x)
    fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
    fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))
    fun2 = Bndfun.initfun_adaptive(f, Interval(-0.5, 0.5))
    fun3 = Bndfun.initfun_adaptive(f, Interval(2, 2.5))
    fun4 = Bndfun.initfun_adaptive(f, Interval(-3, -2))
    funs_a = np.array([fun1, fun0, fun2])
    funs_b = np.array([fun1, fun2])
    funs_c = np.array([fun0, fun3])
    funs_d = np.array([fun1, fun4])

    return {
        'f': f,
        'fun0': fun0,
        'fun1': fun1,
        'fun2': fun2,
        'fun3': fun3,
        'fun4': fun4,
        'funs_a': funs_a,
        'funs_b': funs_b,
        'funs_c': funs_c,
        'funs_d': funs_d
    }

def test__init__pass(construction_functions):
    fun0 = construction_functions['fun0']
    fun1 = construction_functions['fun1']
    fun2 = construction_functions['fun2']
    Chebfun([fun0])
    Chebfun([fun1])
    Chebfun([fun2])
    Chebfun([fun0, fun1])


def test__init__fail(construction_functions):
    funs_a = construction_functions['funs_a']
    funs_b = construction_functions['funs_b']
    funs_c = construction_functions['funs_c']
    funs_d = construction_functions['funs_d']
    with pytest.raises(IntervalOverlap):
        Chebfun(funs_a)
    with pytest.raises(IntervalOverlap):
        Chebfun(funs_b)
    with pytest.raises(IntervalGap):
        Chebfun(funs_c)
    with pytest.raises(IntervalGap):
        Chebfun(funs_d)


def test_initempty():
    emptyfun = Chebfun.initempty()
    assert emptyfun.funs.size == 0


def test_initconst(construction_functions):
    fun0 = construction_functions['fun0']
    fun1 = construction_functions['fun1']
    fun2 = construction_functions['fun2']
    assert Chebfun.initconst(1, [-1, 1]).isconst
    assert Chebfun.initconst(-10, np.linspace(-1, 1, 11)).isconst
    assert Chebfun.initconst(3, [-2, 0, 1]).isconst
    assert Chebfun.initconst(3.14, np.linspace(-100, -90, 11)).isconst
    assert not Chebfun([fun0]).isconst
    assert not Chebfun([fun1]).isconst
    assert not Chebfun([fun2]).isconst
    assert not Chebfun([fun0, fun1]).isconst

def test_initidentity():
    _doms = (
        np.linspace(-1, 1, 2),
        np.linspace(-1, 1, 11),
        np.linspace(-10, 17, 351),
        np.linspace(-9.3, -3.2, 22),
        np.linspace(2.5, 144.3, 2112),
    )
    for _dom in _doms:
        ff = Chebfun.initidentity(_dom)
        a, b = ff.support
        xx = np.linspace(a, b, 1001)
        tol = eps * ff.hscale
        assert infnorm(ff(xx) - xx) <= tol
    # test the default case
    ff = Chebfun.initidentity()
    a, b = ff.support
    xx = np.linspace(a, b, 1001)
    tol = eps * ff.hscale
    assert infnorm(ff(xx) - xx) <= tol

def test_initfun_adaptive_continuous_domain(construction_functions):
    f = construction_functions['f']
    ff = Chebfun.initfun_adaptive(f, [-2, -1])
    assert ff.funs.size == 1
    a, b = ff.breakdata.keys()
    (
        fa,
        fb,
    ) = ff.breakdata.values()
    assert a == -2
    assert b == -1
    assert abs(fa - f(-2)) <= eps
    assert abs(fb - f(-1)) <= eps


def test_initfun_adaptive_piecewise_domain(construction_functions):
    f = construction_functions['f']
    ff = Chebfun.initfun_adaptive(f, [-2, 0, 1])
    assert ff.funs.size == 2
    a, b, c = ff.breakdata.keys()
    fa, fb, fc = ff.breakdata.values()
    assert a == -2
    assert b == 0
    assert c == 1
    assert abs(fa - f(-2)) <= eps
    assert abs(fb - f(0)) <= eps
    assert abs(fc - f(1)) <= 2 * eps


def test_initfun_adaptive_raises(construction_functions):
    f = construction_functions['f']
    initfun = Chebfun.initfun_adaptive
    with pytest.raises(InvalidDomain):
        initfun(f, [-2])
    with pytest.raises(InvalidDomain):
        initfun(f, domain=[-2])
    with pytest.raises(InvalidDomain):
        initfun(f, domain=0)


def test_initfun_adaptive_empty_domain(construction_functions):
    f = construction_functions['f']
    result = Chebfun.initfun_adaptive(f, domain=[])
    assert result.isempty

def test_initfun_fixedlen_continuous_domain(construction_functions):
    f = construction_functions['f']
    ff = Chebfun.initfun_fixedlen(f, 20, [-2, -1])
    assert ff.funs.size == 1
    a, b = ff.breakdata.keys()
    (
        fa,
        fb,
    ) = ff.breakdata.values()
    assert a == -2
    assert b == -1
    assert abs(fa - f(-2)) <= eps
    assert abs(fb - f(-1)) <= eps


def test_initfun_fixedlen_piecewise_domain_0(construction_functions):
    f = construction_functions['f']
    ff = Chebfun.initfun_fixedlen(f, 30, [-2, 0, 1])
    assert ff.funs.size == 2
    a, b, c = ff.breakdata.keys()
    fa, fb, fc = ff.breakdata.values()
    assert a == -2
    assert b == 0
    assert c == 1
    assert abs(fa - f(-2)) <= 3 * eps
    assert abs(fb - f(0)) <= 3 * eps
    assert abs(fc - f(1)) <= 3 * eps


def test_initfun_fixedlen_piecewise_domain_1(construction_functions):
    f = construction_functions['f']
    ff = Chebfun.initfun_fixedlen(f, [30, 20], [-2, 0, 1])
    assert ff.funs.size == 2
    a, b, c = ff.breakdata.keys()
    fa, fb, fc = ff.breakdata.values()
    assert a == -2
    assert b == 0
    assert c == 1
    assert abs(fa - f(-2)) <= 3 * eps
    assert abs(fb - f(0)) <= 3 * eps
    assert abs(fc - f(1)) <= 6 * eps

def test_initfun_fixedlen_raises(construction_functions):
    f = construction_functions['f']
    initfun = Chebfun.initfun_fixedlen
    with pytest.raises(InvalidDomain):
        initfun(f, 10, [-2])
    with pytest.raises(InvalidDomain):
        initfun(f, n=10, domain=[-2])
    with pytest.raises(InvalidDomain):
        initfun(f, n=10, domain=0)
    with pytest.raises(BadFunLengthArgument):
        initfun(f, [30, 40], [-1, 1])
    with pytest.raises(TypeError):
        initfun(f, [], [-2, -1, 0])


def test_initfun_fixedlen_empty_domain(construction_functions):
    f = construction_functions['f']
    result = Chebfun.initfun_fixedlen(f, n=10, domain=[])
    assert result.isempty


def test_initfun_fixedlen_succeeds(construction_functions):
    f = construction_functions['f']
    # check providing a vector with None elements calls the
    # Tech adaptive constructor
    dom = [-2, -1, 0]
    g0 = Chebfun.initfun_adaptive(f, dom)
    g1 = Chebfun.initfun_fixedlen(f, [None, None], dom)
    g2 = Chebfun.initfun_fixedlen(f, [None, 40], dom)
    g3 = Chebfun.initfun_fixedlen(f, None, dom)
    for funA, funB in zip(g1, g0):
        assert sum(funA.coeffs - funB.coeffs) == 0
    for funA, funB in zip(g3, g0):
        assert sum(funA.coeffs - funB.coeffs) == 0
    assert sum(g2.funs[0].coeffs - g0.funs[0].coeffs) == 0
