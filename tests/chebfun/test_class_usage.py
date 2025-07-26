"""Unit-tests for chebpy/core/chebfun.py - ClassUsage class"""

import pytest
import numpy as np

from chebpy import chebfun
from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import Domain, Interval

from ..utilities import infnorm

# aliases
eps = DefaultPreferences.eps


@pytest.fixture
def class_usage_functions():
    f0 = Chebfun.initempty()
    f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])
    return f0, f1, f2

def test__str__(class_usage_functions):
    f0, f1, f2 = class_usage_functions
    _ = str(f0)
    _ = str(f1)
    _ = str(f2)


def test__repr__(class_usage_functions):
    f0, f1, f2 = class_usage_functions
    _ = repr(f0)
    _ = repr(f1)
    _ = repr(f2)


def test_copy(class_usage_functions):
    f0, f1, f2 = class_usage_functions
    f0_copy = f0.copy()
    f1_copy = f1.copy()
    f2_copy = f2.copy()
    assert f0_copy.isempty
    assert f1_copy.funs.size == 1
    for k in range(f1.funs.size):
        fun = f1.funs[k]
        funcopy = f1_copy.funs[k]
        assert fun != funcopy
        assert sum(fun.coeffs - funcopy.coeffs) == 0
    for k in range(f2.funs.size):
        fun = f2.funs[k]
        funcopy = f2_copy.funs[k]
        assert fun != funcopy
        assert sum(fun.coeffs - funcopy.coeffs) == 0


def test__iter__(class_usage_functions):
    f0, f1, f2 = class_usage_functions
    for f in [f0, f1, f2]:
        a1 = [x for x in f]
        a2 = [x for x in f.funs]
        assert np.equal(a1, a2).all()

def test_x_property():
    _doms = (
        np.linspace(-1, 1, 2),
        np.linspace(-1, 1, 11),
        np.linspace(-9.3, -3.2, 22),
    )
    for _dom in _doms:
        f = Chebfun.initfun_fixedlen(np.sin, 1000, _dom)
        x = f.x
        a, b = x.support
        pts = np.linspace(a, b, 1001)
        tol = eps * f.hscale
        assert infnorm(x(pts) - pts) <= tol


def test_restrict_():
    """Tests the ._restrict operator"""
    # test a variety of domains with breaks
    doms = [(-4, 4), (-4, 0, 4), (-2, -1, 0.3, 1, 2.5)]
    for dom in doms:
        ff = Chebfun.initfun_fixedlen(np.cos, 25, domain=dom)
        # define some arbitrary subdomains
        yy = np.linspace(dom[0], dom[-1], 11)
        subdoms = [yy, yy[2:7], yy[::2]]
        for subdom in subdoms:
            xx = np.linspace(subdom[0], subdom[-1], 1001)
            gg = ff._restrict(subdom)
            vscl = ff.vscale
            hscl = ff.hscale
            lscl = max([fun.size for fun in ff])
            tol = vscl * hscl * lscl * eps
            # sample the restricted function and comapre with original
            assert infnorm(ff(xx) - gg(xx)) <= tol
            # check there are at least as many funs as subdom elements
            assert len(gg.funs) >= len(subdom) - 1
            for fun in gg:
                # check each fun has length 25
                assert fun.size == 25


def test_restrict__empty(class_usage_functions):
    f0, _, _ = class_usage_functions
    assert f0._restrict([-1, 1]).isempty

def test_simplify():
    dom = np.linspace(-2, 1.5, 13)
    f = chebfun(np.cos, dom, 70).simplify()
    g = chebfun(np.cos, dom)
    assert f.domain == g.domain
    for n, fun in enumerate(f):
        # we allow one degree of freedom difference
        # TODO: check this
        assert fun.size - g.funs[n].size <= 1


def test_simplify_empty(class_usage_functions):
    f0, _, _ = class_usage_functions
    assert f0.simplify().isempty


def test_restrict():
    dom1 = Domain(np.linspace(-2, 1.5, 13))
    dom2 = Domain(np.linspace(-1.7, 0.93, 17))
    dom3 = dom1.merge(dom2).restrict(dom2)
    f = chebfun(np.cos, dom1).restrict(dom2)
    g = chebfun(np.cos, dom3)
    assert f.domain == g.domain
    for n, fun in enumerate(f):
        # we allow a few degrees of freedom difference either way
        # TODO: once standard chop is fixed, may be able to reduce this to 0
        assert fun.size - g.funs[n].size <= 5


def test_restrict_empty(class_usage_functions):
    f0, _, _ = class_usage_functions
    assert f0.restrict([-1, 1]).isempty

def test_translate(class_usage_functions):
    _, f1, f2 = class_usage_functions
    c = -1.5
    g1 = f1.translate(c)
    g2 = f2.translate(c)
    # check domains match
    assert f1.domain + c == g1.domain
    assert f2.domain + c == g2.domain
    # check fun lengths match
    assert f1.funs.size == g1.funs.size
    assert f2.funs.size == g2.funs.size
    # check coefficients match on each fun
    tol = eps
    for f1k, g1k in zip(f1, g1):
        assert infnorm(f1k.coeffs - g1k.coeffs) <= tol
    for f2k, g2k in zip(f2, g2):
        assert infnorm(f2k.coeffs - g2k.coeffs) <= tol


def test_translate_empty(class_usage_functions):
    f0, _, _ = class_usage_functions
    assert f0.translate(3)
