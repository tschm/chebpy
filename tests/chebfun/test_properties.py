"""Unit-tests for chebpy/core/chebfun.py - Properties class"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import Domain


@pytest.fixture
def property_functions():
    f0 = Chebfun.initempty()
    f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])
    return f0, f1, f2


def test_breakpoints(property_functions):
    f0, f1, f2 = property_functions
    assert f0.breakpoints.size == 0
    assert np.equal(f1.breakpoints, [-1, 1]).all()
    assert np.equal(f2.breakpoints, [-1, 0, 1, 2]).all()


def test_domain(property_functions):
    f0, f1, f2 = property_functions
    d1 = Domain([-1, 1])
    d2 = Domain([-1, 0, 1, 2])
    assert isinstance(f0.domain, np.ndarray)
    assert isinstance(f1.domain, Domain)
    assert isinstance(f2.domain, Domain)
    assert f0.domain.size == 0
    assert f1.domain == d1
    assert f2.domain == d2


def test_hscale(property_functions):
    f0, f1, f2 = property_functions
    assert isinstance(f0.hscale, float)
    assert isinstance(f1.hscale, float)
    assert isinstance(f2.hscale, float)
    assert f0.hscale == 0
    assert f1.hscale == 1
    assert f2.hscale == 2


def test_isempty(property_functions):
    f0, f1, f2 = property_functions
    assert f0.isempty
    assert not f1.isempty
    assert not f2.isempty


def test_isconst(property_functions):
    f0, f1, f2 = property_functions
    assert not f0.isconst
    assert not f1.isconst
    assert not f2.isconst
    c1 = Chebfun.initfun_fixedlen(lambda x: 0 * x + 3, 1, [-2, -1, 0, 1, 2, 3])
    c2 = Chebfun.initfun_fixedlen(lambda x: 0 * x - 1, 1, [-2, 3])
    assert c1.isconst
    assert c2.isconst


def test_support(property_functions):
    f0, f1, f2 = property_functions
    assert isinstance(f0.support, Domain)
    assert isinstance(f1.support, Domain)
    assert isinstance(f2.support, Domain)
    assert f0.support.size == 0
    assert np.equal(f1.support, [-1, 1]).all()
    assert np.equal(f2.support, [-1, 2]).all()


def test_vscale(property_functions):
    f0, f1, f2 = property_functions
    assert f0.vscale == 0
    assert f1.vscale == 1
    assert f2.vscale == 4
