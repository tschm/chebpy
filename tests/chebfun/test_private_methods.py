"""Unit-tests for Private Methods in chebpy/core/chebfun.py"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import Domain

from ..utilities import infnorm

# aliases
sin = np.sin
eps = DefaultPreferences.eps


@pytest.fixture
def private_methods_functions():
    def f(x):
        return sin(x - 0.1)

    f1 = Chebfun.initfun_adaptive(f, [-2, 0, 3])
    f2 = Chebfun.initfun_adaptive(f, np.linspace(-2, 3, 5))

    return f1, f2


# in the test_break_x methods, we check that (1) the newly computed domain
# is what it should be, and (2) the new chebfun still provides an accurate
# approximation
def test__break_1(private_methods_functions):
    f1, _ = private_methods_functions
    altdom = Domain([-2, -1, 1, 2, 3])
    newdom = f1.domain.union(altdom)
    f1_new = f1._break(newdom)
    assert f1_new.domain == newdom
    assert f1_new.domain != altdom
    assert f1_new.domain != f1.domain
    xx = np.linspace(-2, 3, 1000)
    error = infnorm(f1(xx) - f1_new(xx))
    assert error <= 3 * eps


def test__break_2(private_methods_functions):
    f1, _ = private_methods_functions
    altdom = Domain([-2, 3])
    newdom = f1.domain.union(altdom)
    f1_new = f1._break(newdom)
    assert f1_new.domain == newdom
    assert f1_new.domain != altdom
    xx = np.linspace(-2, 3, 1000)
    error = infnorm(f1(xx) - f1_new(xx))
    assert error <= 3 * eps


def test__break_3(private_methods_functions):
    _, f2 = private_methods_functions
    altdom = Domain(np.linspace(-2, 3, 1000))
    newdom = f2.domain.union(altdom)
    f2_new = f2._break(newdom)
    assert f2_new.domain == newdom
    assert f2_new.domain != altdom
    assert f2_new.domain != f2.domain
    xx = np.linspace(-2, 3, 1000)
    error = infnorm(f2(xx) - f2_new(xx))
    assert error <= 3 * eps
