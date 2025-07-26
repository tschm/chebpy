"""Unit-tests for Complex functionality in chebpy/core/chebfun.py"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun


@pytest.fixture
def complex_chebfun():
    return Chebfun.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x), [-1, 1])


def test_init_empty():
    Chebfun.initempty()


def test_roots(complex_chebfun):
    z = complex_chebfun
    r0 = z.roots()
    r1 = (z - 1).roots()
    r2 = (z - 1j).roots()
    r3 = (z + 1).roots()
    r4 = (z + 1j).roots()
    assert r0.size == 0
    assert np.allclose(r1, [0])
    assert np.allclose(r2, [0.5])
    assert np.allclose(r3, [-1, 1])
    assert np.allclose(r4, [-0.5])


def test_rho_ellipse_construction(complex_chebfun):
    z = complex_chebfun
    zz = 1.2 * z
    e = 0.5 * (zz + 1 / zz)
    assert e(1) - e(-1) == pytest.approx(0, abs=1e-14)
    assert e(0) + e(-1) == pytest.approx(0, abs=1e-14)
    assert e(0) + e(1) == pytest.approx(0, abs=1e-14)


def test_calculus(complex_chebfun):
    z = complex_chebfun
    assert np.allclose([z.sum()], [0])
    assert (z.cumsum().diff() - z).isconst
    assert (z - z.cumsum().diff()).isconst


def test_real_imag(complex_chebfun):
    z = complex_chebfun
    # check definition of real and imaginary
    zreal = z.real()
    zimag = z.imag()
    np.testing.assert_equal(zreal.funs[0].coeffs, np.real(z.funs[0].coeffs))
    np.testing.assert_equal(zimag.funs[0].coeffs, np.imag(z.funs[0].coeffs))
    # check real part of real chebtech is the same chebtech
    assert zreal.real() == zreal
    # check imaginary part of real chebtech is the zero chebtech
    assert zreal.imag().isconst
    assert zreal.imag().funs[0].coeffs[0] == 0
