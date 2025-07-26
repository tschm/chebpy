import pytest

from chebpy import UserPreferences
from chebpy.core.settings import _preferences


def userPref(name):
    """Get the current preference value for name.
    This is how chebpy core modules access these."""
    return getattr(_preferences, name)


def test_update_pref():
    eps_new = 1e-3
    eps_old = userPref("eps")
    with UserPreferences() as prefs:
        prefs.eps = eps_new
        assert eps_new == userPref("eps")
    assert eps_old == userPref("eps")


def test_reset():
    def change(reset_named=False):
        prefs = UserPreferences()
        prefs.eps = 99
        if reset_named:
            prefs.reset("eps")
        else:
            prefs.reset()

    eps_bak = userPref("eps")
    change(False)
    assert eps_bak == userPref("eps")
    change(True)
    assert eps_bak == userPref("eps")
