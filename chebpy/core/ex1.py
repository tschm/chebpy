import numpy as np


def chebpts2(n: int) -> np.ndarray:
    """Compute Chebyshev points of the second kind.

    This function calculates the n Chebyshev points of the second kind in the
    interval [-1, 1], which are the extrema of the Chebyshev polynomial T_{n-1}
    together with the endpoints ±1.

    Args:
        n (int): Number of points to compute.

    Returns:
        numpy.ndarray: Array of n Chebyshev points of the second kind.

    Note:
        The points are ordered from left to right on the interval [-1, 1].
    """
    if n == 1:
        pts = np.array([0.0])
    else:
        nn = np.arange(n)
        pts = np.cos(nn[::-1] * np.pi / (n - 1))
    return pts

def bench():
    from numpy.polynomial.chebyshev import chebpts2 as np_chebpts2
    np.testing.assert_allclose(chebpts2(1), np.array([0.0]))

    for n in range(2, 100):
        np.testing.assert_array_almost_equal(chebpts2(2), np_chebpts2(2))

if __name__ == '__main__':
    bench()

    #np.polynomial.chebyshev.chebpts2
