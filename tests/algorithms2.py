import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

from chebpy.core.algorithms import vals2coeffs2, chebpts2

def f(x):
    return x**2 + np.sin(x)

# Target interval
a, b = -5, 3

# Chebyshev points on [-1, 1], then mapped to [a, b]
n = 17
pts = chebpts2(n=n)                     # on [-1, 1]
pts_mapped = 0.5 * (b - a) * pts + 0.5 * (a + b)

# Sample f at mapped Chebyshev points
vals = f(pts_mapped)

# Compute coefficients (on standard [-1, 1] domain)
c = vals2coeffs2(vals)

# Create Chebyshev object on domain [a, b]
p = cheb.Chebyshev(c, domain=[a, b])

# Evaluate and plot
x = np.linspace(a, b, 500)
plt.plot(x, f(x), 'k--', label='Original f(x)')
plt.plot(x, p(x), 'r-', label='Chebyshev interpolant')
plt.scatter(pts_mapped, vals, c='blue', label='Chebyshev points')
plt.grid(True)
plt.legend()
plt.title("Chebyshev Interpolation on [-5, 3]")
plt.show()