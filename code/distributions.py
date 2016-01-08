import numpy as np
from scipy import stats


## Poisson distribution
lmbda = 3 # mean
y = stats.poisson.pmf(range(10), lmbda)
# plt.plot(y, 'ro')

## Normal distribution
mu, sig = 10, 3
xs = np.random.linspace(-10, 30)
y = stats.norm.pdf(xs, mu, sig)
# plt.plot(xs, y, 'ro')