## Poisson distribution

```python
from scipy import stats
lmbda = 3 # mean
y = stats.poisson.pmf(range(10), lmbda)
plt.plot(y, 'ro')
```

## Normal distribution

```python
import numpy as np
from scipy import stats
mu, sig = 10, 3
xs = np.random.linspace(-10, 30)
y = stats.norm.pdf(xs, mu, sig)
plt.plot(xs, y, 'ro')
```
