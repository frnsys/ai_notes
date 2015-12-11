Likelihood function $L(\theta)$ is the probability of the data $D$ as a function of the parameters $\theta$.

This often has very small values so typically we work with the log-likelihood function instead:

$$
\ell (\theta) = \log L(\theta)
$$

The _maximum likelihood criterion_ simply involves choosing the parameter $\theta$ to maximize $\ell (\theta)$. This can (sometimes) be done analytically by computing the derivative and setting it to zero and yields the _maximum likelihood estimate_.

MLE's weakness is that if you have only a little training data, it can overfit. This problem is known as _data sparsity_. For example, you flip a coin twice and it happens to land on heads both times. Your maximum likelihood estimate for $\theta$ (probability that the coin lands on heads) would be 1! We can then try to generalize this estimate to another dataset and test it by measuring the log-likelihood on the test set. If a tails shows up at all in the test set, we will have a test log-likelihood of $-\infty$.

We can instead use Bayesian techniques for parameter estimation. In Bayesian parameter estimation, we treat the parameters $\theta$ as a random variable as well, so we learn a joint distribution $p(\theta, D)$.

We first require a prior distribution $p(\theta)$ and the likelihood $p(D|\theta)$ (as with maximum likelihood).

We want to compute $p(\theta|D)$, which is accomplished using Bayes' rule:

$$
p(\theta|D) = \frac{p(\theta)p(D|\theta)}{\int p(\theta') p(D|\theta')d\theta'}
$$

Though we work with only the numerator for as long as possible (i.e. we delay normalization until it's necessary):

$$
p(\theta|D) \varpropto p(\theta)p(D|\theta)
$$

The more data we observe, the less uncertainty there is around the parameter, and the likelihood term comes to dominate the prior - we say that the _data overwhelm the prior_.

We also have the _posterior predictive distribution_ $p(D'|D)$, which is the distribution over future observables given past observations. This is computed by computing the posterior over $\theta$ and then marginalizing out $\theta$:

$$
p(D'|D) = \int p(\theta|D) p(D'|\theta) d\theta
$$

The normalization step is often the most difficult, since we must compute an integral over potentially many, many parameters.

We can instead formulate Bayesian learning as an optimization problem, allowing us to avoid this integral. In particular, we can use _maximum a-posteriori_ (MAP) approximation.

Whereas with the previous Bayesian approach (the "full Bayesian" approach) we learn a distribution over $\theta$, with MAP approximation we simply get a point estimate (that is, a single value rather than a full distribution). In particular, we get the parameters that are most likely under the posterior:

$$
\begin{aligned}
\hat \theta_{\text{MAP}} &= \argmax_{\theta} p(\theta|D) \\
&= \argmax_{\theta} p(\theta,D) \\
&= \argmax_{\theta} p(\theta)p(D|\theta) \\
&= \argmax_{\theta} \log p(\theta) + \log p(D|\theta)
\end{aligned}
$$

Maximizing $\log p(D|\theta)$ is equivalent to MLE, but now we have an additional prior term $\log p(\theta)$. This prior term functions somewhat like a regularizer. In fact, if $p(\theta)$ is a Gaussian distribution centered at 0, we have L2 regularization.






## References

- Learning probabilistic models. Roger Grosse, Nitish Srivastava.