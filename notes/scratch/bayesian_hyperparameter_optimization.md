Hyperparameter tuning is often treated as an art, i.e. without a reliable and practical systematic process for optimizing them. There are some methods, such as random search and grid search, but they don't perform particularly well.

We can use Bayesian optimization to select good hyperparameters for us. We can sample hyperparameters from a Gaussian process (the prior) and use the result as observations to compute a posterior distribution. Then we select the next hyperparameters to try by optimizing the expected improvement over the current best result or the Gaussian process upper confidence bound (UCB). In particular, we choose an _acquisition function_ to construct a utility function from the model posterior - this is what we use to decide what next set of hyperparameters to try.

One problem is that computing the results of a hyperparameter sample can be very expensive (for instance, if you are training a large neural network).

We use a Gaussian process because its properties allow us to compute marginals and conditionals in closed form.

Some notation for the following:

- $f(x)$ is the function drawn from the Gaussian process prior, where $x$ is the set of hyperparameters
- observations are in the form $\{x_n, y_n\}_{n=1}^{N}$, where $y_n \sim \mathcal N (f(x_n), v)$ and $v$ is the variance of noise introduced into the function observations
- the acquisition function is $a : \mathcal X \to \mathbb R^+$, where $\mathcal X$ is the hyperparameter space
- the next set of hyperparameters to try is $x_{\text{next}} = \argmax_x a(x)$
- the current best set of hyperparameters is $x_{\text{best}}$
- $\Phi()$ denotes the cumulative distribution function of the standard normal

A few popular choices of acquisition functions include:

- _probability of improvement_: with a Gaussian process, this can be computed analytically as:

$$
\begin{aligned}
a_{\text{PI}}(x ; \{x_n, y_n\} \theta) &= \Phi(\gamma(x)) \\
\gamma(x) &= \frac{f(x_{\text{best}} - \mu(x; \{x_n, y_n\}, \theta)}{\sigma(x; \{x_n, y_n\}, \theta)}
\end{aligned}
$$

- _expected improvement_: under a Gaussian process, this also has a closed form:

$$
a_{\text{EI}} (x; \{x_n, y_n\}, \theta) = \sigma(x; \{x_n, y_n\}, \theta) (\gamma(x)\Phi(\gamma(x)) + \mathcal N (\gamma(x); 0, 1))
$$

- _Gaussian process upper confidence bound_: use upper confidence bounds (when maximizing, otherwise, lower confidence bounds) to construct acquisition functions that minimize regret over the course of their optimization:

$$
a_{\text{LCB}} (x; \{x_n, y_n\}, \theta) = \mu(x; \{x_n, y_n\}, \theta) - \kappa \sigma(x; \{x_n, y_n\}, \theta)
$$

Where $\kappa$ is tunable to balance exploitation against exploration.

Some difficulties with Bayesian optimization of hyperparameters include:

- often unclear what the appropriate choice for the covariance function and its associated hyperparameters (these hyperparameters are distinct from the ones the method is optimizing; i.e. these are in some sense "hyper-hyperparameters")
- the function evaluation can be a time-consuming optimization procedure. One method is to optimize expected improvement _per second_, thereby taking wall clock time into account. That way, we prefer to evaluate points that are not only likely to be good, but can also be evaluated quickly. However, we don't know the _duration function_ $c(x) : \mathcal X \to \mathbb R^+$, but we can use this same Gaussian process approach to model $c(x)$ alongside $f(x)$.

Furthermore, we can parallelize these Bayesian optimization procedures (refer to paper)

## References

- Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle, Ryan P. Adams.