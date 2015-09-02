
# Bayesian Inference

_Bayesian inference_ is an approach to statistics contrasted with frequentist approaches.

Generally, the approach is:

1. You have some observed data.
2. You have a model for this data. That is, you have a program which attempts to predict or generate "true" data that is qualitatively similar to your observed data.
3. Your model has some unknown parameters. That is, your program is not very good at generating new data.
4. __Bayesian inference__ "inverts" your model so that, instead of generating new data from your parameters, you input observed data to generate your parameters.
5. The parameters you get from Bayesian inference are not explicit parameters (i.e. fixed values), but rather a probability distribution over the parameters. That is, you get a range of values that are likely to be your parameters, given the observed data you had.
6. This probability distribution over the parameters is the "posterior", and can be used to define hypothesis tests and refine your parameters.
7. You can plug the posterior parameters back into your model and try generating new data. If the generated data is still off from the observed data, you may have a bad model. This is called the __posterior predictive test__.

- <http://www.reddit.com/r/statistics/comments/15y6oi/can_someone_very_briefly_defineexplain_bayesian/>

Bayesian models are also described by (estimated) parameters, and we describe the uncertainty of those estimates with probability distributions. Fundamentally, this is:

$$
p(\theta|y)
$$

Where the parameters $\theta$ are the unknown, so we express them as a probability distribution, given the observations $y$. This probability distribution is the __posterior distribution__.

So you must decide (specify) probability distributions for both the data sample and for the unknown parameters. These decisions involve making a lot of assumptions. Then you must compute a posterior distribution, which often cannot be calculated analytically - so other methods are used (such as simulations, described later).

From the posterior distribution, you can calculate point estimates, credible intervals, quantiles, and make predictions.

Finally, because of the assumptions which go into specifying the initial distributions, you must test your model and see if it fits the data and seems reasonable.


## Frequentist vs Bayesian approaches

For frequentists, probability is thought of in terms of frequencies, i.e. the probability of the event is the amount of times it happened over the total amount of times it could have happened.

In frequentist statistics, the observed data is considered random; if you gathered more observations they would be different according to the underlying distribution. The parameters of the model, however, are considered fixed.

For Bayesians, probability is belief or certainty about an event. Observed data is considered fixed, but the model parameters are random (uncertain) instead and considered to be drawn from some probability distribution.

Another way of phrasing this is that frequentists are concerned with uncertainty in the data, whereas Bayesians are concerned with uncertainty in the parameters.


## Bayes' Rule

In frequentist statistics, many different estimators may be used, but in Bayesian statistics the only estimator is Bayes' Formula (aka Bayes' Rule or Bayes' Theorem).

Bayes' Theorem, aka Bayes' Rule:

- $H$ is the hypothesis (more commonly represented as the parameters $\theta$)
- $D$ is the data

$$
P(H|D) = \frac{P(H)P(D|H)}{P(D)}
$$

- $P(H)$ = the probability of the hypothesis before seeing the data. The _prior_.
- $P(H|D)$ = probability of the hypothesis, given the data. The _posterior_.
- $P(D|H)$ = the probability of the data under the hypothesis. The _likelihood_.
- $P(D)$ = the probability of data under _any_ hypothesis. The _normalizing constant_.

For an example of likelihood:

If I want to predict the sides of a dice I rolled, and then I rolled an 8, then $P(D|\text{a six sided die}) = 0$. That is, it is impossible to have my observed data under the hypothesis of having a six sided die.

A key insight to draw from Bayes' Rule is that $P(H|D) \propto p(H)P(D|H)$, that is, the posterior is proportional to the product of the prior and the likelihood.

Note that the normalizing constant $P(D)$ usually cannot be directly computed and is equivalent to $\int P(D|H)P(H) dH$ (which is usually intractable since their are usually multiple parameters of interest, resulting in a multidimensional integration problem. If $\theta$, the parameters, is one dimensional, then you could integrate it rather easily).

One workaround is to do approximate inference with non-normalized posteriors, since we know that the posterior is proportional to the numerator term:

$$
P(H|D) \propto P(H)P(D|H)
$$

Another workaround to approximate the posterior using simulation methods such as Monte Carlo.


## Some probability distributions

Given a set of hypotheses $H_0, H_1, \dots, H_n$, the distribution for the priors of these hypotheses is the _prior distribution_, i.e. $P(H_0), P(H_1), \dots, P(H_n)$.

The distribution of the posterior probabilities is the _posterior distribution_, i.e. $P(H_0|D), P(H_1|D), \dots, P(H_n|D)$.

Let $Z$ be some random variable. Associated with $Z$ is a _probability distribution function_ that assigns probabilities to the different outcomes that $Z$ can take.

There are 3 classifications of random variables and their distributions:

- _discrete_ - in which case, the distribution is called the _probability mass function_ (PDF)
- _continuous_ - in which case, the distribution is called the _probability density function_ (PDF)
- _mixed_ - discrete + continuous

The probability that $Z$ takes on a value $k$ is denoted $P(Z=k)$.

### The Poisson Distribution

One discrete distribution is the _Poisson_ distribution:

$$
P(Z=k) = \frac{\lambda^k e^{-\lambda}}{k!}, k=0,1,2,\dots
$$

If $Z$ is Poisson-distributed, we notate it:

$$
Z \sim Poi(\lambda)
$$

$\lambda$ is a parameter of the distribution. For Poisson, it can be any positive number.

Increasing $\lambda$ assigns more probability to large values; decreasing it assigns more probability to small values. It is sometimes called the _intensity_ of the distribution.

For a Poisson distribution, the expected value is equal to its parameter:

$$
E[Z|\lambda] = \lambda
$$

The Poisson distribution assigns a probability to _every_ non-negative integer.


### The Exponential Distribution

A random variable which is continuous may have _exponential density_, often describe as an _exponential random variable_:

$$
f_z(z|\lambda) = \lambda e^{-\lambda z}, z \geq 0
$$

Here we say $Z$ is _exponential_:

$$
Z \sim Exp(\lambda)
$$

Its expected value is the inverse of its parameter:

$$
E[Z|\lambda] = \frac{1}{\lambda}
$$


### The challenge of statistics

In statistical analysis, we try to determine (_infer_) _true values_ based on _observed values_.

The challenge in statistics is observing $Z$ and trying to determine $\lambda$ from it. In Bayesian inference, we don't assign an explicit value $\lambda$. Rather, we define it over a probability distribution as well: what values is $\lambda$ _likely_ to take on? That is, we treat $\lambda$ itself as a random variable.

We may say for instance that $\lambda$ is drawn from an exponential distribution:

$$
\lambda \sim Exp(\alpha)
$$

Here $\alpha$ is a _hyperparameter_, that is, it is a parameter for our parameter $\lambda$.


### The Binomial Distribution

The Binomial distribution is a discrete distribution. It has two parameters:

- $N$ - a positive integer representing the number of trials
- $p$ - the probability of an event occurring in a single trial

The mass distribution looks like:

$$
P(Z=k) = {N \choose k}p^k (1-p)^{N-k}
$$

For a random variable $Z$ it is denoted

$$
Z \sim Bin(N, p)
$$

Here $Z$ ends up being the number of events that occurred over our trials.

It's expected value is:

$$
E[Z|N,p] = Np
$$

The special case $N=1$ corresponds to the _Bernoulli distribution_.

If we have $Z_1, Z_2, \dots, Z_N$ Bernoulli random variables with the same $p$, then $X = Z_1 + Z_2 + \dots + Z_N \sim Binomial(N,p)$.

The expected value of a Bernoulli random variable is $p$ (because $N=1$).

### The Bernoulli Distribution

Notated $Z \sim Ber(p))$. $Z$ is 1 with probability $p$ and $Z$ is 0 with probability $1-p$.


### The Normal Distribution

A normal random variable $X$ is represented:

$$
X \sim N(\mu, \frac{1}{\tau})
$$

Where the parameters are:

- $\mu$ = the mean
- $\tau$ = the precision. Note that $\tau^{-1} = \sigma^2$. The smaller the $\tau$, the larger the spread of the distribution (that is, the greater the uncertainty).

The probability density function is:

$$
f(x|\mu, \tau) = \sqrt{\frac{\tau}{2\pi}} exp(-\frac{\tau}{2}(x - \mu)^2)
$$

The expected value is:

$$
E[X|\mu, \tau] = \mu
$$


## Markov Chain Monte Carlo (MCMC)

When we get our prior distribution, how do we identify what the most likely value for our random variable is?

When working in one dimension, it is not too hard. We can plot the graph and look at it.

But we may be doing a Bayesian inference problem with $N$ unknowns (i.e. a multivariate inference problem, where $N$ becomes a vector) and we end up working in $N$-dimensional space, and identifying this most likely value can be computationally intractable.

We can use Markov Chain Monte Carlo to intelligently search this space to find the posterior "mountain" in the posterior distribution "landscape". Here we can find the most likely "true" value for our random variable.

A Monte Carlo estimate would involve collecting thousands of random samples from your posterior distribution and use those to answer a question. Say you want to see the probability that your random variable is less than 0.5. You'd sample ten thousand independent points and count how many of those were less than 0.5, and use that as your answer.

MCMC integrates Markov chains into the process - that is, it uses a memoryless process which only cares about the current and next value. It works like this:

1. Start at the current position.
2. Propose moving to a new position.
3. Accept or reject the new position based on the position's adherence to the data and prior distributions.
4. If accept: move to the new position and go back to step 1. Else, don't move and just go back to step 1.
5. After a large number of iterations, return all accepted positions.

If all goes well, this list of positions should represent your distribution.

Note that there is a "burn-in" period of the first few thousand points when the algorithm is just starting; this burn-in period's data is usually very noisy. So these first values are often just thrown out.

Note that MCMC is a class of algorithms, not an algorithm itself. Two popular MCMC algorithms are Gibbs sampling and Metropolis-Hastings.

### Metropolis-Hastings Algorithm

The Metropolis-Hasting algorithm uses Markov chains with rejection sampling.

The proposal density $g(\theta_t)$ is chosen as in rejection sampling, but it depends on $\theta_{t-1}$, i.e. $g(\theta_t|\theta_{t-1})$.

First select some initial $\theta$, $\theta_1$.

Then for $n$ iterations:

* Draw a candidate $\theta_t^c \sim g(\theta_t|\theta_{t-1})$
* Compute the Metropolis-Hastings ratio: $R = \frac{f(\theta_t^c)g(\theta_{t-1}|\theta_t^c)}{f(\theta_{t-1})g(\theta_t^c|\theta_{t-1})}$
* Draw $u \sim \text{Uniform}$
* If $u < R$, accept $\theta_t = \theta_t^c$, otherwise, $\theta_t = \theta_{t-1}$

There are a few required properties of the Markov chain for this to work properly:

- The stationary distribution of the chain must be the target density:
    - The chain must be _recurrent_ - that is, for all $\theta \in \Theta$ in the _target density_ (the density we wish to approximate), the probability of returning to any state $\theta_i \in Theta = 1$. That is, it must be possible _eventually_ for any state in the state space to be reached.
    - The chain must be _non-null_ for all $\theta \in \Theta$ in the target density; that is, the expected time to recurrence is finite.
    - The chain must have a stationary distribution equal to the target density.
- The chain must be _ergodic_, that is:
    - The chain must be _irreducible_ - that is, any state $\theta_i$ can be reached from any other state $\theta_j$ in a finite number of transitions (i.e. the chain should not get stuck in any infinite loops)
    - The chain must be _aperiodic_ - that is, there should not be a fixed number of transitions to get from any state $\theta_i$ to any state $\theta_j$. For instance, it should not always take three steps to get from one place to another - that would be a period. Another way of putting this - there are no fixed cycles in the chain.

It can been proven that the stationary distribution of the Metropolis-Hastings algorithm is the target density (proof omitted).

Th ergodic property (whether or not the chain "mixes" well) can be validated with some _convergence diagnostics_  A common method is to plot the chain's values as their drawn and see if the values tend to concentrate around a constant; if not, you should try a different proposal density.

Alternatively, you can look at an autocorrelation plot, which measures the internal correlation (from -1 to 1) over time, called "lag". We expect that the greater the lag, the less the points should be autocorrelated - that is, we expect autocorrelation to smoothly decrease to 0 with increasing lag. If autocorrelation remains high, then the chain is not fully exploring the space. Autocorrelation can be improved by _thinning_, which is a technique where only every $k$th draw is kept and others are discarded.

Finally, you also have the options of running multiple chains, each with different starting values, and combining those samples.

You should also use burn-in.

#### References

- POLS 506: Basic Monte Carlo Procedures and Sampling. Justin Esarey. <https://www.youtube.com/watch?v=cxWzsCoYT8Q>
- POLS 506: Metropolis-Hastings, the Gibbs Sampler, and MCMC. Justin Esarey. <https://www.youtube.com/watch?v=j4nEAqUUnVw>


## Getting a single value answer

Bayesian inference returns a distribution (the posterior) but we often need a single value (or a vector in multivariate cases). So we choose a value from the posterior. This value is a _Bayesian point estimate_.

Selecting the MAP (_maximum a posterior_) value is insufficient because it neglects the shape of the distribution.

Suppose $P(\theta|X)$ is the posterior distribution of $\theta$ after observing data $X$.

The _expected loss_ of choosing estimate $\hat \theta$ to estimate $\theta$ (the true parameter), also known as the _risk_ of estimate $\hat \theta$ is:

$$
l(\hat \theta) = E_{\theta} [L(\theta, \hat \theta)]
$$

Where $L(\theta, \hat \theta)$ is some loss function.

You can approximate the expected loss using the Law of Large Numbers, which just states that as sample size grows, the expected value approaches the actual value. That is, as $N$ grows, the expected loss approaches 0.

For approximating expected loss, it looks like:

$$
\frac{1}{N} \sum^N_{i=1} L(\theta_i, \hat \theta) \approx E_{\theta}[L(\theta, \hat \theta)] = l(\hat \theta)
$$

You want to select the estimate $\hat theta$ which minimizes this expected loss:

$$
\text{argmin}_{\hat \theta} E_{\theta}[L(\theta, \hat \theta)]
$$


## Choosing a prior distribution Bayesian inference

With Bayesian inference, we must _choose_ a prior distribution, then apply data to get our posterior distribution. The prior is chosen based on domain knowledge or intuition or perhaps from the results of previous analysis; that is, it is chosen subjectively - there is no prescribed formula for picking a prior. If you have no idea what to pick, you can just pick a uniform distribution as your prior.

Your choice of prior will affect the posterior that you get, and the subjectivity of this choice is what makes Bayesian statistics controversial - but it's worth noting that all of statistics, whether or frequentist or Bayesian, involves many subjective decisions (e.g. frequentists must decide on an estimator to use, what data to collect and how, and so on) - what matters most is that you are explicit about your decisions and why you made them.

Say we perform an Bayesian analysis and get a posterior. Then we get some new data for the same problem. We can re-use the posterior from before as our prior, and when we run Bayesian analysis on the new data, we will get a new posterior which reflects the additional data. We don't have to re-do any analysis on the data from before, all we need is the posterior generated from it.

For any unknown quantity we want to model, we say it is drawn from some prior of our choosing. This is usually some parameter describing a probability distribution, but it could be other values as well. This is central to Bayesian statistics - all unknowns are represented as distributions of possible values. In Bayesian statistics: if there's a value and you don't know what it is, come up with a prior for it and add it to your model!

If you think of distributions as landscapes or surfaces, then the data deforms the prior surface to mold it into the posterior distribution.

The surface's "resistance" to this shaping process depends on the selected prior distribution.

When it comes to selecting Bayesian priors, there are two broad categories:

- _objective priors_ - these let the data influence the posterior the most
- _subjective priors_ - these allow the practitioner to asset their own views in to the prior. This prior can be the posterior from another problem or just come from domain knowledge.

An example objective prior is a _uniform_ (flat) prior where every value has equal weighting. Using a uniform prior is called _The Principle of Indifference_. Note that a uniform prior restricted within a range is _not_ objective - it has to be over _all_ possibilities.

Here are some useful priors.

### The Gamma Distribution

$$
X \sim Gamma(\alpha, \beta)
$$

This is over positive real numbers.

It is just a generalization of the exponential random variable:

$$
Exp(\beta) \sim Gamme(1, \beta)
$$

The PDF is:

$$
f(x|\alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha -1}e^{\beta x}}{\Gamma(\alpha)}
$$

Where $\Gamma(\alpha)$ is the _Gamma function_.

### The Beta Distribution

$$
X \sim Beta(\alpha, \beta), X \in [0,1]
$$

Its PDF is:

$$
f(x|\alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

Where $B(\alpha, \beta)$ is the _Beta function_.

The Beta distribution is a generalization of the uniform distribution:

$$
Uniform() \sim Beta(1,1)
$$

The Beta & Binomial distributions are also related:

Say we want to find some unknown proportion or probability $p$. We assign a $Beta(\alpha, \beta)$ prior to $p$.

We then observe some data generated by a Binomial process, say $X \sim Binomial(N, p)$, with $p$ still unknown.

Then the posterior also ends up being a Beta distribution, i.e. $p|X \sim Beta(\alpha + X, \beta + N + X)$.

This is an example of a _conjugate prior_. Say random variable $X$ comes from a well-known distribution, $f_{\alpha}$ where $\alpha$ are the possibly unknown parameters of $f$. It could be a normal, binomial, etc distribution.

For a given distribution $f_{\alpha}$, there may exist a prior distribution $p_{\beta}$ such that

$$
\overbrace{p_{\beta}}^{\text{prior}} \cdot \overbrace{f_{\alpha}(X)}^{\text{data}} = \overbrace{p_{\beta'}}^{\text{posterior} }
$$

These usually only exist for simple one-dimensional problems.


Note that the more data you have (as $N$ increases), the choice of prior becomes less important.


## Empirical Bayes

Empirical Bayes is a method which combines frequentist and Bayesian approaches by using frequentist methods to select the hyperparameters.

For instance, say you want to estimate the $\mu$ parameter for a normal distribution.

You could use the empirical sample mean from the observed data:

$$
\mu_p = \frac{1}{N}\sum^N_{i=0} X_i
$$
Where $\mu_p$ denotes the prior $\mu$.

Though if working with not much data, this kind of ends like double-counting your data.



---

Generally real-world statistical inference problems involve some unknown quantity $\theta$ and observed data $X$. Bayesian inference amounts to:

1. Specifying a sampling model for the observed data $X$, conditioned on the unknown $\theta$, such that

$$
X \sim f(X|\theta)
$$

where $f(X|\theta)$ is either the PDF or the PMF (as appropriate).

2. Specifying a marginal or distribution $\pi(\theta)$ for $\theta$, which is the prior distribution ("prior" for short):

$$
\theta \sim \pi(\theta)
$$

From this we wish to compute the posterior, that is, uncover the distribution for $\theta$ given the observed data $X$, like so:

$$
\pi(\theta|X) = \frac{\pi(\theta)L(\theta|X)}{\int \pi(\theta) L(\theta|X) d\theta}
$$

where $L(\theta|X) \propto f(\theta|X)$ in $\theta$, called the likelihood of $\theta$ given $X$.

When it comes to building models, $\theta$ is the parameters which describes our model, and we want to find the most likely parameters given our observed data.



## Sampling/Simulations

With Bayesian inference, in order to describe your posterior, you often must evaluate complex multidimensional integrals (i.e. from very complex, multidimensional probability distributions), which can be computationally intractable.

Instead you can generate sample points from the posterior distribution and use those samples to compute whatever descriptions you need. This technique is called Monte Carlo integration, and the process of drawing repeated random samples in this way is called Monte Carlo simulation.

#### Monte Carlo Integration

Monte Carlo integration is a way to approximate complex integrals using random number generation.

Say we have a complex integral:

$$
\int h(x)dx
$$

If we can decompose $h(x)$ into the product of a function $f(x)$ and a probability density function $p(x)$ describing the probabilities of the inputs $x$, then:

$$
\int h(x)dx = \int f(x)p(x)dx = E_{p(x)}[f(x)]
$$

That is, the result of this integral is the expected value of $f(x)$ over the density $p(x)$.

We can approximate this expected value by taking the mean of many, many samples ($n$ samples):

$$
\int h(x)dx = E_{p(x)}[f(x)] \approx \frac{1}{n} \sum^n_{i=1} f(x_i)
$$

This process of approximating the integral is _Monte Carlo integration_.

For very simple cases of known distributions, we can sample directly, e.g.

    import numpy as np

    # Say we think the distribution is a Poisson distribution
    # and the parameter of our distribution, lambda,
    # is unknown and what we want to discover.
    lam = 5

    # Collect 100000 samples
    sim_vals = np.random.poisson(lam, size=100000)

    # Get whatever descriptions we want, e.g.
    mean = sim_vals.mean()

    # For poisson, the mean is lambda, so we expect
    # them to be approximately equal (given a large enough sample size)
    abs(lam - mean()) < 0.001


### Markov Chain Monte Carlo

For many scenarios it's not so easy to draw independent random samples from the posterior. The distribution may be multivariate or have an unknown function we can generate values from.

Instead we can use a family of techniques known as Markov Chain Monte Carlo to generate samples for us.

#### Markov Chains

Markov chains are a stochastic process in which the next state depends only on the current state.

Consider a random variable $X$ and a time index $t$. The state of $X$ at time $t$ is notated $X_t$.

For a Markov chain, the state $X_{t+1}$ depends only on the current state $X_t$, that is:

$$
P(X_{t+1} = x_{t+1}|X_t = x_t, X_{t-1} = x_{t-1}, \dots, X_0 = x_0) = P(X_{t+1} = x_{t+1}| X_t = x_t)
$$

Where $P(X_{t+1} = x_{t+1})$ is the __transition probability__ of $X_{t+1} = x_{t+1}$. The collection of transition probabilities is called a __transition matrix__ (for discrete states); more generally is is called a __transition kernel__.


If we consider $t$ going to infinity, the Markov chain settles on a __stationary distribution__, where $P(X_t) = P(X_{t-1})$. The stationary distribution does not depend on the initial state of the network. Markov chains are _erdogic_, i.e. they "mix", which means that the influence of the initial state weakens with time (the rate at which it mixes is its _mixing speed_).

If we call the $k \times k$ transition matrix $P$ and the marginal probability of a state at time $t$ is a $k \times 1$ vector $\pi$, then the distribution of the state at time $t+1$ is $\pi'P$. If $\pi'P = \pi'$, then $pi$ is the stationary distribution of the Markov chain.

##### References

- POLS 506: Basic Monte Carlo Procedures and Sampling. Justin Esarey. <https://www.youtube.com/watch?v=cxWzsCoYT8Q>

#### Markov Chain Monte Carlo

Rather than directly compute the integral for posterior distributions in Bayesian analysis, we can instead draw several (thousands, millions, etc) samples from the probability distribution through the technique of Markov Chain Monte Carlo (MCMC, which builds off of Monte Carlo integration), then use these samples to compute whatever descriptions we'd like about the distribution (often this is some expected value of a function, $E[f(x)]$, where its inputs are drawn from distribution, i.e. $x \sim p$, where $p$ is some probability distribution).

You start with some random initial sample and, based on that sample, you pick a new sample. This is the Markov Chain aspect of MCMC - the next sample you choose depends only on the current sample. This works out so that you spend most your time with high probability samples (b/c they have higher transition probabilities) but occasionally jump out to lower probability samples. Eventually the MCMC chain will converge on a random sample.

So we can take all these $N$ samples and, for example, compute the expected value:

$$
E[f(x)] \approx \frac{1}{N} \sum^N_{i=1}f(x_i)
$$

Because of the random initialization, there is a "burn-in" phase in which the sampling model needs to be "warmed up" until it reaches an equilibrium sampling state, the _stationary distribution_. So you discard the first hundred or thousand or so samples as part of this burn-in phase. You can (eventually) arrive at this stationary distribution _independent of where you started_ which is why the random initialization is ok - this is an important feature of Markov Chains.

MCMC is a general technique of which there are specific algorithms. One of the most popular ones is _Gibbs sampling_.

#### Gibbs Sampling

It is easy to sample from simple distributions. For example, for a binomial distribution, you can basically just flip a coin. For a multinomial distribution, you can basically just roll a dice.

If you have a multinomial, multivariate distribution, e.g. $P(x_1, x_2, \dots, x_n)$, things get more complicated. If the variables are independent, you can factorize the multivariate distribution as a product of univariate distributions, treating each as a univariate multinomial distribution, i.e. $P(x_1, x_2, \dots, x_n) = P(x_1) \times P(x_2) \times \dots \times P(x_n)$. Then you can just sample from each distribution individually, i.e. as a dice roll.

However - what if these aren't independent, and we want to sample from the _joint distribution_ $P(x_1, x_2, \dots, x_n)$? We can't factorize it into simpler distributions like before.

With Gibbs sampling we can approximate this joint distribution under the condition that we can easily sample from the conditional distribution for each variable, i.e. $P(x_i | x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n)$. (This condition is satisfied on Bayesian networks.)

We take advantage of this and iteratively sample from these conditional distributions and using the most recent value for each of the other variables (starting with random values at first). For example, sampling $x_1|x_2, \dots, x_n$, then fixing this value for $x_1$ while sampling $x_2|x_1, x_3, \dots, x_n$, then fixing both $x_1$ and $x_2$ while sampling $x_3|x_1, x_2, x_4, \dots, x_n$, and so on.

If you iterate through this a large number of times you get an approximation of samples taken from the actual joint distribution.

Another way to look at Gibbs sampling:

Say you have random variables $c, r, t$ (cloudy, raining, thundering) and you have the following probability tables:

| c | P(c) |
|---|------|
| 0 | 0.5  |
| 1 | 0.5  |

| c | r | P(r&#124;c) |
|---|---|-------------|
| 0 | 0 | 0.9         |
| 0 | 1 | 0.1         |
| 1 | 0 | 0.1         |
| 1 | 1 | 0.9         |

| c | r | t | P(t&#124;c,r) |
|---|---|---|---------------|
| 0 | 0 | 0 | 0.9           |
| 0 | 0 | 1 | 0.1           |
| 0 | 1 | 0 | 0.5           |
| 0 | 1 | 1 | 0.5           |
| 1 | 0 | 0 | 0.6           |
| 1 | 0 | 1 | 0.4           |
| 1 | 1 | 0 | 0.1           |
| 1 | 1 | 1 | 0.9           |

We can first pick some starting sample, e.g. $c=1,r=0,t=1$.

Then we fix $r=0, t=1$ and randomly pick another $c$ value according to the probabilities in the table (here it is equally likely that we get $c=0$ or $c=1$). Say we get $c=0$. Now we have a new sample $c=0,r=0,t=1$.

Now we fix $c=0,t=1$ and randomly pick another $r$ value. Here $r$ is dependent only on $c$. $c=0$ so we have a $0.9$ probability of picking $r=0$. Say that we do. We have another sample $c=0,r=0,t=1$, which happens to be the same as the previous sample.

Now we fix $c=0,r=0$ and pick a new $t$ value. $t$ is dependent on both $c$ and $r$. $c=0,r=0$, so we have a $0.9$ chance of picking $t=0$. Say that we do. Now we have another sample $c=0,r=0,t=0$. Then we repeat this process until convergence (or for some specified number of iterations).

Your samples will reflect the actual joint distribution of these values, since more likely samples are, well, more likely to be generated.


#### Rejection Sampling

Monte Carlo integration allows us to draw samples from a posterior distribution with a known parametric form. It does not, however, enable us to draw samples from a posterior distribution without a known parametric form. We may instead use __rejection sampling__ in such cases.

We can take our function $f(x)$ and if it has bounded/finite _support_ ("support" is the $x$ values where $f(x)$ is non-zero, and can be thought of the range of meaningful $x$ values for $f(x)$), we can calculate its maximum and then define a bounding rectangle with it, encompassing all of the support values. This envelope function should contain all possible values of $f(x)$  Then we can randomly generate points from within this box and check if they are under the curve (that is, less than $f(x)$ for the point's $x$ value). If a point is not under the curve, we reject it. Thus we approximate the integral like so:

$$
\frac{\text{points under curve}}{\text{points generated}} \times \text{box area} = \lim_{n \to \infty} \int_A^B f(x)dx
$$

In the case of unbounded support (i.e. infinite tails), we instead choose some _majorizing_ or _enveloping_ function $g(x)$ ($g(x)$ is typically a probability density itself and is called a _proposal density_) such that $cg(x) \geq f(x) \, , \forall x \in (-\infty, \infty)$, where $c$ is some constant. This functions like the bounding box from before. It completely encloses $f$. Ideally we choose $g(x)$ so that it is close to the target distribution, that way most of our sampled points can be accepted.

Then, for each $x_i$ we draw (i.e. sample), we also draw a uniform random value $u_i$. Then if $u_i < \frac{f(x_i)}{cg(x_i)}$, we accept $x_i$, otherwise, we reject it.

The intuition here is that the probability of a given point being accepted is proportional to the function $f$ at that point, so when there is greater density in $f$ for that point, that point is more likely to be accepted.

In multidimensional cases, you draw candidates from every dimension simultaneously.

##### References

- POLS 506: Basic Monte Carlo Procedures and Sampling. Justin Esarey. <https://www.youtube.com/watch?v=cxWzsCoYT8Q>

## Credible Intervals (or "credible regions")

In Bayesian statistics, The closest analog to confidence intervals in frequentist statistics is the __credible interval__. It is _much_ easier to interpret than the confidence interval because it is exactly what most people confuse the confidence interval to be. For instance, the 95% credible interval is the interval in which we expect to find $\theta$ 95% of the time.

Mathematically this is expressed as:

$$
P(a(y) < \theta < b(y)|Y=y) = 0.95
$$

We condition on $Y$ because in Bayesian statistics, the data is fixed and the parameters are random.

## Beta-Binomial Model

The Beta-Binomial model is a useful Bayesian model because it provides values between 0 and 1, which is useful for estimating probabilities or percentages.

It involves, as you might expect, a beta and a binomial distribution.

So say we have $N$ trials and observe $n$ successes. We describe these observations by a binomial distribution, $n \sim \text{Binomial}(N, p)$ for which $p$ is unknown. So we want to come up with some distribution for $p$ (remember, with Bayesian inference, you do not produce point estimates, that is, a single value, but a distribution for your unknown value to describe the uncertainty of its true value).

For frequentist inference we'd estimate $\hat p = \frac{n}{N}$ which isn't quite good for low numbers of $N$.

This being Bayesian inference, we first must select a prior. $p$ is a probability and therefore is bound to $[0, 1]$. So we could choose a uniform prior over that interval; that is $p \sim \text{Uniform}(0,1)$.

However, $\text{Uniform}(0, 1)$ is equivalent to a beta distribution where $\alpha=1, \beta=1$, i.e. $\text{Beta}(1,1)$. The beta distribution is bound between 0 and 1 so it's a good choice for estimating probabilities.

We prefer a beta prior over a uniform prior because, given binomial observations, the posterior will also be a beta distribution.

It works out nicely mathematically:

$$
\begin{aligned}
p &\sim \text{Beta}(\alpha, \beta) \\
n &\sim \text{Binomial}(N, p) \\
p \, | \, n, N &\sim \text{Beta}(\alpha + n, \beta + N - n)
\end{aligned}
$$

So with these two distributions, we can directly compute the posterior with no need for simulation (e.g. MCMC).

### Example

We run 100 trials and observe 10 successes. What is the probability $p$ of a successful trial?

Our knowns are $N=100, n=10$. A binomial distribution describes these observations, but we have the unknown parameter $p$.

For our prior for $p$ we choose $\text{Beta}(1,1)$ since it is equivalent to a uniform prior over $[0,1]$ (i.e. it is an objective prior).

We can directly compute the posterior now:

$$
\begin{aligned}
p \, | \, n, N &\sim \text{Beta}(\alpha + n, \beta + N - n) \\
p &\sim \text{Beta}(11, 91)
\end{aligned}
$$

Then we can draw samples from the distribution and compute its mean or other descriptive statistics such as the credible interval.

## Bayesian Inference process

A good approach to Bayesian inference is to:

1. Try and simulate/reconstruct the observed data
2. Identify your unknowns and replace them with random variables
3. Run MCMC to approximate your posteriors

## More on MCMC

MCMC is useful because often we may encounter distributions which aren't easily expressed mathematically (e.g. their functions may have very strange shapes), but we still want to compute some descriptive statistics (or make other computations) from them. MCMC allows us to work with such distributions without needing precise mathematical formulations of them.

More generally, MCMC is really useful if you don't want to (or can't) find the underlying function describing something. As long as you can simulate that process in some way, you don't need to know the exact function - you can just generate enough sample data to work with in its stead. So MCMC is a brute force but effective method.

## Sensitivity Analysis

The strength of the prior affects the posterior - the stronger your prior beliefs, the more difficult it is to change those beliefs (it requires more data/evidence). You can conduct _sensitivity analysis_ to try your approach with various different priors to get an idea of how different priors affect your resulting posterior.

## Conjugate priors

Conjugate priors are priors which, when combined with the likelihood, result in a posterior which is in the same family. These are very convenient because the posterior can be calculated analytically, so there is no need to use approximation such as MCMC.

For example, a binomial likelihood is a conjugate with a beta prior - their combination results in a beta-binomial posterior.

For example, the Gaussian family of distributions are conjugate to itself (_self conjugate_) - a Gaussian likelihood with a Gaussian prior results in a Gaussian posterior.

For example, when working with count data you will probably use the Poisson distribution for your likelihood, which is conjugate with gamma distribution priors, resulting in a gamma posterior.

Unfortunately, conjugate priors only really show up in simple one-dimensional models.

### References

- _Probabilistic Programming and Bayesian Methods for Hackers_, Chapter 6. Cam Davidson Pilon: <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>

## Bayesian Regression

The Bayesian methodology can be applied to regression as well. In conventional regression the parameters are treated as fixed values that we uncover. In Bayesian regression, the parameters are treated as random variables, as they are elsewhere in Bayesian statistics. We define prior distributions for each parameter - in particular, normal priors, so that for each parameter we define a prior mean as well as a covariance matrix for all the parameters.

So we specify:

- $b_0$ - a vector of prior means for the parameters
- $B_0$ - a covariance matrix such that $\sigma^2 B_0$ is the prior covariance matrix of $\beta$
- $v_0 > 0$ - the degrees of freedom for the prior
- $\sigma_0^2 > 0$ - the variance for the prior (which essentially functions as your strength of belief in the prior - the lower the variance, the more concentrated your prior is around the mean, thus the stronger your belief)

So the prior for your parameters then is a normal distribution parameterized by $(b_0, B_0)$.

Then $v_0$ and $\sigma_0^2$ give a prior for $\sigma^2$, which is an inverse gamma distribution parameterized by $(v_0, \sigma_0^2 v_0)$.

Then there are a few formulas:

$$
\begin{aligned}
b_1 &= (B_0^{-1}  + X'X)^{-1}(B_0^{-1}b_0 + X'X\hat \beta) \\
B_1 &= (B_0^{-1} + X'X)^{-1} \\
v_1 &= v_0 + n \\
v_1 \sigma_1^2 &= v_0 \sigma_0^2 + S + r \\
S &= \text{sum of squared errors of the regression} \\
r &= (b_0-\hat \beta)'(B_0 + (X'X)^{-1})^{-1}(b_0 - \hat \beta) \\
f(\beta~|~\sigma^2,y,x) &= \Phi(b_1, \sigma^2 B_1) \\
f(\sigma^2~|~y,x) &= \text{inv.gamma}(\frac{v_1}{2}, \frac{v_1\sigma_1^2}{2}) \\
f(\beta~|~y,x) &= \int f(\beta~|~\sigma^2,y,x) f(\sigma^2~|~y,x)d\sigma^2 = t(b_1, \sigma_1^2B_1,\text{degrees of freedom}=v_1)
\end{aligned}
$$

So the resulting distribution of parameters is a multivariate $t$ distribution.

### References

- POLS 506: Simple Bayesian Models. Justin Esarey. <https://www.youtube.com/watch?v=ps5MYi81IsE>



#### References

- <http://www.stat.tamu.edu/~fliang/STAT605/lect01.pdf>
- <https://www.youtube.com/watch?v=12eZWG0Z5gY>
- <http://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/>
- <http://homepages.dcc.ufmg.br/~assuncao/pgm/aulas2014/mcmc-gibbs-intro.pdf>
- <https://plot.ly/ipython-notebooks/computational-bayesian-analysis/>
- _Think Bayes_, Allen Downey.
- Computational Statistics II. Chris Fonnesbeck. SciPy 2015: <https://www.youtube.com/watch?v=heFaYLKVZY4> and <https://github.com/fonnesbeck/scipy2015_tutorial>
- Bayesian Statistical Analysis. Chris Fonnesbeck. SciPy 2014: <https://github.com/fonnesbeck/scipy2014_tutorial>
- _Probabilistic Programming and Bayesian Methods for Hackers_, Cam Davidson Pilon: <https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers>
- <https://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/>
