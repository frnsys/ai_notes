notes from John Schulman's Deep Reinforcement Learning lectures for MLSS 2016 in Cadiz.

[Lecture 1](https://www.youtube.com/watch?v=aUrX-rP_ss4)
[Lecture 2](https://www.youtube.com/watch?v=oPGVsoBonLM)
[Lecture 3](https://www.youtube.com/watch?v=rO7Dx8pSJQw)
[Lecture 4](https://www.youtube.com/watch?v=gb5Q2XL5c8A)

---

Broadly, two approaches to RL:

- policy optimization: the policy is parameterized and you try to optimize expected reward
    - includes policy gradients, derivative-free optimization (DFO)/evolutionary algorithms (though DFO doesn't work well for large numbers of parameter)
- dynamic programming: you can exactly solve some simple control problems (i.e. MDPs) with dynamic programming
    - includes policy iteration, value iteration
    - for more useful/realistic problems we have to use approximate versions of these algorithms (e.g. Q-learning)

There are also actor-critic methods which are policy gradient methods that use value functions.

_Deep reinforcement learning_ is just reinforcement learning with nonlinear function approximators, usually updating parameters with stochastic gradient descent.

---

Policies:

- deterministic policies: $a = \pi(s)$
- stochastic policies: $a \sim \pi(a|s)$

Policies may be parameterized, i.e. $\pi_{\theta}$

---

Cross-entropy method (a DFO/evolutionary algorithm, for parameterized policies/policy optimization):

- initialize $\mu \in \mathbb R^d, \sigma \in \mathbb R^d$
- for each iteration
    - collection $n$ samples of $\theta_i \sim N(\mu, \diag(\theta))$ (i.e. sample a population of parameter vectors)
    - perform a noisy evaluation $R_i \sim \theta_i$ (i.e. for each parameter vector, evaluate its reward)
    - select the top $p$ percent of samples (e.g. $p=20$); this is the _elite set_ (the high-fitness individuals)
    - fit a Gaussian distribution, with diagonal covariance, to the elite set, obtaining a new $\mu, \sigma$
- return the final $\mu$