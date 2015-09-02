
## Hidden Markov Models (HMM)

Hidden Markov Models are Bayes nets in which each state $S_i$ depends only on the previous state $S_{i-1}$ (so the sequence of states is a Markov chain) and emits an observation $z_i$. It is hidden because we do not observe the states directly, but only these observations. The actual observations are stochastic (e.g. an underlying state may produce one of many observations with some probability). We try to infer the state based on these observations.

![Hidden Markov Model](assets/hmm.svg)

The parameters of a HMM are:

- The initial state distribution, $P(S_0)$
- The transition distribution, $P(S_2|S_1)$
- The measurement/observation distribution, $P(z_1|S_1)$

### Example

Say we have the following HMM:

![Hidden Markov Model Example](assets/hmm_example.svg)

We don't know the starting state, but we know the probabilities:

$$
\begin{aligned}
P(R_0) &= \frac{1}{2} \\
P(S_0) &= \frac{1}{2}
\end{aligned}
$$

Say on the first day we see that this person is happy and we want to know whether or not it is raining. That is:

$$
P(R_1|H_1)
$$

We can use Bayes' rule to compute this posterior:

$$
P(R_1|H_1) = \frac{P(H_1|R_1)P(R_1)}{P(H_1)}
$$

We can compute these values by hand:

$$
\begin{aligned}
P(R_1) &= P(R_1|R_0)P(R_0) + P(R_1|S_0)P(S_0) \\
P(H_1) & = P(H_1|R_1)P(R_1) + P(H_1|S_1)P(S_1)
\end{aligned}
$$

$P(H_1|R_1)$ can be pulled directly from the graph.

Then you can just run the numbers.
