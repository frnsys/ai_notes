Two main types of graphical models:

- Bayesian models: aka _Bayesian networks_ (_Bayes nets_) or _belief networks_. Used when there are causal relationships between the random variables.
- Markov models: when there are noncausal relationships between the random variables.

## Bayesian Networks

Two nodes (variables) in a Bayes net are on an _active trail_ if a change in one node affects the other. This includes cases where the two nodes have a causal relationship, an evidential relationship, or have some common cause.

