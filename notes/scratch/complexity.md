Two approaches to science:

- mathematical, defined by equations, using proofs ("classical" models).
  - typically deterministic
  - generally involve many simplifying assumptions
  - uses linear approximations to model non-linear systems
- computational-based, typically defined by simple rules, using simulations ("complex" models)
  - often stochastic
  - often also involve simplifying assumptions, but less
  - deals better with non-linear systems

(Chapter 1 of _Think Complexity_ provides a good overview of these two approaches).

## Pareto distributions

A Pareto distribution has a CDF with the form:

$$
\text{CDF}(x) = 1 - (\frac{x}{x_m})^{-\alpha}
$$

They are characterized as having a long tail (i.e. many small values, few large ones), but the large values are large enough that they still make up a disproportionate share of the total (e.g. the large values take up 80% of the distribution, the rest are 20%).

Such a distribution is described as _scale-free_ since they are not centered around any particular value. Compare this to Gaussian distributions which are centered around some value.

Such a distribution is said to obey the _power law_. A distribution $P(k)$ obeys the power law if, as $k$ gets large, $P(k)$ is asymptotic to $k^{-\gamma}$, where $\gamma$ is a parameter that describes the rate of decay.

Such distributions are (confusingly) sometimes called _scaling distributions_ because they are invariant to changes of scale, which is to say that you can change the units the quantities are expressed in and $\gamma$ doesn't change.

The _complimentary distribution_ (CCDF) of a distribution is $1 - \text{CDF}(x)$.

## Scale-free networks

A network is be scale-free if its distribution of degrees is a scale-free distribution.

## Agent-based models

Agent-based models include:

- individual agents model intelligent behavior, usually with a simple set of rules
- the agents are situated in some space or a network and interact with each other locally
- the agents usually have imperfect, local information
- there is usually variability between agents
- often there are random elements, either among the agents or in the world

## References

- Think Complexity (Version 1.2.3). Allen B. Downey. 2012.