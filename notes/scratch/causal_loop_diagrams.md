notes from _Business Dynamics: Systems Thinking and Modeling for a Complex World_ (John D. Sterman).

# Causal Loop Diagrams (CLD)

![A simple CLD](assets/cld01.svg)

A causal loop diagrams is a way of visualizing relationships between variables in a simulation.

Variables are connected via _causal links_ which may be positive (the tail variable $A$ pressures $B$ to increase) or negative (the tail variable $A$ pressures $B$ to decrease).

![Causal links](assets/cld02.svg)

These links may form loops in the diagram, which may be either positive (i.e. _reinforcing_) or negative (i.e. _balancing_). This is known as the _loop polarity_.

![Loop polarities](assets/cld03.svg)

The loop polarity is related to the _open loop gain_ of the loop ("gain" as in signal strength). The gain is calculated by "breaking open" the loop at some point (hence "open loop").

One shorthand way to identify loop polarity is to count the number of negative links in the loop. If the number of negative links is even, the loop is positive. If odd, the loop is negative (a "net reversal").

The sign of the open loop gain is the loop polarity, so we can identify the loop polarity by computing the open loop gain, as follows.

Say we have a loop of variables $x_1, \dots, x_n$. Assume that $x_1$ is where we break the loop. This splits it into $x_1^I$ (input) and $x_1^O$ (output). The open loop gain is just $\frac{\partial x_1^O}{\partial x_1^I}$, which can be computed using the chain rule, i.e. $\frac{\partial x_1^O}{\partial x_1^I} = \frac{\partial x_1^O}{\partial x_{n-1}} \frac{\partial x_{n-1}}{\partial x_{n-2}} \dots \frac{\partial x_2}{\partal x_1^I}$

If you aren't sure whether a link should be negative or positive, it probably means you're missing a causal variable.

![Missing causal variable](assets/cld04.svg)

You should also indicate time delays if they are significant.

![Include delays](assets/cld05.svg)

You should also distinguish between actual and perceived conditions, i.e. distinguish between "Product Quality" and "Reported Product Quality".

# Stocks and Flows

A flow is a rate of change (i.e. a derivative) in some variable and a stock is a quantity affect by flows.

Other terms for "flow" include "rate", "derivative", "throughput".

Other terms for stock include "integrals", "state variables", "buffers", "levels".

To compute a stock at time $t$ we compute the integral of the in and out flows to the stock:

$$
\text{Stock}(t) = \int_{t_0}^t \[\text{inflow}(s) - \text{outflow}(s)\] ds + \text{Stock}(t_0)
$$

Where $\text{inflow}(s), \text{outflow}(s)$ are the inflow and outflow at time $s$

Equivalently, we can write this as:

$$
\frac{d(\text{Stock})}{dt} = \text{inflow}(t) - \text{outflow}(t)
$$