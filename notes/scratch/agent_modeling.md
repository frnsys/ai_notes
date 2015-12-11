## Brownian agents

A _Brownian agent_ is described by a set of state variables $u_i^{(k)}$ where $i \in [1, \dots, N]$ refers to the individual agent $i$ and $k$ indicates the different variables.

These state variables may be _external_, which are observable from outside the agent, or _internal degrees of freedom_ that must be inferred from observable actions.

The state variables can change over time due to the environment or internal dynamics. We can generally express the dynamics of the state variables as follows:

$$
\frac{d u_i^{(k)}}{dt} = f_i^{(k)} + \mathcal F_i^{\text{stoch}}
$$

The _principle of causality_ is represented here: any _effect_ such as a temporal change of variable $u$ has some _causes_ on the right-hand side of the equation; such causes are described as a _superposition_ of deterministic and stochastic influences imposed on the agent $i$.

In this formulation, $f_i^{(k)}$ is a deterministic term representing influences that can be specified on the time and length scale of the agent, whereas $F_i^{\text{stoch}}$ is a stochastic term which represents influences that exist, but not observable on the time and length scale of the agent.

The deterministic term $f_i^{(k)}$ captures all specified influences that cause changes to the state variable $u_i^{(k)}$, including interactions with other agents $j \in N$, so it could be a function of the state variables of other agents in addition to external conditions.


## References

- An Agent-Based Model of Collective Emotions in Online Communities. Frank Schweitzer, David Garcia. Swiss Federal Institute of Technology Zurich. 2008.