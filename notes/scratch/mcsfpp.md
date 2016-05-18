_Modeling Complex Systems for Public Policies_. Edited by Bernardo Alves Furtado, Patrícia A. M. Sakowski, Marina H. Tóvolli. 2015.

- complex systems includes network analysis, information theory, cellular automata, and agent-based models.
- CAs vs ABMs: CAs are fixed in space (i.e. for focusing on local, physically bounded interactions)
- "the whole is more than the sum of the parts"; complex systems cannot be reduced to its individual components; the interaction between all the parts leads to emergent phenomena


---

## Chapter 2: Complex Systems: Concepts, Literature, Possibilities, and Limitations. William Rand.

Two main phenomena of complex systems are:

- _emergence_: the whole is more than the mere sum of its parts; it cannot be described or reduced to the behavior or properties of its components
- _feedback_: the emergent properties and behaviors affect the individual components which in turn affects the emergent properties and behaviors and so on

_Leverage points_ are points within the complex system that can, when changed, affect the system itself.

_Tipping points_ (also called _phase transitions_ or _bifurcations_) are points where the system suddenly changes drastically due to a comparatively small adjustment.

Complex systems may exhibit _path dependence_, where (distantly) past events affect the possibilities of present state. _Sensitivity to initial conditions_ is a characteristic form of path dependence in complex systems, where slight changes in the system's starting point can lead to drastically different outcomes.

Complex systems are _nonlinear_; that is, the outputs are not linearly related to the inputs. The aforementioned properties describe this nonlinearity - changes in the input are not proportional to changes in the output.

The _robustness_ of a complex system is its ability to withstand wholesale removal of or large changes to its components without significant differences in its outcomes. Complex systems may also _adapt_ and _evolve_.

An important critique of modeling is _the Lucas critique of policy_ (1976), which states that models (in terms of understanding policy) are focused on macro-level behaviors, but because low-level individual behaviors will shift in response to policy, the predictions such models make will be wrong. Complex systems modeling, because it focuses on the low-level individual, are more resistant to this critique.

Complex systems modeling also works well with diversity and heterogeneity, which are often left out of models because they complicate things, even though they may be crucial to a good model. They also model networks well, and the interconnectedness and interactions that come with them.

A common tool for modeling complex systems is _agent-based modeling_ (ABM). The advantage of ABMs is that they are relatively easy to describe - they aren't as abstract as purely mathematical models.

The power of complex systems modeling comes at a cost - they may be computationally expensive and contain large numbers of free parameters. They also require theoretical expertise around what is modeled.

---

## Chapter 4: Simulation Models for Public Policy. James E. Gentile, Chris Glazner, Matthew Koehler.

> Given the complexity of even the smallest of social systems, [the analysis of policy outcomes] is not trivial. Social systems are comprised of autonomous people who do not behave in perfectly rational ways, and they have different explanatory mental models for how society works. Social systems do not behave in deterministic ways that lend themselves to a simple spreadsheet analysis or a closed form mathematical formulation at the causal level. The behavior of social systems cannot be neatly constructed, as a watchmaker would build a watch to keep time. (p. 73)

> The relationship between a cause and its effect can be understood through models. At its most basic form, a model can simply be a mental concept, a description of a belief for how a system will respond to change. (p. 73)

> ...one might be tempted to argue that "big data" are the solution. One could simply analyze enough data from a system to understand all of its potential dynamics to include outliers. However, from a policy perspective this analytic approach is of limited utility. What a big data analysis can provide is the correlative structures present within a dataset. This is quite different than the causal structure. Moreover, policy analysis is typically undertaken to inform a desired change to the system. This being the case, the potential new system would be "out of sample" from the big data analysis and how the old and new systems relate may not be clear.
>
> ABMs, on the other hand, allow one to investigate potential generating mechanisms and experiment with causal structures. As Epstein has termed it: "If we did not generate $x$, we did not explain $x$" (Epstein, 2006). As pointed out by Axtell, growing a particular outcome only demonstrates sufficiency (Axtell, 2000). One can demonstrate what will cause an outcome but, likely, will not be able to prove that is the actual mechanism being used by the system under study. (pp. 76-77)

With ABMs, the model must be run many times to explore the mapping between the inputs and outputs. Outputs can be broadly categorized as so:

- _expected valid_ - by using input values known to result in known proper behavior
- _expected invalid_ - by using input values known to result in degenerate behavior
- _unexpected results_

_Axtell's Levels of Epmirical Relevance_ (2005) give a rough categorization of ABMs' specificity.

- Level 0: "micro-level qualitative correspondence-agents that behave plausibly for a given system"
- Level 1: "macro-level qualitative correspondence to the referent"
- Level 2: "macro-level quantitative correspondence"
- Level 3: "micro-level quantitative correspondence-agents that behave identically to their 'real world counterparts'"

Levels 0 and 1 are more appropriate for "thought experiments and initial investigations"; for more serious applications, levels 2 and 3 should be used.

Axtell also describes three levels of correspondence between the ABM and the referent system:

- _identity_ - the simulation produces identical results to the referent
- _distributional_ - the simulation produces results that are statistically indistinguishable from the referent
- _relational_ - the simulation produces results that are statistically distinguishable but qualitatively similar to the referent

The process of developing such simulations typically involves working with domain experts, implementing a model, verifying the model's implementation (e.g. via unit testing), and then validating the model's outputs against external data sources (if available). It is extremely important to carefully document the modeling process and its implementation so that it may be reproduced. This is especially true because these models can get very complicated very quickly.

---

## Chapter 5: Operationalizing Complex Systems. Jaime Simão Sichman.

_Multi-agent systems_ (formerly called "distributed artificial intelligence", or DAI), which form the basis of the _multi-agent-based simulation_ (MABS) simulation technique.

Another property of complex systems it that they are _open_ - new individuals or entities may come and go from the system. There is also the property of _2nd order emergence_, in which the global patterns that emerge from individual behavior may persist after their originating individuals have left the system, and remaining individuals perceive and respond to those effects (perhaps in different ways), generating more emergent effects.

Multi-agent systems are essentially societies of autonomous artificial agents. These agents may have varying complexity. The simplest are _reactive_ agents, and more complex ones are _deliberative_ agents.

These systems can be quite difficult to develop - more of an art than a science. The typical process involves real-world data collection, a development of an initial simulation model and parameters/exogenous factors (which ideally are based on real values, but such values may not be available), verification (e.g. unit testing) that the implementation works as expected, running of the simulation, and validation of the simulation's results with other data collected (various statistical methods, such as R2 and mean absolute error, can be used for validation). Determining what data is relevant to the agents' internal decision making process and how that works is tricky and nuanced. Validation may also be difficult, because of path dependence, or because of stochastic elements of the simulation, and so on.

After validation, _sensitivity analysis_ is necessary to determine how sensitive the simulation is to the initial assumptions that were made. This process involves slightly changing the initial conditions and parameters and rerunning the simulation. The easiest way to do this is to vary these parameters randomly so that a distribution over outcomes is generated.

---

## Chapter 7: The Complex Nature of Social Systems. Claudio J. Tessone

> A model is an abstract, and to some extent idealised, description of reality that still captures a specific phenomenon. It is therefore limited by construction. This is true in particular for the complex systems approach to social systems. Models in this realm are not intended to reproduce _society_ as a whole, but to shed light on mechanisms behind social phenomena. (pp. 141-142)

Social systems in particular are characterized by heterogeneity, unlike other systems such as molecular or biological ones. Complex behavior arises out of the interactions between these heterogeneous individuals (as part of networks) as well as interactions with _signals_ which are typically exogenous and environmental (e.g. advertisements, media, etc), though they may be influenced by agent behavior as well.

Social systems are finite, a result of which is _demographic noise_ - an "intrinsic randomness" (p. 143).

