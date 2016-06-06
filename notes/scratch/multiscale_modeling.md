notes from [Coarse graining and multiscale techniques](http://www.engr.ucsb.edu/~shell/che210d/Multiscale.pdf). M. Scott Shell.

The world can be modeled across multiple scales. Running a simulation of, say, a society is not computationally feasible if working from first principles - e.g. you can't take a simulation of individual atomic interactions and scale it out to simulate a full human society. Rather, you'd just model the latter with only as much detail as is needed to capture the interesting phenomena at that level.

Dealing with these kinds of scenarios in which we are interested in phenomena that happen at various scales is a problem of _multiscale modeling_. Models for the higher-levels of the systems we are interested in are called _coarse-grained_ models.

Broadly, there are two methods for building these multiscale models:

- _bottom-up_, in which lower-level simulations are used to parameterize higher-level models
- _top-down_, in which higher-level simulations are used to parameterize lower-level models

In the bottom-up case, you could group elements at the lower-level into abstracted elements at the higher-level, e.g. at a lower-level look at the individuals in a neighborhood, at a higher-level consider the neighborhood as a single entity.

Multiscale modeling won't work if the system is very sensitive to small changes.


Multiscale simulation: like if you're zoomed into a city block, you want to see all the detail there, once you zoom super far out, you don't need to see the individuals anymore.