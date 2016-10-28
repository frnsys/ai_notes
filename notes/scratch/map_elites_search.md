# Mouret, J. B., & Clune, J. (2015). [Illuminating search spaces by mapping elites](https://arxiv.org/pdf/1504.04909.pdf). arXiv preprint arXiv:1504.04909.

MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) search

Some optimization problems involve search spaces that are non-differentiable or cannot be expressed mathematically. We may have some performance metric to identify high-performing solutions, but we do not have access to the underlying function that determines the performance. In such cases we would use a "black box" optimization algorithm such as evolutionary algorithms.

Note: in evolutionary algorithms, the feature space is sometimes called the "behavior space".

General idea of MAP-Elites: searching in a high (potentially infinite-dimensional) space, user specifies some lower-dimensional feature space; performance of solutions is measured in this lower-dimensional space.

"Illumination algorithm", contrasted with optimization algorithms (but can be used as optimization algorithms):

> designed to return the highest-performing solution at each point in the feature space. They thus illuminate the fitness potential of a each region of the feature space.

Novelty search is an example of an illumination algorithm, as is MAP-Elites.

MAP-Elites algorithm:

- create an empty $N$-dimensional map of elites (solutions $\mathcal X$ and their performances $\mathcal P$)
- for $I$ iterations (with iterator $i$), or until some termination criteria is satisfied
    - if $i < G$, generate random solution $x'$
    - else randomly select an elite $x$ from $\mathcal X$
        - create randomly modified copy of $x$, $x'$ (via mutation/crossover)
    - record $x'$'s feature descriptor $b'$ (mapping it to a cell in the feature space)
    - compute performance $p'$ of $x'$
    - if $\mathcal P(b')$ is empty or worse than $p'$
        - set $\mathcal P(b') = p'$
        - set $\mathcal X(b') = x'$

Example:

- user chooses a performance measure (i.e. fitness function) $f(x)$ that evaluates a solution $x$
- user chooses $N$ dimensions of interest; these define the feature space
    - e.g. with a robot it could be its height, weight, and energy consumption
- the $N$ dimensions are discretized at some granularity (either user-specifier or limited by computational resources). This discretization makes the feature space into a lattice (i.e. it's broken up into cells)
- we have a feature function (aka "behavior function") $b(x)$ that maps a solution $x$ to an $N$-dimensional vector describing its features (i.e. mapping it to the feature space). some of these features may need to be measured through e.g. simulation, such as energy consumption.
- search happens in the search space, where we search through genotypes/genomes $x$ (rather than directly searching phenotypes)