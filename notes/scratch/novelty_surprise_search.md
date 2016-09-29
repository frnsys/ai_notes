# Gravina, D., Liapis, A., & Yannakakis, G. N. (2016). [Surprise search: Beyond objectives and novelty](http://antoniosliapis.com/papers/surprise_search_beyond_objectives_and_novelty.pdf). In Proceedings of the Genetic and Evolutionary Computation Conference. ACM.

The problem with using only a fitness function (i.e. an explicit objective) for evolutionary algorithms: local optima.

In the context of evolutionary algorithms, _deception_ describes when the combination of highly-fit components result in a solution further from the global optimum (towards a local optimum). Deception, in combination with sampling error and the ruggedness of a fitness landscape, are the main contributors to the difficulty of an evolutionary computation problem.

This problem can be mitigated by incorporating a _novelty_ criteria - the dissimilarity of a new solution against existing solutions. This _novelty search_ typically results in better solutions significantly more quickly than the conventional fitness-based approach.

The general approach involves keeping track of previous highly-novel solutions (a "_novel archive_"). The novelty of a solution is the average distance from either neighbors in the novel archive or in the current population. The particular distance measure is problem-dependent.

The paper proposes an additional criteria of _surprise_. Whereas novelty just requires that a solution be sufficiently different from existing ones, surprise also requires deviation from expectations.

One way of understanding the distinction is by viewing surprise as the time derivative of novelty; i.e. novelty is position, surprise is velocity.

To quantify the surprise of a solution, a _surprise archive_ is maintained of the past $h$ generations and a predictive model $m$ is learned to generate expectations. A new solution is compared against the members $k$ (the level of _prediction locality_) groups (if $k=1$, compared to the entire population $P$, if $k=P$, compared to each individual of the population). $k$-mean sis used to form the population groups; each generation (except the first) uses the $k$ centroids from the previous generation. The expectations $p$ is a function of these, i.e. $p = m(h,k)$. Each population group is used to generate a expectation.

The _surprise value_ $s$ of a solution $i$ is computed as the average distance to the $n$-nearest expectations:

$$
s(i) = \frac{1}{n} \sum_{j=0}^n d_s (i, p_{i,j})
$$

where $d_s$ is the domain-dependent measure of behavioral difference between an individual and its expectation and $p_{i,j}$ is the $j$-closest prediction point (expectation) to individual $i$.
