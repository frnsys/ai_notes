- complexity of joint distributions: for a multinomial distribution with $k$ states for each of its $n$ variables, the full distribution requires $k^n - 1$ parameters
- complexity can be reduced using a CPD graph structure such as a Bayes Net (BN)
- learning parameters of a BN: straightforward, like any CPD, you can use maximum likelihood estimates (MLE) or Bayesian estimates with a Dirichlet prior


## Learning the structure of a BN

Includes local and global components...

### Local: independence tests

#### Measures of deviance-from-independence between variables

For variables $x_i, x_j$ in dataset $\mathcal D$ of $M$ samples...

1. Pearson's Chi-squared ($\chi^2$) statistic:

$$
d_{\chi^2}(\mathcal D) = \sum_{x_i, x_j} \frac{(M[x_i,x_j] - M \cdot \hat P(x_i) \cdot \hat P(x_j))^2}{M \cdot \hat P(x_i) \cdot \hat P(x_j)}
$$

Independence increases as this value approaches 0

2. Mutual information (KL divergence) b/w joint and product of marginals:

$$
d_I(\mathcal D) = \frac{1}{M} \sum_{x_i,x_j} M[x_i, x_j] \log \frac{M[x_i, x_j]}{M[x_i]M[x_j]}
$$

Independence increases as this value approaches 0

### A decision rule for accepting/rejecting hypothesis of independence

Choose some p-value $t$, acept if $d(\mathcal D) <= t$, else reject.

### Global: scoring the structure

For a graph $\mathcal G$ with $n$ variables

1. Log-likelihood score:

$$
\text{score}_L (\mathcal G: \mathcal D) = \sum_{\mathcal D} \sum_{i=1}^n \log \hat P (x_i | \text{parents}(x_i))
$$

2. Bayesian score:

$$
\text{score}_B (\mathcal G: \mathcal D) = \log p(\mathcal D | \mathcal G) + \log p(\mathcal G)
$$

3. Bayes information criterion (with Dirichlet prior over graphs):

$$
\text{score}_{BIC} (\mathcal G: \mathcal D) = l(\hat \theta) : \mathcal D) - \frac{\log M}{2} \text{Dim}(\mathcal G)
$$

### Learning algorithms

- Constraint-based
    - find best structure to explain determined dependencies
    - sensitive to errors in testing individual dependencies
- Score-based
    - search the space of networks to find high-scoring structure
    - requires heuristics (e.g. greedy or branch-and-bound)
- Bayesian model averaging
    - prediction over all structures
    - may not have closed form

## References

- [Big Data, Machine Learning, Causal Models](http://www.cedar.buffalo.edu/~srihari/talks/ICSIP-2014.pdf). Sagur N. Srihari.