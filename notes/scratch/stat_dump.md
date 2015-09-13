
$$
\DeclareMathOperator{\SST}{SST}
\DeclareMathOperator{\SSW}{SSW}
\DeclareMathOperator{\SSB}{SSB}
$$

---

## Sum of Squares

### The sum of squares within (SSW)

$$ \SSW = \sum_{i=1}^m (\sum{j=1}^n (x_{ij} - \bar{x_i})^2) $$

- This shows how much of SST is due to variation _within_ each group, i.e. variation from within that group's mean.
- The degrees of freedom here is calculated $m(n-1)$.

### The sum of squares between (SSB)

$$ \SSB = \sum_{i=1}^m [n_m [(\bar{x_m} - \bar{\bar{x}})^2]] $$

- This shows how much of SST is due to variation between the group means
- The degrees of freedom here is calculated $m-1$.

### The total sum of squares (SST)

$$ \SST = \sum_{i=1}^m (\sum{j=1}^n (x_{ij} - \bar{\bar{x}})^2) $$
$$ \SST = \SSW + \SSB $$

- Note: $\bar{\bar{x}}$ is the mean of means, or the "grand mean".
- This is the total variation for the groups
- The degrees of freedom here is calculated $mn - 1$.



