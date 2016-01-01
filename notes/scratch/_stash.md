
# Things that have been edited out but may still be useful elsewhere

---

## Decision Theory

__Decision theory__ is a framework for which the general goal is to _minimize expected loss_. In machine learning, minimizing loss is the typical optimization problem.

Given labeled data points $(x_1, y_1), \dots, (x_n, y_n)$ and some __loss function__ $L(y, \hat y)$, in which $y$ is the true label and $\hat y$ is the predicted label, we want to choose some function $f(x) = \hat y$ (i.e. yields a predicted label or value for a given $x$) that minimizes $L(y, f(x))$.

But because we don't know $x$ or $y$ we want to minimize the average loss, that is, the __expected loss__:

$$
E[L(Y, \hat y) | X = x] = \sum_{y \in Y} L(y, \hat y) P(y|x)
$$

Thus our goal can be stated as:

$$
\hat y = \argmin_y E[L(Y, \hat y) | X = x] = \argmax_y P(y|x)
$$

That is, choose $\hat y$ to minimize the expected loss, which is the same as the $y$ which is maximally probably given $x$.


An example loss function is the _square loss_, which is used in regression:

$$
L(y, \hat y) = (y - \hat y)^2
$$


---


##### The Cauchy-Schwarz Inequality

For two non-zero vectors, $\vec{x}, \vec{y} \in \mathbb R^n$, the following are true:

- The absolute value of their dot product is less than or equal to the product of their
lengths: $|\vec{x} \cdot \vec{y}| \le ||\vec{x}|| \times ||\vec{y}||$
- The absolute value of their dot product will be equal to the product of their lengths
only if they are collinear (i.e. if $\vec{x} = c\vec{y}$): $|\vec{x} \cdot \vec{y}| = ||\vec{x}|| \times ||\vec{y}||$

---


##### Intuition of dot products

The dot product between two vectors tells you how much in the same direction they're
going. So it is minimized when they are orthogonal, and maximized when they are
collinear.

---

#### Intuition of cross products

Cross products are kind of the opposite of dot products - the cross product is minimized
when the vectors are collinear and maximized when they are orthogonal.

#### The $sin$ of the vectors' angle

A property of the cross product is:

$$ || \vec{a} \times \vec{b} || = || \vec{a} || || \vec{b} || sin\theta $$

where $\theta$ is the angle between $\vec{a}, \vec{b}$.

---


## How likely is something to happen?

If something has a 1-in-$m$ chance of happening, does its likelihood increase the more trials there are? That is, how likely is it happen to at least once over $n$ trials?

You certainly aren't guaranteed that it _will_ happen in $m$ trials, although the phrase "1-in-$m$" makes it sound that way. Think about flipping a coin - there is a 1-in-2 chance it will be heads, but you have seen that it is possible to get two heads in a row or two tails in a row.

A better question to ask is "How likely is this to _never_ happen"? Then we can calculate $P(\text{never})$ and calculate $P(\text{atLeastOnce}) = 1 - P(\text{never})$.

With the coin example, let's say we want to know how likely a heads flip will be. $P(\text{heads}) = 1/2$ so $P(\text{heads}') = 1 - P(\text{heads})$ ^[Note that $P(X')$ is the probability of $X$ _not_ happening, also denoted $P(\bar X)$.].

So say we have two trials ($n = 2$). What is the probability that we _never_ get a heads? It's just:

$$ (1 - 1/2)(1 - 1/2) = (1 - 1/2)^2 = 0.25 $$

We can generalize this to:

$$ P(\text{never}) = (1 - \frac{1}{m})^n $$

As $n$ grows larger, $P(never)$ grows smaller - but it never becomes 0. In fact, it approaches 0.37^[This happens to be $1/e$ (one over Euler's number).] as depicted in the accompanying figure.

![$P(\text{never})$](assets/pnever.svg)

So the probability of the event happening at least once in these 1-in-$m$ scenarios is around 0.63.

- <http://www.countbayesie.com/blog/2015/2/18/one-in-a-million-and-e>


## Scale-free networks

A network is be scale-free if its distribution of degrees is a scale-free distribution.

---

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



