For a random variable $X$ following distribution $p$, the _information_ of a sample $x$ is:

$$
I(x) = \log_2 \frac{1}{p(x)} = - \log_2 p(x)
$$

The _entropy_ of the random variable $X$ is the expected information of the random variable, i.e.:

$$
H(X) = E_X [I(X)] = -\sum_x p(x) \log_2 p(x)
$$

It can also be thought of the amount of uncertainty or surprise of the random variable. A uniform distribution, where every outcome is equally likely to occur, has high entropy.


The $log_2$ (i.e. binary bits) can be thought of like asking "yes/no" questions.