> The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. A message saying "the sun rose this morning" is so uninformative as to be unnecessary to send, but a message saying "there was a solar eclipse this morning" is very informative. (p56)


The __self-information__ of an event is:

$$
I(X) = -\ln P(X)
$$

and, when using natural log, is measured in _nats_ (when using $\log_2$, it is measured in _bits_ or _shannons_). One nat is the information gained by observing an event of probability $\frac{1}{e}$.

Self-information only measures a single event; to measure the amount of uncertainty in a complete probability distribution we can instead use the __Shannon entropy__, which tells us the expected information of an event drawn from that distribution:

$$
H(X) = E_{X \sim P}[I(X)] = -E_{X \sim P}[\ln P(X)]
$$

When $X$ is continuous, Shannon entropy is called the __differential entropy__.

### Kullback-Leibler (KL) divergence

We can measure the difference between two probability distributions $P(X), Q(X)$ over the same random variable $X$ with the KL divergence:

$$
D_{\text{KL}}(P||Q) = E_{X \sim P} [\ln \frac{P(X)}{Q(X)}] = E_{X \sim P} [\ln P(X) - \ln Q(X)]
$$

The KL divergence has the following properties:

- It is non-negative
- It is 0 if and only if:
    - $P$ and $Q$ are the same distribution (for discrete variables)
    - $P$ and $Q$ are equal "almost everywhere" (for continuous variables)
- It is _not_ symmetric, i.e. $D_{\text{KL}}(P||Q) \neq D_{\text{KL}}(Q||P)$, so it is not a true distance metric

The KL divergence is related to __cross entropy__ $H(P,Q)$:

$$
\begin{aligned}
H(P,Q) &= H(P) + D_{\text{KL}}(P||Q) \\
&= E_{X \sim P} \log Q(X)
\end{aligned}
$$

## References

- Deep Learning. Yoshua Bengio, Ian Goodfellow, Aaron Courville. <http://www-labs.iro.umontreal.ca/~bengioy/dlbook/>
