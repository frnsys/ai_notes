chaos theory

Sensitivity to initial conditions - a slight change in initial conditions can result in widely different outcomes at a later time.

Chaotic systems are deterministic but may appear random over long enough timeframes.

Consider:

$$
\begin{aligned}
x &\to ax(1-x) \\
a &= 2
\end{aligned}
$$

We'll start and set $x=0.25$. We get $2(0.25)(1-0.25) = 0.375$, then we plug that back in as $x$. We get $2(0.375)(1-0.375) = 0.46875$. We take this result and plug it back in as $x$. We keep repeating this and we converge on $0.5$. If we slightly change our starting value, e.g. $x=0.251$, we still converge on $0.5$.

If we set $a=3.2$ and repeat the above, again starting with $x=0.25$, we don't converge on a number, rather we end up oscillating - the results jump from $x=0.5131$ and $x=0.7995$. If we slightly change our starting value as before, we end up with the same oscillation.

If we set $a=3.83$ and do this again, we end up cycling through a few values. If we slightly change our starting value as before, we end up with the same cycling pattern.

If we set $a=3.9$ and do this again, no pattern appears to emerge. The sequence of values produced here are chaotic. If we slightly change our initial $x$, the resulting sequence will diverge significantly from when it was $x=2.5$.