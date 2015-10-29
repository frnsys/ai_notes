$$
\providecommand{\argmax}{\operatorname*{argmax}}
$$

$n$-armed bandit problem: there are $n$ slot machines (a slot machine is sometimes called a "one-armed bandit"), that each payout (reward, a random variable) $R_{i,k}$ for a play/turn $k$ for machine $i$, but each machine has a different payout probability distribution (in the _stationary_ version of the problem, which is what we consider here, these payout probabilities are fixed). How do we play these machines in order to maximize our reward?

We have two goals here:

- _explore_ the machines by trying them out to get a sense of which ones have good payouts
- _exploit_ the machine(s) with the best payouts

A simple algorithm is to keep track of the mean payout from each machine, and play the machine with the highest mean payout. That is, for each slot machine $i$, keep track of its mean payout $Q_i$ up to the $k$th play for this machine ($k_i$):

$$
Q_{i,k_i} = \frac{R_{i,1} + R_{i,2} + \dots + R_{i,k_i}}{k_i}
$$

Though this is more efficiently implemented as a running average:

$$
Q_{i,k_i + 1} = Q_{i,k_i} + \frac{1}{k_i} [R_{i,k_i} - Q_{i,k_i}]
$$

Where $R_{i,k_i}$ is the reward received on the current play from slot machine $i$.

Then play the machine $i$:

$$
\argmax_i Q_i
$$

However, with this approach we fail to explore the other machines - so we include an additional parameter $\epsilon$, which is the probability that we pick a random machine to try for a turn, instead of picking the max-average payout machine. This is the simple greedy approach.

For the non-stationary problem, we weigh the outcomes of more recent plays when computing the means.

A variant on this approach is _softmax selection_. With the previous approach, when exploring with probability $\epsilon$, we uniformly randomly chose a machine. But it can be better to combine exploration and exploitation by picking machines by their probability of good payouts. We use softmax to get a probability distribution across the machines.

The softmax equation is:

$$
\frac{e^{Q_{i,k_i}/\tau}}{\sum_{j=1}^n e^{Q_{j,k_j}/\tau}}
$$

Where $\tau$ is the _temperature_ parameter which scales the probability distribution of the choices. High temperatures cause probabilities to be more similar, low temperatures cause them to be more different. $\tau$ is selected more or less by educated guess.

Then each turn we randomly choose a machine according to this computed distribution.

## References

- <http://outlace.com/Reinforcement-Learning-Part-1/>