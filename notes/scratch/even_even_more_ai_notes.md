### Filtering/monitoring cont'd

Kalman filters make the approximation that everything is Gaussian (i.e. transmissions and emissions).

#### Inference base cases in an HMM

The first base case: consider the start of an HMM:

$$
P(X_1) \to P(E_1|X_1)
$$

Inferring $P(X_1|e_1)$, that is, $P(X_1)$ given we observe a piece of evidence $e_1$, is straightforward:

$$
\begin{aligned}
P(x_1|e_1) &= \frac{P(x_1, e_1)}{P(e_1)} \\
&= \frac{P(e_1|x_1)P(x_1)}{P(e_1)} \\
&\varpropto_{X_1} P(e_1|x_1)P(x_1)
\end{aligned}
$$

That is, we applied the definition of conditional probability and then expanded the numerator with the product rule.

For an HMM, $P(E_1|X_1)$ and $P(X_1)$ are specified, so we have the information needed to compute this. We just compute $P(e_1|X_1)P(X_1)$ and normalize the resulting vector.

The second base case:

Say we want to infer $P(X_2)$, and we just have the HMM:

$$
X_1 \to X_2
$$

That is, rather than observing evidence, time moves forward one step.

For an HMM, $P(X_1)$ and $P(X_2|X_1)$ are specified.

So we can compute $P(X_2)$ like so:

$$
\begin{aligned}
P(x_2) &= \sum_{x_1} P(x_2, x_1) \\
&= \sum_{x_1} P(x_2|x_1)P(x_1)
\end{aligned}
$$

From these two base cases we can do all that we need with HMMs.

##### Passage of time

Assume that we have the current belief $P(X|\text{evidence to date})$:

$$
B(X_t) = P(X_t|e_{1:t})
$$

After one time step passes, we have:

$$
P(X_{t+1}|e_{1:t}) = \sum_{x_t} P(X_{t+1}|x_t) P(x_t|e_{1:t})
$$

Which can be written compactly as:

$$
B'(X_{t+1}) = \sum_{x_t} P(X'|x)B(x_t)
$$

Intuitively, what is happening here is: we look at each place we could have been, $x_t$, consider how likely it was that we were there to begin with, $B(x_t)$, and multiply it by the probability of getting to $X'$ had you been there.


##### Observing evidence

Assume that we have the current belief $P(X|\text{previous evidence})$:

$$
B'(X_{t+1}) = P(X_{t+1}|e_{1:t})
$$

Then:

$$
P(X_{t+1}|e_{1:t+1}) \varpropto P(e_{t+1}|X_{t+1}) P(X_{t+1}|e_{1:t})
$$

See the above base case for observing evidence - this is just that, and remember, renormalize afterwards.

Another way of putting this:

$$
B(X_{t+1}) \varpropto P(e|X) B'(X_{t+1})
$$

#### The Forward Algorithm

Now we can consider the forward algorithm (the one presented previously was a simplification).

We are given evidence at each time and want to know:

$$
B_t(X) = P(X_t|e_{1:t})
$$

We can derive the following updates:

$$
\begin{aligned}
P(x_t|e_{1:t}) &\varpropto_X P(x_t, e{1:t}) \\
&= \sum_{x_{t-1}} P(x_{t-1}, x_t, e_{1:t}) \\
&= \sum_{x_{t-1}} P(x_{t-1}, e_{1:t-1}) P(x_t|x_{t-1})P(e_t|x_t) \\
&= P(e_t|x_t) \sum_{x_{t-1}} P(x_{t-1}, e_{1:t-1})
\end{aligned}
$$

Which we can normalize at each step (if we want $P(x|e)$ at each time step) or all together at the end.

This is just variable elimination with the order $X_1, X_2, \dots$.

This computation is proportional to the square number of states.

### Particle Filtering

Sometimes we have state spaces which are too large for exact inference (i.e. too large for the forward algorithm) or just to hold in memory. For example, if the state space $X$ is continuous.

Instead, we can use _particle filtering_, which provides an approximate solution.

With particle filtering, we keep track of samples ("particles") of $X$ - not all its values.

The time per step for particle filtering is linear in the number of samples, but we may need a very large number of samples.

Each particle can be thought of as a hypothesis that we are in the state it represents. The more particles there are for a state mean the more likely we are in that state.

More formally, our representation of $P(X)$ is now a list of $N$ particles (generally $N << X$, and we don't need to store $X$ in memory anymore, just the particles).

$P(x)$ is approximated by the number of particles with value $x$ (i.e. the more particles that have value $x$, the more likely state $x$ is).

Particles have weights, and they all start with a weight of 1.

As time passes, we "move" each particle by sampling its next position from the transition model:

$$
x' = \text{sample}(P(X'|x))
$$

As we gain evidence, we fix the evidence and downweight samples based on the evidence:

$$
\begin{aligned}
w(x) &= P(e|x) \\
B(X) \varpropto P(e|x)B'(X)
\end{aligned}
$$

These particle weights reflect how likely the evidence is from that particle's state. A result of this is that the probabilities don't sum to one anymore.

This is similar to likelihood weighting.

Rather than tracking the weighted samples, we resample.

That is, we sample $N$ times from the weighted sample distribution (i.e. we draw with replacement). This is essentially renormalizing the distribution and has the effect of "moving" low-weight (unlikely) samples to where high-weight samples are (i.e. to likely states), so they become more "useful".


### Dynamic Bayes' Nets (DBN)

A dynamic Bayes' net is a Bayes' net replicated through time, i.e. variables at time $t$ can be conditioned on those from time $t-1$ (the structure is reminiscent of an RNN).

An HMM is a simple dynamic Bayes' net.

There are also DBN particle filters in which each particle represents a full assignment to the world (i.e. a full assignment of all variables in the Bayes' net). Then at each time step, we sample a successor for each particle.

When we observe evidence, we weight each _entire_ sample by the likelihood of the evidence conditioned on the sample.

Then we resample - select prior samples in proportion to their likelihood.

Basically, a DBN particle filter is a particle filter where each particle represents multiple assigned variables rather than just one.

### Most Likely Explanation (MLE)

With Most Likely Explanation, the concern is not the state at time $t$, but the most likely sequence of states that led to time $t$, given observations.

For MLE, we use an HMM and instead we want to know:

$$
\argmax_{x_{1:t}} P(x_{1:t}|e_{1:t})
$$

We can use the _Viterbi algorithm_ to solve this, which is essentially just the forward algorithm where the $\sum$ is changed to a $\max$:

$$
\begin{aligned}
m_t[x_t] &= \max_{x_{1:t-1}} P(x_{1:t-1}, x_t, e_{1:t}) \\
&= P(e_t|x_t) \max_{x_{t-1}} P(x_t|x_{t-1}) m_{t-1}[x_{t-1}]
\end{aligned}
$$

In contrast, the forward algorithm:

$$
\begin{aligned}
f_t[x_t] &= P(x_t, e_{1:t}) \\
&= P(e_t|x_t) \sum_{x_{t-1}} P(x_t|x_{t-1}) f_{t-1}[x_{t-1}]
\end{aligned}
$$

---

Machine learning


## Naive Bayes

The main assumption of Naive Bayes is that all features are independent effects of the label. This is a really strong simplifying assumption but nevertheless in many cases Naive Bayes performs well.

Naive Bayes is also _statistically efficient_ which means that it doesn't need a whole lot of data to learn what it needs to learn.

If we were to draw it out as a Bayes' net:

$$
\begin{aligned}
Y &\to F_1 \\
Y &\to F_2 \\
&\dots \\
Y &\to F_n
\end{aligned}
$$

Where $Y$ is the label and $F_1, F_2, \dots, F_n$ are the features.

The model is simply:

$$
P(Y|F_1, \dots, F_n) \varpropto P(Y) \prod_i P(F_i|Y)
$$

This just comes from the Bayes' net described above.

The Naive Bayes learns $P(Y, f_1, f_2, \dots, f_n)$ which we can normalize (divide by $P(f_1, \dots, f_n)$) to get the conditional probability $P(Y|f_1, \dots, f_n)$:

$$
P(Y, f_1, \dots, f_n) =
\begin{matrix}
P(y_1, f_1, \dots, f_n) \\
P(y_2, f_1, \dots, f_n) \\
\vdots \\
P(y_k, f_1, \dots, f_n)
\end{matrix} =
\begin{matrix}
P(y_1) \prod_i P(f_i|y_1) \\
P(y_2) \prod_i P(f_i|y_2) \\
\vdots \\
P(y_k) \prod_i P(f_i|y_k)
\end{matrix}
$$

So the parameters of Naive Bayes are $P(Y)$ and $P(F_i|Y)$ for each feature.

## Apprenticeship

We can use machine learning to learn how to navigate a state space (i.e. to plan) by "watching" another agent (an "expert") perform. For example, if we are talking about a game, the learning agent can watch a skilled player play, and based on that learn how to play on its own.

In this case, the examples are states $s$, the candidates are pairs $(s,a)$, and the "correct" actions are those taken by the exprt.

We define features over $(s,a)$ pairs: $f(s,a)$.

The score of a q-state $(s,a)$ is given by $w \cdot f(s,a)$.

This is basically classification, where the inputs are states and the labels are actions.

## Case-Based Reasoning

Case-based approaches to machine learning involve predicting an input's label based on similar instances. The key issue here is how you define similarity.

For example, _nearest-neighbor classification_. $K$-NN involves voting on the label of the $k$ nearest neighbors (it requires a weighting scheme) to the input. The similarity could just be the dot product of the two vectors, i.e. $\text{sim}(x, x') = x \cdot x'$. Normally the vectors are normalized, i.e. such that $||x|| = 1$.

These are non-parametric models. There is not a fixed set of parameters (which isn't to say that there are _no_ parameters, though the name "non-parametric" would have you think otherwise). Rather, the complexity of the classifier increases with the data.

Non-parametric models are typically better in the limit, but worse in the non-limit.

### Similarity functions

Many similarities are based on feature dot products:

$$
\text{sim}(x, x') = x \cdot x' = \sum_i x_i x_i'
$$

## Starcraft

Starcraft is hard for AI because:

- adversarial
- long horizon
- partially observable (fog-of-war)
- realtime (i.e. 24fps, one action per frame)
- huge branching factor
- concurrent (i.e. players move simultaneously)
- resource-rich

There is no single algorithm (e.g. minimax) that will solve it off-the-shelf.

The Berkeley Overmind won AIIDE 2010 (a Starcraft AI competition). It used:

- search: for path planning for troops
- CSPs: for base layout (i.e. buildings/facilities)
- minimax: for targeting of opponent's troops and facilities
- reinforcement learning (potential fields): for micro control (i.e. troop control)
- inference: for tracking opponent's units
- scheduling: for managing/prioritizing resources
- hierarchical control: high-level to low-level plans



## Misc notes

If the true function is in your hypothesis space $H$, we say it is _realizable_ in $H$.


## Entropy and information

Information, measured in bits, answers questions - the more initial uncertainty there is about the answer, the more information the answer contains.

The amount of bits needed to encode an answer depends on the distribution over the possible answers (i.e., the uncertainty about the answer).

Examples:

- the answer to a boolean question with a prior $(0.5, 0.5)$ requires 1 bit to encode (i.e. just 0 or 1)
- the answer to a 4-way question with a prior $(0.25, 0.25, 0.25, 0.25)$ requires 2 bits to encode
- the answer to a 4-way question with a prior $(0, 0, 0, 1)$ requires 0 bits to encode, since the answer is already known (no uncertainty)
- the answer to a 3-way question with prior $(0.5, 0.25, 0.25)$ requires, on average, 1.5 bits to encode

More formally, we can compute the average number of bits required to encode uncertain information as follows:

$$
\sum_i p_i \log_2 \frac{1}{p_i}
$$

This quantity is called the _entropy_ of the distribution ($H$), and is sometimes written as the equivalent:

$$
H(p_1, \dots, p_n) = \sum_i -p_i \log_2 p_i
$$

If you do something such that the answer distribution changes (e.g. observe new evidence), the difference between the entropy of the new distribution and the entropy of the old distribution is called the _information gain_.