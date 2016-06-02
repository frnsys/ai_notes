notes from David Silver's Reinforcement Learning course [Advanced Topics: RL](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa) ([see also](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)). 2015 (COMPM050/COMPGI13).

Reinforcement learning characteristics:

- no supervisor (nothing top-down saying what's right and what's wrong as in supervised learning), only a reward signal
- feedback (reward) may be delayed (not instantaneous)
    - e.g. giving up short-term gain may be better for long-term gain
- time is important (i.e. sequences of actions)
- the agent's actions affects the subsequent data it receives

A _reward_ $R_t$ is a scalar feedback signal that indicates how well the agent is doing at step $t$. The agent's objective is to maximize cumulative reward.

Note that the reward must be scalar though you could take a reward vector over different aspects (e.g. this increases my income, but decreases my happiness) and take its dot product with some weights (e.g. I value my happiness more than my income) to retrieve a scalar value.

The _reward hypothesis_ states that goals can be described by the maximization of expected cumulative reward and is the central assumption of reinforcement learning.

The term "reward" may be a bit confusing, since they may be negative as well, i.e. "punishment".

Rewards can come at each time step (for example, if you want an agent to do something quickly, you can set a reward of -1 for each time step), or at the end of the task, or at the end of each "episode" if the task has no end (but you need to chunk up time in some way), or for certain states, etc.

The general framework is that the agent decides on and executes some action $A_t$ which affects the environment, receives observation $O_t$ from the environment, and gets a reward $R_t$.

The _history_ is the sequence of observations, actions, and rewards:

$$
H_t = A_1, O_1, R_1, \dots, A_t, O_t, R_t
$$

We are essentially looking to build a mapping from this history to some action to take next.

Practically, we can't work with the full history, so we summarize it as _state_ $S_t = f(H_t)$ (i.e. it is some function of the history) and so the task is to learn a mapping from states to some action to take next.

We can represent a state for the environment as well (distinct from the agent's state $S_t^a$, and typically hidden from the agent) as $S_t^e$. In multi-agent simulations, from the perspective of a single agent, the other agents can be represented as part of the environment.

We can more specifically define an _information state_ (also called a _Markov state_), which is a state in which:

$$
P[S_{t+1}|S_t] = P[S_{t+1}|S_1, \dots, S_t]
$$

That is, we make a Markov assumption that the probability of the next state depends only on the current state. This assumption allows us to effectively ignore the history previous to the present state.


How we represent state greatly affects learning. Consider an experiment in which a rat either gets shocked or gets a piece of cheese. The rat observes the following two sequences:

1. light, light, lever, bell -> shock
2. bell, light, lever, lever -> cheese

Consider this sequence:

- lever, light, lever, bell

Will the rat get shocked or get cheese?

You might think "shock" because the same 3-item sequence is at the end (light, lever, bell). However, if you represent the state as numbers of each event, you'd say cheese (1 bell, 1 light, 2 levers). Or you could represent it as the entire 4-item sequence in which case it's not clear what the result will be.


In a _fully observable_ environment, the agent directly observes environment state, i.e. $O_t = S_t^a = S_t^e$. In this case we have a Markov decision process (MDP).

In a _partially observable_ environment, the agent indirectly observes the environment (a lot of information is hidden). In this case the agent and environment states are not congruent. This is a partially observable Markov decision process (POMDP). In this case we must construct the agent's state representation separately - one possibility is as the beliefs of the environment state, or use a recurrent neural network to generate the next state, etc.

The main components that may be included in an RL agent:

- policy: the agent's behavior function
    - a map from state to action
    - it may be deterministic, i.e. $a = \pi(s)$
    - or it may be stochastic, i.e. $\pi(a|s) = P[A=a|S=s]$
- value function: how good is each state and/or action
    - a prediction of (i.e. expected) future reward
    - evaluates the goodness or badness of states
    - used to select between actions
    - e.g. $v_{\pi}(s) = E_{\pi}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots | S_t = s]$
    - the value function $V$ is the state value function, which values states, and $Q$ is the action value function, which values actions (given states)
- model: agent's representation of the environment
    - often has two components
    - the _transitions_ $P$ predicts the next state (the "dynamics"), e.g. $P_{ss'}^a = P[S'=s'|S=s,A=a]$
    - the _rewards_ $R$ predicts the next immediate reward, e.g. $R_s^a = E[R|S=s,A=a]$
    - but there are many model-free methods as well

RL agents can be categorized according to which of these components they have:

- _value based_ agents
    - have a value function
    - if it has a value function, the policy is implied (choose the highest-valued action)
- _policy based_ agents
    - have an explicit policy
    - without a value function
- _actor critic_ agents
    - has both a policy and a value function

Further distinction is made between model-free agents (the agent doesn't model the environment, just has a policy and/or value function) or model-based agents (has a model and a policy and/or value function).

Prediction vs control:

- prediction: evaluate the future, given a policy
- control: optimize the future, find the best policy

---

# Markov Decision Processes

Markov decision processes formally describe a fully-observable environment for reinforcement learning. Almost all RL problems can be formalized as MDPs, including partially-observable RL problems: partially-observable problems can be converted into MDPs.

We make the Markov assumption (see above) for MDPs. That is, the current state captures all relevant information from the history, so the history can be thrown away.

We have a _state transition matrix_ which encodes the state transition probabilities, i.e. the probability of going to some state $s'$ from a state $s$.

A Markov process then is a memoryless (i.e. Markov) random process, i.e. a stochastic sequence of states, i.e. a Markov chain. We define a Markov process as a tuple $(S, P)$, where $S$ is a finite set of state and $P$ is the state transition probability matrix:

$$
P_{ss'} = p[S_{t+1} = s' | S_t = s]
$$

A _Markov reward process_ is a Markov chain with values. It is described by a tuple $(S,P,R,\gamma)$, which includes the reward function $R$ and the discount factor $\gamma \in [0,1]$:

$$
R_s = E[R_{t+1} | S_t = s]
$$

The _return_ $G_t$ ("G" for "goal") is the total discounted reward from time-step $t$, i.e.:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The value function $v(s)$ gives the long-term value of state $s$. Formally, it is the expected return of a Markov reward process from state $s$, i.e.:

$$
v(s) = E[G_t | S_t = s]
$$

The Bellman Equation decomposes the value function into two parts:

- the immediate reward $R_{t+1}$
- the discounted value of successor state $\gamma v(S_{t+1})$

So it turns it into a recursive function:

$$
v(s) = E[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]
$$

i.e.:

$$
v(s) = R_s + \gamma \sum_{s' \in S} P_{ss'} v(s')
$$

Or written more concisely using matrices:

$$
v = R + \gamma P v
$$

As an aside, the Bellman equation is a linear equation, so it can be solved directly:

$$
\begin{aligned}
v &= R + \gamma Pv \\
(I - \gamma P) v &= R \\
v &= (I - \gamma P)^{-1}R
\end{aligned}
$$

Where $I$ is the identity matrix.

This however has a computational complexity  of $O(n^3)$ for $n$ states so generally it is not practical.

There are other iterative methods for solving MRPs which are more important/practical, e.g. dynamic programming, Monte-Carlo evaluation, and Temporal-Difference learning.

A Markov decision process (MDP) is a Markov reward process with decisions. It is described by a tuple $(S,A,P,R,\gamma)$ which now includes a finite set of actions $A$, thus now $P$ and $R$ become:

$$
\begin{aligned}
P_{ss'}^a &= p[S_{t+1} = s' | S_t = s, A_t = a] \\
R_s^a &= E[R_{t+1} | S_t = s, A_t=a]
\end{aligned}
$$

A policy $\pi$ is a distribution over actions given states:

$$
\pi(a|s) = p[A_t = a | S_t = s]
$$

A policy fully defines the behavior of an agent. Because it depends only on the current state (and not the history), it is said to be _stationary_ (time-independent).

The _state-value function_ $v_{\pi}(s)$ of an MDP is the expected return starting from state $s$ and then following policy $\pi$:

$$
v_{\pi}(s) = E_{\pi}[G_t|S_t = s]
$$

This is different than the value function, which does not involve a policy for selecting actions but rather only proceeds randomly.

The _action-value function_ $q_{\pi}(s,a)$ is the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$
q_{\pi}(s,a) = E_{\pi}[G_t|S_t = s, A_t = a]
$$

We can also define Bellman equations for these value functions (these are called Bellman expectation equations):

$$
\begin{aligned}
v_{\pi}(s) &= E_{\pi} [R_{t+1} + \gamma v_{\pi} (S_{t+1}) | S_t = s] \\
&= \sum_{a \in A} \pi(a|s) q_{\pi}(s,a) \\
q_{\pi}(s,a) &= E_{\pi} [R_{t+1} + \gamma q_{\pi} (S_{t+1}, A_{t+1}) | S_t = s, A_t = a] \\
&= R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s')
\end{aligned}
$$

We can combine these:

$$
v_{\pi}(s) = \sum_{a \in A} \pi(a|s) (R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s'))
$$

There is some optimal state-value function $v_*(s) = \max_{\pi} v_{\pi}(s)$, i.e. the maximum value function over all policies.

Similarly, there is an optimal action-value function $q_*(s,a) = \max_{\pi} q_{\pi}(s,a)$. This gives us the optimal way for the agent to behave (i.e. we can get the optimal policy from this by always choosing the best action from the optimal action-value function), so this is the most important thing to figure out! That is, the MDP is "solved" when we know $q_*(s,a)$.

From this we can get the Bellman optimality equation:

$$
\begin{aligned}
v_*(s) &= \max_a q_*(s,a) \\
q_*(s,a) &= R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s')
\end{aligned}
$$

Which can then be combined:

$$
v_*(s) = \max_a R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s')
$$

Which can equivalently be written:

$$
q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \max_{a'} q_*(s', a')
$$

The Bellman optimality equation is non-linear, unlike the Bellman equation we saw previously. In general there is no closed form solution, but we can use iterative methods, e.g. value iteration, policy iteration, Q-learning, and Sarsa.

---

## Dynamic programming

"Dynamic" refers to some sequential or temporal aspect to the problem and we want to optimize some program, i.e. a policy.

Dynamic programming is a method for solving complex problems by breaking them into subproblems, then solving the subproblems (divide-and-conquer).

For dynamic programming to work, the problem must have two properties:

- optimal substructure; i.e. the optimal solutions of the subproblems tell us about the optimal solution for the overall problem
- the subproblems should "overlap"; i.e. they should recur throughout the overall problem. That way, by solving a subproblem, we are simultaneously solving many parts of the overall problem (the solutions can be cached and reused).

MDPs satisfy both of these properties with the Bellman equation (it is recursive decomposition) and the value function stores and reuses solutions.

Dynamic programming assumes full knowledge of the MDP, so it is used for _planning_ in an MDP (i.e. prediction and control).

For _prediction_ it takes a fully-specified MDP $(S,A,P,R,\gamma)$ (or MRP $(S,P^{\pi}, R^{\pi}, \gamma)$) and a policy $\pi$ and gives us a value function $v_{\pi}$. So this is not reinforcement learning because the MDP is fully-specified (nothing needs to be learned!).

For _control_ it takes a fully-specified MDP and gives us the optimal value function $v_*$ which gives us an optimal policy $\pi_*$ as well.

### Policy evaluation

Policy evaluation involves taking a policy and an MDP and computing the expected reward for following that policy.

To solve this, we apply the Bellman expectation equation as an iterative update.

We start off with some arbitrary value function $v_1$ (e.g. value of every state is 0), then, use _synchronous_ backups (i.e. consider each state in turn):

- at each iteration $k+1$
- for all states $s \in S$
- update $v_{k+1}(s)$ from $v_k(s')$, where $s'$ is a successor state of $s$

$$
v_{k+1}(s) = \sum_{a \in A} \pi(a|s) (R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_k(s'))
$$

This eventually converges to $v_{\pi}$.

### Policy iteration

Now that we can evaluate a policy, we can figure out how to improve it.

Given a policy $\pi$:

- evaluate the policy $\pi$: $v_{\pi}(s) = E[R_{t+1} + \gamma R_{t+2}  + \dots | S_t = s]$
- improve the policy by acting greedily wrt to $v_{\pi}$: $\pi' = \text{greedy}(v_{\pi})$ (greedy just means we move to the state with the highest value)

And we can iteratively apply this approach (called _greedy policy improvement_), which will converge to the optimal policy $\pi^*$ (no matter how you start).

![Policy iteration intuition [source](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node46.html)](assets/policy_iteration.png)

The policy and the value function influence each other, since the policy dictates which states are explored and the value function influences how the policy chooses states, so they push off each other to convergence.

With the greedy policy, we always choose the best state (remember that the value of a state takes into account future reward from that state as well!) so we update the value function for that state so that it is equal to or greater than it was before. This is how the greedy policy improves the value function.

Because the value function drives the greed policy, that in turn improves the policy.

Eventually the value function will only be equal to (rather than greater than or equal to) what it was before; at this point convergence is achieved.

### Value iteration

The solution $v_*(s)$ can be found with one-step lookahead:

$$
v_*(s) \leftarrow \max_{a \in A} (R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s'))
$$

Then apply these updates iteratively. This is _value iteration_.

Basically we iterate over states and apply this update until convergence, giving us the optimal value function.

The pseudocode is:

- using synchronous backups
    - at each iteration $k + 1$
    - for all states $s \ in S$
    - update $v_{k+1}(s)$ from $v_k(s')$
- until convergence

Unlike policy iteration, there is no explicit policy here, and the intermediate value functions may not correspond to any policy. But once we have $v_*(s)$ we get the optimal policy.

### Asynchronous dynamic programming

The methods so far have used synchronous backups, i.e. each iteration we look at and update every single state simultaneously before updating any state again.

We can instead use _asynchronous backups_ in which states are backed up individually in any order, without waiting for other states to update. This can significantly reduce computation. As long as we continue to select all states (again, order doesn't matter), we will still have convergence.

Three asynchronous dynamic programming methods:

#### In-place dynamic programming

Synchronous value iteration stores two copies of the value function, i.e. for all $s \in S$, we have:

$$
v_{\text{new}}(s) \leftarrow \max_{a \in A}(R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\text{old}}(s'))
$$

_In-place_ value iteration, on the other hand, only stores one copy of the value function, i.e. for all $s \in S$, we instead have:

$$
v(s) \leftarrow \max_{a \in A}(R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v(s'))
$$

That is, we immediately update the value function and always use the newest value function.

#### Prioritised sweeping

Sometimes the order can affect how quickly you reach the optimal value function. _Prioritised sweeping_ uses the magnitude of the Bellman error to guide state selection, e.g.:

$$
|\max_{a \in A}(R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v(s')) - v(s)|
$$

We backup the state with the largest remaining Bellman error, then update the Bellman error of affected states after each backup.

The magnitude of the Bellman error tells us how much our value estimate for that state changed; the intuition is that this change was really big, we want to attend to that state first, since will likely have a large influence on other values.

#### Real-time dynamic programming

With _real-time dynamic programming_ we only select and update the states that the agent actually visits.

### Complexity

These algorithms (i.e. algorithms based on the state-value function $v_{\pi}(s)$ or $v_*(s)$) have complexity $O(mn^2)$ per iteration for $m$ actions and $n$ states.

Dynamic programming uses _full-width_ backups which means we consider _all_ actions and _all_ successor states (i.e. we consider the full branching factor), which is really expensive. Furthermore, to actually do these branch lookaheads we need to know the MDP transitions and reward function (i.e. have a model of the dynamics of the environment).

So we run into the problem of dimensionality, where the number of states $n = |S|$ grows exponentially with the number of state variables. So this is not a great approach for larger problems.

One way around this is by sampling - instead of considering the entire branching factor, we sample a single complete trajectory.

But sometimes the problem is so big that even one backup is too expensive - in these cases, we can use _sample backups_, i.e. we start in a state, sample one action according to our policy, sample one transition according to our dynamics, etc, then backup this branch.

With sample backups, because we are sampling from the dynamics of the environment, this also frees us from needing a model of the dynamics of the environment.

## Model-free prediction

We have an environment that can be represented as an MDP but we are not given the dynamics or reward function (i.e. we don't know what the MDP is). With _model-free_ prediction methods, we can still learn a value function even without this knowledge (i.e. without needing to model the environment).

### Monte-Carlo learning

Not necessarily efficient, but effective.

MC methods learn directly from _complete_ episodes of experience. Note that MC methods must learn from episodes (i.e. they are only applicable to _episodic MDPs_, where episodes eventually terminate). So by "complete" episode, this means the episode is expanded to a terminal state. We do this a lot (each episode is a sample) and estimate the value of our start state as the mean return from these episodes.

Note that the _return_ is the total discounted reward.

The value function is usually the expected return (where $G_t$ is the return):

$$
v_{\pi}(s) = E_{\pi}[G_t | S_t = s]
$$

But Monte-Carlo policy evaluation uses the _empirical mean_ return instead of the expected return (i.e. as mentioned before, we collect sample episodes and compute the mean of their returns).

So how do we get these empirical mean returns for _all_ states in the environment?

There are two methods:

- _first-visit_ Monte-Carlo policy evaluation: to evaluate a state $s$, the first time-step $t$ that state $s$ is visited in an episode (i.e. if the state is returned to later one, ignore), increment counter $N(s) \leftarrow N(s) + 1$, which tracks number of visits to the state, increment the total return $S(s) \leftarrow S(s) + G_t$, then the value is estimated as the mean return $V(s) = \frac{S(s)}{N(s)}$. By the law of large numbers, $V(s) \to v_{\pi}(s)$ as $N(s) \to \infty$; that is, with enough samples, this will converge on the true value.
- _every-visit_ Monte-Carlo policy evaluation: same as the first-visit variant, except now we do these updates for _every_ visit to the state (instead of just the first)

The mean can be computed incrementally, i.e. we can do it in an online fashion. The mean $\mu_1, \mu_2, \dots$ of a sequence $x_1, x_2, \dots$ can be computed incrementally:

$$
\begin{aligned}
\mu_k &= \frac{1}{k} \sum_{j=1}^k x_j \\
&= \frac{1}{k}(x_k + \sum_{j=1}^{k-1} x_j) \\
&= \frac{1}{k} (x_k + (k-1) \mu_{k-1}) \\
&= \mu_{k-1} + \frac{1}{k}(x_k - \mu_{k-1})
\end{aligned}
$$

Using this, we'd change our $V(s)$ update to:

$$
V(s) \leftarrow V(s) + \frac{1}{N(s)} (G_t - V(s))
$$

For non-stationary problems (which is the typical case) we might want to have a running mean (i.e. forget old episodes):

$$
V(s) \leftarrow V(s) + \alpha (G_t - V(s))
$$

### Temporal Difference learning

TD methods, like Monte Carlo learning, learns from episodes, but it can learn from _incomplete_ episodes by _bootstrapping_ (similar to dynamic programming). We substitute the rest of the trajectory (the rest of the episode, before it is finished) with an estimate of what will happen from that state onwards. You take another step, generating an estimate for that step and updating the previous estimate with what what you've learned from then new step.

So whereas with Monte-Carlo learning we update $V(S_t)$ towards the _actual_ return $G_t$, with temporal-difference learning ($TD(0)$) we update $V(S_t)$ towards the _estimated_ return $R_{t+1} + \gamma V(S_{t+1})$ (like the Bellman equations):

$$
V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$

$R_{t+1} + \gamma V(S_{t+1})$ is called the _TD target_.

$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called the _TD error_ (the difference between our estimate before and after taking the step).

The fact that TD methods do not require complete methods means that it can be applied to non-terminating environments.

TDs are often more efficient for Markov environments because they exploit the Markov property (assuming the current state summarizes all previous states). MC methods, on the other hand, do not exploit this property.

Both MC and TD use samples, whereas dynamic programming does not (dynamic programming is exhaustive).

### $TD(\lambda)$

A compromise between TD (looks 1 step into the future) and MC (looks all the way to the end) is to look $n$ steps into the future (i.e. we observe $n$ steps and update the estimate $n$ steps in the past).

So we define the $n$-step return as:

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

So $n$-step temporal-difference learning's update is:

$$
V(S_t) \leftarrow V(S_t) + \alpha (G_t^{(n)} - V(S_t))
$$

We don't have to commit to one value for $n$, we can actually average over $n$-step returns over different $n$. In fact, we can consider _all_ $n$ values with $TD(\lambda)$. Here, we compute the $\lambda$-return $G_t^{\lambda}$ which combines all $n$-step returns $G_t^{(n)}$:

$$
G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

$\lambda$ takes a value from 0 to 1.

So the update is (this is called _forward-view_ $TD(\lambda)$):

$$
V(S_t) \leftarrow V(S_t) + \alpha (G_t^{\lambda} - V(S_t))
$$

Note that like MC this can only be computed from complete episodes.

There is _backward-view_ $TD(\lambda)$ which allows online updates at every step (i.e. from incomplete sequences).

We compute _eligibility traces_ for the credit assignment problem. These combine the frequency heuristic (assign credit to the most frequent states) and the recency heuristic (assign credit to the most recent states):

$$
\begin{aligned}
E_0(s) &= 0 \\
E_t(s) &= \gamma \lambda E_{t-1}(s) + \mathbb{1}(S_t = s)
\end{aligned}
$$

So in backward-view $TD(\lambda)$ we keep an eligibility trace for every state $s$. We update value $V(s)$ for every state $s$ in proportion to TD-error $\delta t$ and eligibility trace $E_t(s)$:

$$
\begin{aligned}
\delta_t &= R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \\
V(s) \leftarrow V(s) + \alpha \delta_t E_t(s)
\end{aligned}
$$

When $\lambda=0$ we have the "vanilla" TD algorithm as presented earlier.