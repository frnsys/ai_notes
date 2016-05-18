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

We start off with some arbitrary value function $v_1$ (e.g. value of every state is 0), then, use _synchronous_ backups:

- at each iteration $k+1$
- for all states $s \in S$
- update $v_{k+1}(s)$ from $v_k(s')$, where $s'$ is a successor state of $s$

$$
v_{k+1}(s) = \sum_{a \in A} \pi(a|s) (R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_k(s'))
$$

This eventually converges to $v_{\pi}$.