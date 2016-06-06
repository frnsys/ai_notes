[Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1605.06676v2.pdf). Jakob N. Foerster, Nando de Freitas, Yannis M. Assael, Shimon Whiteson.

task:

- multi-agent
- fully cooperative: all agents seek to maximize the same discounted sum of rewards
- partially observable: no agent observes the underlying Markov state, but each receives private observations from it
- sequential

the learning process is centralized: agents communication is unrestricted during learning
execution is decentralized: agents can only communicate over a discrete, limited-bandwidth channel

there are two approaches covered here:

1. reinforced inter-agent learning (RIAL)
    - uses deep-Q learning with an RNN to address partial observability
    - one variant is independent Q-learning: agents independently learn parameters, treating other agents as part of the environment
    - another variant has all parameters shared across agents (i.e. they learn a single, shared neural network)
2. differentiable inter-agent learning (DIAL)
    - gradients are communicated across agents
    - during training, these are communicated as real values
    - during execution, they are discretized

through these approaches, agents learn their own communication protocols.

some background:

(see elsewhere for deep-Q network [DQN] notes)

independent DQNs are multi-agent settings in which each agent behaves independently and learns its own Q function (using a DQN). the reward an agent receives is shared across other agents. independent Q-learning can lead to convergence problems (since as agents learn, they cause the environment to change for all other agents), but in practice it works well

DQNs and independent DQNs assume full observability (the agent receive the full state $s_t$ as input)

deep recurrent Q-networks (DRQN) are applicable to single-agent settings with partially observable environments (that is, $s_t$ is hidden and the agent only receives observation $o_t$, which is correlated with $s_t$). here $Q(o,u)$ (instead of $Q(s,u)$, which is for fully-observable environments; note that $u$ is an action) is approximated with an RNN (which, as a feature of RNNs, aggregates observations over time). since RNNs take the hidden state of the network $h_{t-1}$, we can more accurately write $Q(o_t, h_{t-1}, u)$ as the learned function.

this paper explores partially observable environments in multi-agent settings. here, in addition to selecting an action $u \in U$, here called an _environment action_ (to distinguish it; otherwise this is just like an action is typical reinforcement learning formulations), agents also select a _communication action_ $m \in M$ that other agents observe but has no direct impact on the environment or the reward.

so here agents actually learn two $Q$ functions: $Q_u$, for environment actions, and $Q_m$, for communication actions.

agents have no a priori communication protocol so they must come up with one on their own - so the question is: how to agents efficiently communicate what they know to other agents? agents need to be able to _understand_ each other.

## RIAL

experience replay (typically used for DQNs) is disabled here because it is less effective in multi-agent situations (since agents change the environment so much that past experience memories may be invalidated)

RIAL can have independent parameters for each agent (i.e. they each learn their own networks) or shared across agents (they all learn the same networks). even in the latter case agents can still behave differently (during execution) because they receive different observations (and thus accumulate their own hidden state). learning is also much faster in the latter case since there are much fewer parameters.

another extension is to include an agent's index (i.e. id) $a$ as part of the input to $Q$, so agents can specialize.

so in parameter sharing, the agents learn two $Q$-functions $Q_u(o_t^a, m_{t-1}^{a'}, h_{t-1}^a, u_{t-1}^a, m_{t-1}^a, a, u_t^a)$ and $Q_m(\cdot)$, where:

- $u_{t-1}^a$ and $m_{t-1}^a$ are the last action inputs
- $m_{t-1}^{a'}$ are messages from other agents

## DIAL

DIAL goes a step further than shared-parameter RIAL: gradients are pushed across agents. That is, during centralized learning, communication actions are replaced with direct connections between the output of one agent's network and the input of another's. In this way, agents can "communicate" real valued messages to each other.

This aggregate network is called a _C-Net_. It outputs two types of values:

- $Q(\cdot)$, the Q-values for the environment actions
- $m_t^a$, the real-valued message to other agents
    - this bypasses action selection and instead is processed by the _discretise/regularise unit_ $\text{DRU}(m_t^a)$
    - the DRU regularizes the message during centralized learning, i.e. $\text{DRU}(m_t^a) = \text{Logistic}(\mathcal N(m_t^a, \sigma))$, where $\sigma$ is the standard deviation of the noise added to the channel
    - and discretizes the message during decentralized execution, i.e. $\text{DRU}(m_t^a) = \mathbb{1}{\{m_t^a > 0\}}$


See also: [Learning Multiagent Communication with Backpropagation](http://arxiv.org/pdf/1605.07736v1.pdf) (Sainbayar Sukhbaatar, Arthur Szlam, Rob Fergus)
