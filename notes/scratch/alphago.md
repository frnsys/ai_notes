Mastering the game of Go with deep neural networks and tree search. David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Medeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis. Nature, Vol 529. January 28, 2016.

In a perfect-information game, there is some optimal value function $v^*(s)$ for a given state $s$ that holds when all players play perfectly.

A search space may be too large to fully expand to terminal nodes. There are two techniques for reducing this search space:

1. The _depth_ can be reduced truncating the search tree under state $s$ and using an approximate value function $v(s) \approx v^*(s)$ for that state.
2. The _breadth_ can be reduced by sampling moves (actions) from a learned policy $p(a|s)$ (a probability distribution over possible actions $a$ from state $s$) rather than exhaustively considering every possible move from that state.

A _Monte Carlo rollout_ searches the complete depth of a tree, but along a single branch; that is, at each step it samples one move from the policy and extends the tree with that.

_Monte Carlo Tree Search_ (MCTS) uses such multiple rollouts to value each state.

AlphaGo has three policy (neural) network and a value (neural) network.

The three policy networks are:

1. a supervised learning policy network, $p_{\sigma}$, which is a convolutional neural network
2. a simple linear softmax policy network, $p_{\pi}$, which is less accurate than $p_{\sigma}$ but much faster
3. a reinforcement learning policy network $p_{\rho}$, which has the same architecture as $p_{\sigma}$ and has weights initialized from $p_{\sigma}$ as well (basically, it is a copy of $p_{\sigma}$ that is improved through reinforcement learning)

$p_{\rho}$'s reward function $r(s)$ is zero for all non-terminal time steps. It is trained from randomly selected previous iterations of itself (so as to avoid overfitting to a particular opponent policy).

Typically, the approximate value function $v(s)$ is just a linear combination of features. Here, a neural network is used instead - in particular, a convolutional neural network with the same architecture as $p_{\sigma}$, but outputs a single prediction rather than a distribution.

See the paper for more details on how each of these networks are trained.

The policy networks and value networks are then used as parts of Q-Learning.
