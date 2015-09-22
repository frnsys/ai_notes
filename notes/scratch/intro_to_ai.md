## Graph definitions

- if node A leads to node B, then node A is a _parent_ of B and B is a _successor_ or _child_ of A
    - if the edge connecting A to B is due to an operator $q$, we say that "B is a successor to A under the operator $q$".
- if a node has no successors, it is a _terminal_
- if there is a path between node A and node C such that node A is a parent of a parent ... of a parent of C, then A is an _ancestor_ of C and C is a _descendant_ of A.
    - if the graph is cyclical, e.g. there is a path from A through C back to A, then A is both an ancestor and a descendant of C.

## State-space and situation-space representations

In the context of artificial intelligence, problems are often represented using the __state-space__ representation, in which the possible states of the problem and the operations that move between them are represented as a graph.

More formally, we consider a problem to have a set of possible starting states $S$, a set of operators $F$ which can be applied to the states, and a set of goal states $G$. A solution to a problem formalized in this way, called a _procedure_, consists of a starting state $s \in S$ and a sequence of operators that define a path from $s$ to a state in $G$. Typically a problem is represented as a tuple of these values, $(S,F,G)$.

The distinction between state-space and _situation-space_ is as follows: if the relevant parts of the problem are _fully-specified_ (fully-known), then we work with _states_ and _operators_, and have a state-space problem. If there is missing information (i.e., the problem is _partially-specified_), then we work with _situations_ and _actions_, and we have a situation-space problem. Most of what is said for state-space problems is applicable to situation-space problems.

This state-space model can be applied to itself, in such that a given problem can be decomposed into subproblems (also known as _subgoals_); the relationships between the problem and its subproblems (and their subproblems' subproblems, etc) are also represented as a graph. Successor relationships can be grouped by AND or OR arcs which group edges together. A problem node with subproblems linked by AND edges must have all of the grouped subproblems resolved; a problem with subproblems linked by OR edges must have only one of the subproblems resolved. Using this graph, you can identify a path of subproblems which can be used to solve the primary problem. This process is known as _problem reduction_.

## Heuristic Search Theory

Here we consider one type of graphs called _trees_. Trees have a few constraints:

- only one node that does not have a parent, i.e. the _root node_.
- every other node in the tree is a descendant of the root node
- every other node has only one parent

Since many state-space problems can be extremely complex and produce very large graphs, we often do not want to generate the full graph and wish only to focus on fruitful portions of it.  __heuristic search theory__ provides means of searching through such graphs without needing to generate all the states.

Thus any search procedure which does not go through all states is called a _heuristic search procedure_ (because some heuristic is used to reduce the search space). Any search procedure which can be _proven_ to find a solution, given that one exists, is called an _algorithmic search procedure_. A search procedure may be heuristic, algorithmic, both, or neither. For example, an exhaustive search through the entire state graph would be algorithmic, but not heuristic.

We have _generator functions_ which are used to generate the graph. We an apply a set of operators to a given node to generate all successors of that node. A procedure that generates all successors for a node is called a _generator function_ and is notated $\Gamma$. Such a procedure is referred to as _expanding_ a node.

A _breadth-first_ search procedure expands a given node and then expands each of those nodes in the order they were generated. It is an algorithmic search procedure.

A _depth-first_ search procedure expands a given node and then expands the most recently generated node first. In its most basic form, depth-first search is not algorithmic, but the _bounded depth-first_ variant is. In this variant, some depth bound $l$ is specified. The depth of a node is its number of ancestors. A branch is expanded until the depth bound is reached, at which point the search backs up to the closest unexpanded node and begins expanding depth-first there.

Breadth-first and depth-first search, in their basic forms, are both _blind_ search procedures because they do not take into account the location of goal nodes in the state space; as a result, in their basic form, they are not heuristic search procedures.

In addition to generator functions, we may also use _evaluation functions_, which can compute some value for a given node which indicates how likely it is to be on a path to the goal. An evaluation function, for example, could compute a probability that a node leads to the goal. As an unfortunate convention, the lower the number an evaluation function assigns to a node, the more valuable it is. That is, if we have an evaluation function $f$ and two nodes $n$ and $n'$, then $n$ is more valuable than $n'$ if $f(n) < f(n')$.

An _ordered search procedure_ uses an evaluation function and expands unexpanded nodes with the minimum $f(n)$, defaulting to the most recently generated one if there are ties.

One is Hart-Nilsson-Raphael: let $g(n)$ be the depth of node $n$ and $h(n)$ be an estimate of the length of the shortest path from $n$ to a goal node. If for any node $h(n) \leq h_p(n)$, then an ordered search procedure using the evaluation function $f(x) = g(x) + h(x)$ will always find the shortest solution path if one exists. It does not need to search the entire space, and as such, it is both algorithmic and heuristic.

These search procedures are _unidirectional_, since they only expand in one direction (from the start state down). However, there are also _bidirectional_ search procedures which start from both the start state and from the goal state. They can be difficult to use, however.

## Planning

For partially-specified problems, there may not be enough information to identify a solution path. Rather, a __plan__ may be more appropriate.

Plans may be recursive (e.g. include plans to make more plans), may include loops (e.g. do something until successful), conditions (e.g. do X if Y),

## Game Playing

Much AI effort has gone into designing AIs that can play games; in particular, games of _strategy_.

A game of strategy consists of:

- a sequence of moves
- choices between finite alternative moves
- rules which specify:
    - what alternative moves are available, and which player can make them
    - the _payments_ (e.g. a score, points, etc), which can be positive, zero, or negative, resulting from a move

Each player's goal is to maximize the payment they receive.

Variations include (not mutually exclusive):

- Chance (randomness) may be involved, in which case the game is a game of _chance_, otherwise it is a _nonchance_ game.
- A game of _perfect information_ is one in which every player knows every move up to the current point, otherwise it is a game of _imperfect information_.
- A game may have $n > 2$ players, and is called an $n$-player game.
- Players may be competing, cooperating or a mixture of both (i.e. team-based games)
- _Strictly competitive_ or _zero-sum_ games are those in which the payments must sum to zero; that is, any player's gain is a loss for another player.

Games may be represented as graphs in which the nodes are game states and edges are moves. If chance is involved, a given move may lead to several different possible game states.

The main difference between problem solving and game playing is that in game playing the AI does not have complete control over what path is taken - the other player(s) makes choices as well.

The paths that expand from a given node are known together as the _game tree_ of that node. The game tree of the root node is known as the _game tree of the game_. The terminal nodes of a game tree are sometimes called the _tips_ of the game tree.

There are two categories of approaches: local and global.

The global approach looks at the game as a whole. It may find ways of reducing it to simpler games, or prove theorems about the game to derive certain properties, or use past experience of the game.

The local approach looks only at part of the game space.

Complete game spaces are usually too large to be generated.

### Minimax Search

In a two-player competitive zero-sum perfect information game, one approach is __minimax__. In minimax, the value of a particular node for a player is the maximum value of that node's successor nodes that go to that player and is the minimum value of that node's successor nodes that go the other player. This makes sense because the player wants to maximize their own score while minimizing the score of the other player. The value of the node that results from this analysis is called the _theoretical value_ of that node.

If the player is able to compute the theoretical values of all the nodes and always chooses the best-valued node, it is said that the player is playing "perfectly", which means that the player will get the highest possible payment given that the other player is also playing perfectly. If the other player does not play perfectly, then the player will get an even higher payment as a result.

However, it is unlikely that a program will be able to generate the complete game tree and compute all necessary theoretical values. Rather, it is more feasible to only consider "plausible" or "reasonable" successors to each node, rather than all the successors. The game tree which results from only considering these reasonable successors is called a _reasonable game tree_.

The "reasonableness" of a successor is given by an estimate of its theoretical value. This estimate may be computed from a _static evaluation function_ which does not take into account that successor's successors.

### The Alpha-Beta technique

An improvement on minimax search which discards branches that are shown not to improve the score (see the notes elsewhere)

You compute static evaluations (with the static evaluation function) for the tip nodes of the game tree and go bottom-up, applying minimax to estimate the ancestor nodes' values (the resulting value is called the _backed-up evaluation_).

### Game tree generation

Generally you want to generate game trees so that successors to each node are ordered left-to-right in descending order of their eventual backed-up evaluations (such an ordering is called the "correct" ordering). Naturally, it is quite difficult to generate this ordering before these evaluations have been computed.

Thus a _plausible_ ordering must suffice. These are a few techniques for generating plausible orderings of nodes:

- _Generators_ first produce the most immediately desirable choices (though without regard to possible consequences further on)
- _Shallow search_ first generates some of the tree and then uses some static evaluation function and compute backed-up evaluations upwards to order the results.
- _Dynamic generation_, in which alpha-beta is applied to identify plausible branches of the game tree, then branch is evaluated which can cause the ordering to change.

Generation procedures may have some _termination criteria_ which defines when to stop generating a game tree. For instance, it could be up to a maximum depth where only game trees below some depth are produced, or it could be minimum depth where only game trees above some depth are produced, or it could stop when it encounters a tip node for the full game tree, and so on.

_Forward pruning_ is when a generation procedure stop generating the rest of a game tree. There are other kinds of forward pruning, such as _n-best foward pruning_, where only the top $n$ most plausible successors are continued and the rest are pruned, or _tapered n-best forward pruning_, in which $n$ is decreased as the depth of the node increases.

## References

Introduction to Artificial Intelligence (2nd ed). Philip C. Jackson, Jr. 1985.
