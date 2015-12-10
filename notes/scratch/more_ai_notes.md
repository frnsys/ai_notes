# Search and planning

## Agents

- An _agent_ is an entity that _perceives_ and _acts_.
- A _rational agent_ selects actions that maximize its (expected) _utility_.
- Characteristics of the _percepts_, _environment_, and _action space_ dictate techniques for selecting rational actions.

_Reflex_ agents choose actions based on the current percept (and maybe memory). They are concerned almost exclusively with the _current_ state of the world - they do not consider the future consequences of their actions, and they don't have a goal that their are working towards. Rather, they just operate off of simple "reflexes".

Agents that _plan_ consider long(er) term consequences of their actions, have a model of how the world changes based on their actions, and work towards a particular goal (or goals), and can find an optimal solution (plan) for achieving its goal or goals.

## State space graphs

- Nodes are (abstracted) world configurations
- Arcs represent successors (action results)
- The goal test is a set of goal nodes (which may just include a single goal)
- Each state occurs only once

We rarely build the full graph in memory (because it is often way too big). Rather, we work with _search trees_.

At the top of the search tree is the current state, which branches out into possible future states (i.e. the children are successors).

We can't build the full tree (it would be infinite if there are circular paths in the state space graph), so we only build sections that we are immediately concerned with. There is also often a lot of repetition in search trees.

We apply search algorithms to the search tree.

## Search algorithms

When evaluating search algorithms, we care about:

- completeness - is it guaranteed to find a solution if one exists?
- optimal - is it guaranteed to find the optimal solution, if one exists?
- size complexity - how much space does the algorithm need? Basically, how big can the fringe get?
- time complexity - how does runtime change with input size? Basically, how many nodes get expanded?

### Tree size

Given a tree with branching factor $b$ and maximum depth $m$, there are $O(b^m)$ nodes in the tree.

## Uninformed search algorithms

These search algorithms are all the same as general tree search algorithm, they just include different strategies for choosing the next node from the fringe.

### Depth-First Search

- time complexity: expands $O(b^m)$ nodes (if $m$ is finite)
- size complexity: the fringe takes $O(bm)$ space
- complete if $m$ is not infinite (i.e. if there are no cycles)
- optimal: no, it finds the "leftmost" solution

### Breadth-First Search

- time complexity: expands $O(b^s)$ nodes, where $s$ is the depth of the shallowest solution
- size complexity: the fringe takes $O(b^s)$ space
- complete: yes
- optimal: yes, if all costs are 1, otherwise, a deeper path could have a cheaper cost

### Iterative Deepening

The general idea is to combine depth-first search's space advantage with breadth-first search's time/shallow-solution advantages.

- Run depth-first search with depth limit 1
- If no solution:
  - Run depth-first search with depth limit 2
  - If no solution:
    - Run depth-first search with depth limit 3

(etc)

### Uniform Cost Search

We can make breadth-first search sensitive to path cost (that is, return the cheapest path rather than the shallowest path) with _uniform cost search_, in which we simply prioritize paths by their cost rather than by their depth.

- time complexity: If we say the solution costs $C*$ and arcs cost at least $\epsilon$, then the "effective depth" is roughly $\frac{C*}{/epsilon}$, so the time complexity is $O(b^{C*/\epsilon})$
- size complexity: the fringe takes $O(b^{C*/\epsilon})$ space
- complete: yes if the best solution has finite cost and minimum arc cost is positive
- optimal: yes

## Informed search algorithms

Informed search algorithms improve on uninformed search by incorporating _heuristics_ which tell us whether or not we're getting closer to the goal. With heuristics, we can search less of the search space.

In particular, we want _admissible_ heuristics, which is simply a heuristic that never overestimates the distance to the goal.

Note that sometimes inadmissible heuristics (i.e. those that sometimes overestimate the distance to the goal) can be useful.

### Greedy search

Expand the node that you think is closest to the goal state ("closest" as determined by the heuristic).

This often ends up with a suboptimal path, however.

### A\*

(see other notes on A\*)

A\* tree search is optimal if the heuristic is admissible.

Uniform cost search is a special case ($h=0$ is admissible).

## Graph search

With search trees, we often end up with states repeated throughout the tree, which will have redundant subtrees, and thus end up doing (potentially a lot of) redundant computation.

We can avoid this simply by keeping track of the states we have already expanded (as a closed set), then we only expand states which we have not already expanded.

This is _graph search_, and it is a simple extension of tree search.

Completeness is not affected by graph search, but it is not optimal. We may close off a branch because we have already expanded that state elsewhere, but it's possible that the shortest path still goes through that state.

A\* graph search can be made optimal with one more constraint - that its heuristics are _consistent_.

#### Consistent heuristics

The main idea of _consistency_ is that the estimated heuristic costs should be less than or equal to the actual costs for _each arc_.

Given a heuristic $h$, and given an arc between nodes $A$ and $B$, consistency requires that:

$$
h(A) - h(B) \leq \text{cost}(A \to B)
$$

For all connected nodes $A, B$ in the search graph.

This is a stronger condition than admissibility since admissibility is concerned only with the cost between a node and the goal. Consistency implies admissibility.

With consistent heuristics, A\* graph search is optimal.

Uniform cost search is a special case ($h=0$ is consistent).

## Constraint Satisfaction Problems (CSPs)

Search as presented previously makes the following assumptions about the world:

- a single agent
- deterministic actions
- fully observed state
- discrete state space

And we are given a plan (a sequence/path of actions) as the solution.

There are other kinds of search problems which are _identification_ problems, in which our concern is not a path of actions, but rather identifying the goal itself.

Constraint satisfaction problems are specialized for identification problems.

A state is defined by a set of variables $X_i$, with values from a domain $D$ (sometimes the domain varies by $i$).

We want to satisfy a set of _constraints_ on what combinations of values are allowed on different subsets of variables. So we want to identify states which satisfy these constraints.

Constraints can be specified using a formal language, e.g. code that $A \neq B$ or something like that.

We can also represent constraints as a graph.

In a binary CSP, each constraint relates at most two variables. We can construct a binary constraint graph in which the nodes are variables, and arcs show constraints. We don't need to specify what the constraints are.

If we have constraints that are more than binary (that is, they relate more than just two variables), we can represent the constraints as square nodes in the graph and link them to the variables they relate (as opposed to representing constraints as the arcs themselves).

General-purpose CSP algorithms use this graph structure for faster search.


### Varieties of CSPs

Variables may be:

- discrete, and come from
  - finite domains
  - infinite domains (integers, strings, etc)
- continuous

Constraints may be:

- unary (involve a single variable, this is essentially reducing a domain, e.g. $A \neq \text{green}$)
- binary (involve a pair of variables)
- higher-order (involve three or more variables)

We may also have _preferences_, i.e. soft constraints. We can represent these as costs for each variable assignment. This gives us a _constraint optimization problem_.

### Search formulation

We can formulate CSPs as search problems using constraint graphs - we can use them as search trees or search graphs.

States are defined by the values assigned so far (partial assignments).

The initial state is the empty assignment, $\{\}$.

Successor functions assign a value to an unassigned variable (one at a time).

The goal test is to check if the current assignment is complete (all variables have values) and satisfies all constraints.

Breadth-first search does not work well here because all the solutions will be at the bottom of the search tree (all variables must have values assigned, and that happens only at the bottom).

Depth-first search does a little better, but it is very naive - it can make a mistake early on in its path, but not realize it until reaching the end of a branch.

The main shortcoming with these approaches is that we aren't checking constraints until it's far too late.

### Backtracking search

Backtracking search is the basic uninformed search algorithm for solving CSPs. It is a simple augmentation of depth-first search.

Rather than checking the constraint satisfaction at the very end of a branch, we check constraints as we go, i.e. we only try values that do not conflict with previous assignments. This is called an _incremental goal test_.

Furthermore, we only consider one variable at a time in some order. Variable assignments are commutative (i.e. the order in which we assign them doesn't matter, e.g. $A=1$ and then $B=2$ leads to the same variable assignment as $B=2$ then $A=1$). So at one level, we consider assignments for $A$, at the next, for $B$, and so on.

The moment we violate a constraint, we backtrack and try different a variable assignment.

Simple backtracking can be improved in a few ways:

- ordering
  - we can be smarter about in what order we assign variables
  - we can be smarter about what we try for the next value for a variable
- filtering: we can detect failure earlier
- structure: we can exploit the problem structure

TODO add backtracking pseudocode

#### Filtering

Filtering looks ahead to eliminate incompatible variable assignments early on.

With _forward checking_, when we assign a new variable, we look ahead and eliminate values for other variables that we know will be incompatible with this new assignment. So when we reach that variable, we only have to check values we know will not violate a constraint (that is, we only have to consider a subset of the variable's domain).

If we reach an empty domain for a variable, we know to backup.

With _constraint propagation_ methods, we can check for failure ahead of time.

One constraint propagation method is _arc consistency_.

First, we must consider the _consistency_ of an arc (here, in the context of binary constraints, but this can be extended to higher-order constraints). In the context of filtering, an arc $X \to Y$ is _consistent_ if and only if for _every_ $x$ in the tail there is _some_ $y$ in the head which could be assigned without violating a constraint.

An inconsistent arc can be made consistent by deleting values from its tail; that is, by deleting tail values which lead to constraint-violating head values.

Note that since arcs are directional, a consistency relationship (edge) must be checked in both directions.

We can re-frame forward checking as just enforcing consistency of arcs pointing to each new assignment.

A simple form of constraint propagation is to ensure all arcs in the CSP graph are consistent. Basically, we visit each arc, check if its consistent, if not, delete values from its tail until it is consistent. If we encounter an empty domain (that is, we've deleted all values from its tail), then we know we have failed.

Note that if a value is deleted from a tail of a node, its incoming arcs must be-rechecked.

We combine this with backtracking search by applying this filtering after each new variable assignment. It's extra work at each step, but it should save us backtracking.

TODO add arc consistency pseudocode

Arc consistency can be generalized to $k$-consistency:

- 1-consistency is _node consistency_, i.e. each node's domain has a value which satisfies its own unary constraints.
- 2-consistency is arc consistency: for each pair of nodes, any consistent assignment to one can be extended to the other ("extended" meaning from the tail to the head).
- $k$-consistency: for each $k$ nodes, any consistent assignment to $k-1$ can be extended to the $k$th node.
  - 3-consistency is called _path consistency_

Naturally, a higher $k$ consistency is more expensive to compute.

We can extend this further with _strong_ $k$-consistency which means that all lower orders of consistency (i.e. $k-1$ consistency, $k-2$ consistency, etc) are also satisfied. With strong $k$-consistency, no backtracking is necessary - but in practice, it's never practical to compute.

#### Ordering

One method for selecting the next variable to assign to is called _minimum remaining values_ (MRV), in which we choose the variable with the fewest legal values left in its domain (hence this is sometimes called _most constrained variable_). We know this number if we are running forward checking. Essentially we decide to try the hardest variables first so if we fail, we fail early on and thus have to do less backtracking (for this reason, this is sometimes called _fail-fast ordering_).

For choosing the next value to try, a common method is _least constraining value_. That is, we try the value that gives us the most options later on. We may have to re-run filtering to determine what the least constraining value is.

#### Problem Structure

Sometimes there are features of the problem structure that we can use to our advantage.

For example, we may have independent subproblems (that is, we may have multiple connected components; i.e. isolated subgraphs), in which case we can divide-and-conquer.

In practice, however, you almost never see independent subproblems.

##### Tree-Structured CSPs

Some CSPs have a tree structure (i.e. have no loops). Tree-structured CSPs can be solved in $O(nd^2)$ time, much better than the $O(d^n)$ for general CSPs.

The algorithm for solving tree-structured CSPs is as follows:

1. For order in a tree-structured CSP, we first choose a root variable, then order variables such that parents precede children.
2. Backward pass: starting from the end moving backwards, we visit each arc once (the arc pointing from parent to child) and make it consistent.
3. Forward assignment: starting from the root and moving forward, we assign each variable so that it is consistent with its parent.

This method has some nice properties:

- after the backward pass, all root-to-leaf arcs are consistent
- if root-to-leaf arcs are consistent, the forward assignment will not backtrack

Unfortunately, in practice you don't typically encounter tree-structured CSPs.

Rather, we can improve an existing CSPs structure so that it is _nearly_ tree-structured.

Sometimes there are just a few variables which prevent the CSP from having a tree structure.

With _cutset conditioning_, we assign values to these variables such that the rest of the graph is a tree.

This, for example, turns binary constraints into unary constraints, e.g. if we have a constraint $A \neq B$ and we fix $B = \text{green}$, then we can rewrite that constraint as simply $A \neq \text{green}$.

Cutset conditioning with a cutset size $c$ gives runtime $O(d^c (n-c) d^2)$, so it is fast for a small $c$.

More specifically, the cutset conditioning algorithm:

1. choose a cutset (the variables to set values for)
2. instantiate the cutset in all possible ways (e.g. produce a graph for each possible combination of values for the cutset)
3. for each instantiation, compute the _residual_ (tree-structured) CSP by removing the cutset constraints and replacing them with simpler constraints (e.g. replace binary constraints with unary constraints as demonstrated above)
4. solve the residual CSPs

Unfortunately, finding the smallest cutset is an NP-hard problem.

There are other methods for improving the CSP structure, such as _tree decomposition_.

Tree decomposition involves creating "mega-variables" which represent subproblems of the original problem, such that the graph of these mega-variables has a tree structure. For each of these mega-variables we consider valid combinations of assignments to its variables.

These subproblems must overlap in the right way (the _running intersection property_) in order to ensure consistent solutions.

### Iterative improvement algorithms for CSPs

Rather than building solutions step-by-step, iterative algorithms start with an incorrect solution and try to fix it.

Such algorithms are _local search_ methods in that they work with "complete" states (that is, all variables are assigned, though constraints may be violated/unsatisfied), and there is no fringe.

Then we have operators which reassign variable values.

A very simple iterative algorithm:

- while not solved
  - randomly select any conflicted variable
  - select a value which violates the fewest constraints (the _min-conflicts_ heuristic), i.e. hill climb with $h(n) = \text{# of violated constraints}$

In practice, this min-conflicts approach tends to perform quickly for randomly-generated CSPs; that is, there are some particular CSPs which are very hard for it, but for the most part, it can perform in almost constant time for arbitrarily difficult randomly-generated CSPs.

Though, again, unfortunately many real-world CSPs fall in this difficult domain.

## Local search

Local search has no fringe; that is, we don't keep track of unexplored alternatives. Rather, we continuously try to improve a single option until we can't improve it anymore.

Instead of extending a plan, the successor function in local search takes an existing plan and just modifies a part of it.

Local search is generally much faster and more memory efficient, but because it does not keep track of unexplored alternatives, it is incomplete and suboptimal.

A basic method in local search is _hill climbing_ - we choose a starting point, move to the best neighboring state, and repeat until there are no better positions to move to - we've reached the top of hill. As mentioned, this is incomplete and suboptimal, as it can end up in local maxima.

You can also use _simulated annealing_ (detailed elsewhere) to try to escape local maxima - and this helps, and has a theoretical guarantee that it will converge to the optimal state given infinite time, but of course, this is not a practical guarantee for real-world applications. So simulated annealing in practice can do better but still can end up in local optima.

You can also use _genetic algorithms_ (detailed elsewhere).

## Adversarial search

Focus here is on deterministic, two-player, zero-sum, perfect information games.

We want our algorithm to return a _strategy_ (a _policy_) which tells us how to move from each state (as opposed to a plan from start to finish). This is because we can't plan on opponents acting in a particular way, so we need a strategy to respond to their actions.

One formulation of games (there are many):

- states $S$, starting with $s_0$
- players $P=\{1, \dots, n\}$, usually taking turns
- actions $A$ (may depend on player/state)
- transition function (analogous to a successor function), $S \times A \to S$
- terminal test (analogous to a goal test): $S \to \{t, f\}$
- terminal utilities (computes how much an end state is worth to each player): $S \times P \to R$

And the solution for a player is a policy $S \to A$.

In zero-sum games, agents have opposite utilities (one's gain is another's loss). These are always adversarial, "pure competition" games.

In _general_ games, agents have independent utilities, so there is opportunity for cooperation, indifference, competition, and so on.

We can use minimax (see notes elsewhere).

Minimax is just like exhaustive depth-first search, so its time complexity is $O(b^m)$ and space complexity is $O(bm)$.

Minimax is optimal against a perfect adversarial player, but it is not otherwise.

Generally, game trees are far too deep to search all the way to terminal nodes. Instead, we can use depth-limited search to only go down a few levels. However, since we don't reach the terminal nodes, their values never propagate up the tree to the current options. How will we compute the utility of any given move?

We can come up with an _evaluation function_ which computes a utility for non-terminal positions, i.e. it estimates the value of an action. For instance, with chess, you could just take the different of the number of your units vs the number of the opponent's units. Generally moves that lower your opponent's units is better, but not always.

Similarly, you could also use iterative deepening here and search as deep as you have time for, then when time runs out, return whatever the best move you've computed so far is. An evaluation function is still necessary.

We can further improve minimax by _pruning_ the game tree; i.e. removing branches we know won't be worthwhile (notes on this elsewhere, see alpha-beta).

Note that with alpha-beta, the minimax value computed for the root is always correct, but the values of intermediate nodes may be wrong, and as such, (naive) alpha-beta is not great for action selection. Good ordering of child nodes improves upon this. With a "perfect ordering", time complexity drops to $O(b^{m/2})$.

## Search under uncertainty

In many situations the outcomes of your actions are uncertain.

Search under uncertainty is known as _non-deterministic search_.

We can model uncertainty as a "dumb" adversary.

Whereas in minimax we assume a "smart" adversary, and thus consider worst-case outcomes (i.e. that the opponent plays their best move), with uncertainty, we instead consider average-case outcomes (i.e. expected utilities). This is called _expectimax_ search.

So instead of minimax's min nodes, we have "chance" nodes, though we still keep max nodes. For a chance node, we compute its expected utility as the weighted (by probability) average of its children.

Because we take the weighted average of children for a chance node's utility, we cannot use alpha-beta pruning as we could with minimax. There could conceivably be an unexplored child which increases the expected utility enough to make that move ideal, so we have to explore all child nodes to be sure.

## Mixed games

We can have games that involve adversaries _and_ chance, in which case we would have both minimax layers and expectimax layers (called _expectiminimax_).

## Multi-agent utilities

If the game is not zero-sum or has multiple players, we can generalize minimax as such:

- terminal nodes have utility _tuples_
- node values are also utility tuples
- each player maximizes their own component

This can model cooperation and competition dynamically.

## Utilities

We encode _preferences_ for an agent, e.g. $A \succ B$ means the agent prefers $A$ over $B$ (on the other hand, $A \sim B$ means the agent is indifferent about either).

A _lottery_ represents these preferences under uncertainty, e.g. $[p, A; 1-p B]$.

_Rational_ preferences must obey the axioms of rationality:

- orderability: $(A \succ B) \lor (B \succ A) \lor (A \sim B)$. You either have to like $A$ better than $B$, $B$ better than $A$, or be indifferent.
- transitivity: $(A \succ B) \land (B \succ C) \implies (A \sim C)$
- continuity: $A \succ B \succ C \implies \exists p [p, A; 1 - p, C] \sim B$. That is, if $B$ is somewhere between $A$ and $C$, there is some lottery between $A$ and $C$ that is equivalent to $B$.
- substitutability: $A \sim B \implies [p, A;, 1 -p, C] \sim [p, B; 1 - p, C]$. If you're indifferent to $A$ and $B$, you are indifferent to them in lotteries.
- monotonicity: $A \succ B \implies (p \geq q \Leftrightarrow [p, A;, 1 - p, B] \succeq [q, A; 1-q, B])$. If you prefer $A$ over $B$, when given lotteries between $A$ and $B$, you prefer the lottery that is biased towards $A$.

When preferences are rational, they imply behavior that maximizes expected utility, which implies we can come up with a utility function to represent these preferences.

That is, there exists a real-valued function $U$ such that:

$$
\begin{aligned}
U(A) \geq U(B) &\Leftrightarrow A \succeq B \\
U([p1, S_1; \dots ; p_n, S_n]) &= \sum_i p_i U(S_i)
\end{aligned}
$$

The second equation says that the utility of a lottery is the expected value of that lottery.

## Markov Decision Processes (MDPs)

MDPs are another way of modeling non-deterministic search (i.e. actions have uncertain outcomes; another way of saying this is that our actions are _noisy_).

In MDPs, there may be two types of rewards (which can be positive or negative):

- terminal rewards (i.e. those that come at the end, these aren't always present)
- "living" rewards, which are given for each step (these are always present)

For instance, you could imagine a maze arranged on a grid. The desired end of the maze has a positive terminal reward and a dead end of the maze has a negative terminal reward. Every non-terminal position in the maze also has a reward ("living" rewards) associated with it. Often these living rewards are negative so that each step is penalized, thus encouraging the agent to find the desired end in as few steps as possible.

The agent doesn't have complete knowledge of the maze so every action has an uncertain outcome. It can try to move north - sometimes it will successfully do so, but sometimes it will hit a wall and remain in its current position. Sometimes our agent may even move in the wrong direction (e.g. maybe a wheel gets messed up or something).

This kind of scenario can be modeled as a Markov Decision Process, which includes:

- a set of states $s \in S$
- a set of actions $a \in A$
- a transition function $T(s,a,s')$
  - gives the probability that $a$ from $s$ leads to $s'$, i.e. $P(s'|s,a)$
  - also called the "model" or the "dynamics"
- a reward function $R(s,a,s')$ (sometimes just $R(s)$ or $R(s')$)
- a discount $\gamma$
- a start state
- maybe a terminal state

MDPs, as non-deterministic search problems, can be solved with expectimax search.

MDPs are so named because we make the assumption that action outcomes depend only on the current state (i.e. the Markov assumption).

The solution of an MDP is an optimal _policy_ $\pi* : S \to A$:

- gives us an action to take for each state
- an optimal policy maximizes expected utility if followed
- an explicit policy defines a reflex agent

In contrast, expectimax does not give us entire policies. Rather, it gives us an action for a single state only. It's similar to a policy, but requires re-computing at each step. Sometimes this is fine because a problem may be too complicated to compute an entire policy anyways.

Example: Grid World

Note that the X square is a wall. Every movement has an uncertain outcome, e.g. if the agent moves to the east, it may only successfully do so with an 80% chance.

For $R(s) = -0.01$:

|   | A | B | C | D  |
|---|---|---|---|----|
| 0 | → | → | → | +1 |
| 1 | ↑ | X | ← | -1 |
| 2 | ↑ | ← | ← | ↓  |

At C1 the agent plays very conservatively and moves in the opposite direction of the negative terminal position because it can afford doing so many times until it accidentally randomly moves to another position.

Similar reasoning is behind the policy at D2.

For $R(s) = -0.03$:

|   | A | B | C | D  |
|---|---|---|---|----|
| 0 | → | → | → | +1 |
| 1 | ↑ | X | ↑ | -1 |
| 2 | ↑ | ← | ← | ←  |

With a stronger step penalty, the agent finds it better to take a risk and move upwards at C1, since it's too expensive to play conservatively.

Similar reasoning is behind the change in policy at D2.


For $R(s) = -2$:

|   | A | B | C | D  |
|---|---|---|---|----|
| 0 | → | → | → | +1 |
| 1 | ↑ | X | → | -1 |
| 2 | → | → | → | ↑  |

With such a large movement penalty, the agent decides it's better to "commit suicide" by diving into the negative terminal node and end the game as soon as possible.


Each MDP state projects an expectimax-like search tree; that is, we build a search tree from the current state detailing what actions can be taken and the possible outcomes for each action.

We can describe actions and states together as a _q-state_ $(s,a)$. When you're in a state $s$ and you take an action $a$, you end up in this _q-state_ (i.e. you are committed to action $a$ in state $s$) and the resolution of this q-state is described by the _transition_ $(s,a,s')$, described by the probability which is given by transition function $T(s,a,s')$. There is also a reward associated with a transition, $R(s,a,s')$, which may be positive or negative.

How should we encode preferences for _sequences_ of utilities? For example, should the agent prefer the reward sequence $[0,0,1]$ or $[1,0,0]$? It's reasonable to prefer rewards closer in time, e.g. to prefer $[1,0,0]$ over $[0,0,1]$.

We can model this by _discounting_, that is, _decaying_ reward value exponentially. If a reward is worth 1 now, it is worth $\gamma$ one step later, and worth $\gamma^2$ two steps later ($\gamma$ is called the "discount" or "decay rate").

_Stationary preferences_ are those which are invariant to the inclusion of another reward which delays the others in time, i.e.:

$$
[a_1, a_2, \dots] \succ [b_1, b_2, \dots] \Leftrightarrow [r, a_1, a_2, \dots] \succ [r, b_1, b_2, \dots]
$$

Nonstationary preferences are possible, e.g. if the delay of a reward changes its value relative to other rewards (maybe it takes a greater penalty for some reason).

With stationary preferences, there are only two ways to define utilities:

- Additive utility: $U([r_0, r_1, r_2, \dots]) = r_0 + r_1 + r_2 + \dots$
- Discounted utility: $U([r_0, r_1, r_2, \dots]) = r_0 + \gamma r_1 + \gamma^2 r_2 + \dots$

Note that additive utility is just discounted utility where $\gamma = 1$.

For now we will assume stationary preferences.

If a game lasts forever, do we have infinite rewards? Infinite rewards makes it difficult to come up with a good policy.

We can specify a finite horizon (like depth-limited search) and just consider only up to some fixed number of steps. This gives us nonstationary policies, since $\pi$ depends on the time left.

Alternatively, we can just use discounting, where $0 < \gamma < 1$:

$$
U([r_0, \dots, r_{\infty}]) = \sum_{t=0}^{\infty} \gamma^t r_t \leq \frac{R_{\max}}{1 - \gamma}
$$

A smaller $\gamma$ means a shorter-term focus (a smaller _horizon_).

Another way is to use an _absorbing state_. That is, we guarantee that for every policy, a terminal state will eventually be reached.

Usually we use discounting.

### Solving MDPs

We say that the value (utility) of a state $s$ is $V^*(s)$, which is the expected utility of starting in $s$ and acting optimally. This is equivalent to running expectimax from $s$.

While a reward is for a state in a single time step, a value is the expected utility over all paths from that state.

The value (utility) of a q-state $(s,a)$ is $Q^*(s,a)$, called a _Q-value_, which is the expected utility starting out taking action $a$ from state $s$ and subsequently acting optimally. This is equivalent to running expectimax from the chance node that follows from $s$ when taking action $a$.

The optimal policy $\pi*(s)$ gives us the optimal action from a state $s$.

So the main objective is to compute (expectimax) values for the states, since this gives us the expected utility (i.e. average sum of discounted rewards) under optimal action.

More concretely, we can define value recursively:

$$
\begin{aligned}
V^*(s) &= \max_a Q^*(s,a) \\
Q^*(s,a) &= \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]
\end{aligned}
$$

These are the _Bellman equations_.

They can be more compactly written as:

$$
V^*(s) = \max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]
$$

Again, because these trees can go on infinitely (or may just be very deep), we want to limit how far we search (that is, how far we do this recursive computation). We can specify _time-limited values_, i.e. define $V_k(s)$ to be the optimal value of $s$ if the game ends in $k$ more time steps. This is equivalent to depth-$k$ expectimax from $s$.

To clarify, $k=0$ is the _bottom_ of the tree, that is, $k=0$ is the _last_ time step (since there are 0 more steps to the end).

We can use this with the _value iteration_ algorithm to efficiently compute these $V_k(s)$ values in our tree:

- start with $V_0(s) = 0$ (i.e. with no time steps left, we have an expected reward sum of zero). Note that this is a zero vector over all states.
- given a vector of $V_k(s)$ values, do one ply of expectimax from each state:

$$
V_{k+1}(s) = \max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V_k(s')]
$$

Note that since we are starting at the last time step $k=0$ and moving up, when we compute $V_{k+1}(s)$ we have already computed $V_k(s')$, so this saves us extra computation.

Then we simply repeat until convergence. This converges if the discount is less than 1.

With the value iteration algorithm, each iteration has complexity $O(S^2 A)$. There's no penalty for depth here, but the more states you have, the slower this gets.

The approximations get refined towards optimal values the deeper you go into the tree. However, the policy may converge long before the values do - so while you may not have a close approximation of values, the policy/strategy they convey early on may already be optimal.

## Policy evaluation

How do we evaluate policies?

We can compute the values under a fixed policy. That is, we construct a tree based on the policy (it is a much simpler tree because for any given state, we only have one action - the action the policy says to take from that state), and then compute values from that tree.

More specifically, we compute the value of applying a policy $\pi$ from a state $s$:

$$
V^{\pi}(s) = \sum_{s'} T(s, \pi(s), s') [R(s, \pi(s), s') + \gamma V^{\pi}(s')]
$$

Again, since we only have one action to choose from, the $\max_a$ term has been removed.

We can use an approach similar to value iteration to compute these values, i.e.

$$
\begin{aligned}
V_0^{\pi}(s) &= 0 \\
V_{k+1}^{\pi}(s) &= \sum_{s'} T(s, \pi(s), s') [R(s, \pi(s), s') + \gamma V_k^{\pi}(s')]
\end{aligned}
$$

This approach is sometimes called _simple value iteration_ since we've dropped $\max_a$.

This has complexity $O(S^2)$ per iteration.

## Policy extraction

_Policy extraction_ is the problem opposite to policy evaluation - that is, given values, how do we extract the policy which yields these values?

Say we have optimal values $V^*(s)$. We can extract the optimal policy $\pi^*(s)$ like so:

$$
\pi^*(s) = \argmax_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^*(s')]
$$

That is, we do one step of expectimax.

What if we have optimal Q-values instead?

With Q-values, it is trivial to extract the policy, since the hard work is already capture by the Q-value:

$$
\pi^*(s) = \argmax_a Q^*(s,a)
$$

## Policy iteration

Value iteration is quite slow - $O(S^2 A)$ per iteration. However, you may notice that the maximum value calculated for each state rarely changes. The result of this is that the policy often converges long before the values.

_Policy iteration_ is another way of solving MDPs (an alternative to value iteration) in which we start with a given policy and improve on it iteratively:

- First, we evaluate the policy (calculate utilities for the given policy until the utilities converge).
- Then we update the policy using one-step look-ahead (one-step expectimax) with the resulting converged utilities as the future (given) values (i.e. policy extraction).
- Repeat until the policy converges.

Policy iteration is optimal and, under some conditions, can converge must faster.

More formally:

Evaluation: iterate values until convergence:

$$
V_{k+1}^{\pi_i}(s) = \sum_{s'} T(s, \pi_k(s), s') [R(s, \pi_k(s), s') + \gamma V_k^{\pi_i}(s')]
$$

Improvement: compute the new policy with one-step lookahead:

$$
\pi_{i+1}(s) = \argmax_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V^{\pi_i} (s')]
$$

Policy iteration and value iteration are two ways of solving MDPs, and they are quite similar - they are just variations of Bellman updates that use one-step lookahead expectimax.

## Reinforcement learning

MDPs as described above have been fully-observed - that is, we knew all of their parameters (probabilities and so on). Because everything was known in advance, we could conduct _offline planning_, that is, formulate a plan without needing to interact with the world.

MDP parameters aren't always known from the onset, and then we must engage in _online planning_, in which we must interact with the world to learn more about it to better formulate a plan.

Online planning involves _reinforcement learning_:

- the agent interacts with its environment and receives feedback in the form of rewards
- the agent's utility is defined by the reward function
- the agent must learn to act so as to maximize expected rewards
- learning is based on observed samples of outcomes

With reinforcement learning, we still assume a MDP, it's just not fully specified - that is, we don't know $T$ or $R$.

### Model-based learning

Model-based learning is a simple approach to reinforcement learning.

The basic idea:

- learn an approximate model based on experiences
- solve for values (i.e. using value iteration or policy iteration) as if the learned model were correct

In more detail:

1. learn an empirical MDP model
  - count outcomes $s'$ for each $s, a$
  - normalize to get an estimate of $\hat T(s,a,s')$
  - discover each $\hat R(s,a,s')$ when we experience $(s,a,s')$
2. solve the learned MPD (e.g. value iteration or policy iteration)

### Model-free learning

With model-free learning, instead of trying to estimate $T$ or $R$, we take actions and the actual outcome to what we expected the outcome would be.

With _passive reinforcement learning_, the agent is given an existing policy and just learns from the results of that policy's execution (that is, learns the state values; i.e. this is essentially just policy evaluation, except this is _not_ offline, this involves interacting with the environment).

To compute the values for each state under $\pi$, we can use _direct evaluation_:

- act according to $\pi$
- every time we visit a state, record what the sum of discounted rewards turned out to be
- average those samples

Direct evaluation is simple, doesn't require any knowledge of $T$ or $R$, and _eventually_ gets the correct average values. However, it throws out information about state connections, since each state is learned separately - for instance, if we have a state $s_i$ with a positive reward, and another state $s_j$ that leads into it, it's possible that direct evaluation assigns state $s_j$ a negative reward, which doesn't make sense - since it leads to a state with a positive reward, it should also have some positive reward. Given enough time/samples, this will eventually resolve, but that can require a long time.

Policy evaluation, on the other hand, _does_ take in account the relationship between states, since the value of each state is a function of its child states, i.e.

$$
V_{k+1}^{\pi_i}(s) = \sum_{s'} T(s, \pi_k(s), s') [R(s, \pi_k(s), s') + \gamma V_k^{\pi_i}(s')]
$$

However, we don't know $T$ and $R$. Well, we could just try actions and take samples of outcomes $s'$ and average:

$$
V_{k+1}^{\pi}(s) = \frac{1}{n}\sum_i \text{sample}_i
$$

Where each $\text{sample}_i = R(s, \pi(s), s_i') + \gamma V_k^{\pi}(s_i')$. $R(s, \pi(s), s_i')$ is just the observed reward from taking the action.

This is called _sample-based policy evaluation_.

One challenge here: when you try an action, you end up in a new state - how do you get back to the original state to try another action? We don't know anything about the MDP so we don't necessarily know what action will do this.

So really, we only get one sample, and then we're off to another state.

With _temporal difference learning_, we learn from each experience ("episode"); that is, we update $V(s)$ each time we experience a transition $(s,a,s',r)$. The likely outcomes $s'$ will contribute updates more often. The policy is still fixed (given), and we're still doing policy evaluation.

Basically, we have an estimate $V(s)$, and then we take an action and get a new sample. We update $V(s)$ like so:

$$
V^{\pi}(s) = (1 - \alpha) V^{\pi}(s) + (\alpha) \text{sample}
$$

So we specify a learning rate $\alpha$ (usually small, e.g. $\alpha=0.1$) which controls how much of the old estimate we keep. This learning rate can be decreased over time.

This is an exponential moving average.

This update can be re-written as:

$$
V^{\pi}(s) = V^{\pi}(s) + \alpha (\text{sample} - V^{\pi}(s))
$$

The term $(\text{sample} - V^{\pi}(s))$ can be interpreted as an error, i.e. how off our current estimate $V^{\pi}(s)$ was from the observed sample.

So we still never learn $T$ or $R$, we just keep running sample averages instead; hence temporal difference learning is a model-free method for doing policy evaluation.

However, it doesn't help with coming up with a new policy, since we need Q-values to do so.

### Q-Learning

With _active reinforcement learning_, the agent is actively trying new things rather than following a fixed policy.

The fundamental trade-off in active reinforcement learning is _exploitation vs exploration_. When you land on a decent strategy, do you just stick with it? What if there's a better strategy out there? How do you balance using your current best strategy and searching for an even better one?

Remember that value iteration requires us to look at $\max_a$ over the set of possible actions from a state:

$$
V_{k+1}(s) = \max_a \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma V_k(s')]
$$

However, we can't compute maximums from samples; we can only compute averages from samples (TODO why? don't understand this).

We can instead iteratively compute Q-values (Q-value iteration):

$$
Q_{k+1}(s,a) = \sum_{s'} T(s,a,s') [R(s,a,s') + \gamma \max_{a'} Q(s', a')]
$$

Remember that while a value $V(s)$ is the value of a state, a Q-value $Q(s,a)$ is the value of an action (from a particular state).

Here the $\max$ term pushed inside, and we are ultimately just computing an average, so we can compute this from samples.

This is the basis of the _Q-Learning_ algorithm, which is just sample-based Q-value iteration.

We learn $Q(s,a)$ values as we go:

- take an action $a$ from a state $s$ and see the outcome as a sample $(s,a,s',r)$.
- consider the old estimate $Q(s,a)$
- consider the new sample estimate: $\text{sample} = R(s,a,s') + \gamma \max_{a'} Q(s', a')$, where $R(s,a,s') = r$, i.e. the reward we just received
- incorporate this new estimate into a running average:

$$
Q(s,a) = (1 - \alpha)Q(s,a) + (\alpha)\text{sample}
$$

This can also be written:

$$
Q(s,a) =_{\alpha} R(s,a,s') + \gamma \max_{a'} Q(s', a')
$$

These updates emulate Bellman updates as we do in known MDPs.

Q-learning converges to an optimal policy, even if you're acting suboptimally. When an optimal policy is still learned from suboptimal actions, it is called _off-policy learning_.

We still, however, need to explore and decrease the learning rate (but not too quickly or you'll stop learning things).


Aside: a helpful table:

For a known MDP, we can compute an offline solution:

| Goal                          | Technique                 |
|-------------------------------|---------------------------|
| Compute $V^*, Q^*, \pi^*$     | Value or policy iteration |
| Evaluate a fixed policy $\pi$ | Policy evaluation         |

For an unknown MDP, we can use model-based approaches:

| Goal                          | Technique                                         |
|-------------------------------|---------------------------------------------------|
| Compute $V^*, Q^*, \pi^*$     | Value or policy iteration on the approximated MDP |
| Evaluate a fixed policy $\pi$ | Policy evaluation on the approximated MDP         |

Or we can use model-free approaches:

| Goal                          | Technique      |
|-------------------------------|----------------|
| Compute $V^*, Q^*, \pi^*$     | Q-learning     |
| Evaluate a fixed policy $\pi$ | Value learning |

### Exploration vs exploitation

Up until now we have not considered how we select actions. So how do we? That is, how do we explore?

One simple method is to sometimes take random actions ($\epsilon$-greedy). With a small probability $\epsilon$, act randomly, with probability $1-\epsilon$, act on the current best policy.

After the space is thoroughly explored, you don't want to keep moving randomly - so you can decrease $\epsilon$ over time.

Alternatively, we can use _exploration functions_. Generally, we want to explore areas we have high uncertainty for. More specifically, an exploration function takes a value estimate $u$ and a visit count $n$ and returns an optimistic utility. For example: $f(u,n) = u + \frac{k}{n}$.

We can modify our Q-update to incorporate an exploration function:

$$
Q(s,a) =_{\alpha} R(s,a,s') + \gamma \max_{a'} f(Q(s', a'), N(s',a'))
$$

This encourages the agent not only to try unknown states, but to also try states that lead to unknown states.

In addition to exploration and exploitation, we also introduce a concept of _regret_. Naturally, mistakes are made as the space is explored - regret is a measure of the total mistake cost. That is, it is the difference between your expected rewards and optimal expected rewards.

We can try to minimize regret - to do so, we must not only learn to be optimal, but we must _optimally_ learn how to be optimal.

For example: both random exploration and exploration functions are optimal, but random exploration has higher regret.

### Approximate Q-Learning

Sometimes spaces are far too large to satisfactorily explore. This can be a limit of memory (since Q-learning keeps a table of Q-values) or simply that there are too many states to visit in a reasonable time. In fact, this is the rule rather than the exception. So in practice we cannot learn about every state.

The general idea of _approximate Q-learning_ is to transfer learnings from one state to other similar states. For example, if we learn from exploring one state that a fire pit is bad, then we can generalize that all fire pit states are probably bad.

This is an approach like machine learning - we want to learn general knowledge from a few training states; the states are represented by features (for example, we could have a binary feature $\text{has fire pit}$). Then we describe q-states in terms of features, e.g. as linear functions (called a _Q-function_):

$$
Q(s,a) = w_1 f_1(s,a) + w_2 f_2(s,a) + \dots + w_n f_n(s,a)
$$

Note that we can do the same for value functions as well, i.e.

$$
V(s) = w_1 f_1(s) + w_2 f_2(s) + \dots + w_n f_n(s)
$$

So we observe a transition $(s,a,r,s')$ and then we compute the difference of this observed transition from what we expected, i.e:

$$
\text{difference} = [r + \gamma \max_{a'} Q(s', a')] - Q(s,a)
$$

With exact Q-learning, we would update $Q(s,a)$ like so:

$$
Q(s,a) = Q(s,a) + \alpha [\text{difference}]
$$

With approximate Q-learning, we instead update the weights, and we do so in proportion to their feature values:

$$
w_i = w_i + \alpha [\text{difference}] f_i (s,a)
$$

This is the same as least-squares regression.

That is, given a point $x$, with features $f(x)$ and target value $y$, the error is:

$$
\text{error}(w) = \frac{1}{2} (y - \sum_k w_k f_k(x))^2
$$

The derivative of the error with respect to a weight $w_m$ is:

$$
\frac{\partial \text{error}(w)}{\partial w_m} = - ( y- \sum_k w_k f_k(x)) f_m(x)
$$

Then we update the weight:

$$
w_m = w_m + \alpha (y - \sum_k w_k f_k(x)) f_m(x)
$$

In terms of approximate Q-learning, the target $y$ is $r + \gamma \max_{a'} Q(s', a')$ and our prediction is $Q(s,a)$:

$$
w_m = w_m + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)] f_m (s,a)
$$


### Policy Search

Q-learning tries to model the states by learning q-values. However, a feature-based Q-learning model that models the states well does not necessarily translate to a good feature-based policy (and vice-versa).

Instead of trying to model the unknown states, we can directly try to learn the policies that maximize rewards.

So we can use Q-learning and learn a decent solution, then fine-tune by hill climbing on feature weights. That is, we learn an initial linear Q-function, the nudge each feature weight and up and down to see if the resulting policy is better.

We test whether or not a policy is better by running many sample episodes.

If we have many features, we have to test many new policies, and this hill climbing approach becomes impractical. There are better methods (not discussed here).


# Probabilistic reasoning

Deriving Bayes' rule from the product rule of probability:

$$
\begin{aligned}
P(x,y) &= P(x|y)P(y) = P(y|x)P(x) \\
P(x|y)P(y) &= P(y|x)P(x) \\
P(x|y &= \frac{P(y|x)P(x)}{P(y)}
\end{aligned}
$$

$P(y)$ is just a normalization constant and often we do not need to worry about it (If we are comparing quantities).

### Bayes' Nets

Independent allows us to more compactly represent joint probability distributions, in that independent random variables can be represented as smaller, separate probability distributions.

For example, if we have binary random variables $A,B,C,D$, we would have a joint probability table of $2^4$ entries. However, if we know that $A,B$ is independent of $C,D$, then we only need two joint probability tables of $2^2$ entries each.

Typically, independent is too strong an assumption to make for real-world applications, but we can often make the weaker, yet still useful assumption of conditional independence.

Conditional independence is when one variable makes another variable irrelevant (because the other variable adds no additional information), i.e. $P(A|B,C) = P(A|B)$; knowing $C$ adds no more information when we know $B$.

For example, if $C$ causes $B$ and $B$ causes $A$, then knowledge of $B$ already implies $C$, so knowing about $C$ is kind of useless for learning about $A$ if we already know $B$.

As a more concrete example, given random variables traffic $T$, umbrella $U$, and raining $R$, we could reasonably assume that $U$ is conditionally independent of $T$ given $R$, because rain is the common cause of the two and there is no direct relationship between $U$ and $T$; the relationship is through $R$.

Similarly, given fire, smoke and an alarm, we could say that fire and alarm are conditionally independent given smoke.

As mentioned earlier, we can apply conditional independence to simplify joint distributions.

Take the traffic/umbrella/rain example from before. Their joint distribution is $P(T,R,U$, which we can decompose using the chain rule:

$$
P(T,R,U) = P(R)P(T|R)P(U|R,T)
$$

If we make the conditional independence assumption from before ($U$ and $T$ are conditionally independent given $R$), then we can simplify this:

$$
P(T,R,U) = P(R)P(T|R)P(U|R)
$$

That is, we simplified $P(U|R,T)$ to $P(U|R)$.

We can describe complex joint distributions more simply with these conditional independence assumptions, and we can do so with Bayes' nets (i.e. graphical models), which provide additional insight into the structure of these distributions (in particular, how variables interact locally, and how these local interactions propagate to more distant indirect interactions).

A Bayes' net is a directed acyclic graph.

The nodes in the graph are the variables (with domains). They may be assigned (observed) or unassigned (unobserved).

The arcs in the graphs are interactions between variables (similar to constraints in CSPs). They indicate "direct influence" between variables (not that this is _not_ necessarily the same as causation, it's about the information that observation of one variable gives about the other, which can mean causation, but not necessarily, e.g. it could simply be a hidden common underlying cause), which is to say that they encode conditional independences.

For each node, we have a conditional distribution over the variable that node represents, conditioned on its parents' values.

Bayes' nets implicitly encode joint distributions as a product of local conditional distributions:

$$
P(x_1, x_2, \dots, x_n) = \prod_{i=1}^n P(x_i | \text{parents}(X_i))
$$

This simply comes from the chain rule:

$$
P(x_1, x_2, \dots, x_n) = \prod_{i=1}^n P(x_i | x_1, \dots, x_{i-1})
$$

And then applying conditional independence assumptions.

The graph must be acyclic so that we can come up with a consistent ordering when we apply the chain rule (that is, decide the order for expanding the distributions). If the graph has cycles, we can't come up with a consistent ordering because we will have loops.

Note that arcs can be "reversed" (i.e. parent and children can be swapped) and encode the same joint distribution - so joint distributions can be represented by multiple Bayes' nets. But some Bayes' nets are better representations than others - some will be easier to work with; in particular, if the arcs _do_ represent causality, the network will be easier to work with.

Bayes' nets are much smaller than representing such joint distributions without conditional independence assumptions.

A joint distribution over $N$ boolean variables takes $2^n$ space (as demonstrated earlier).

A Bayes' net, on the other hand, where the $N$ nodes each have at most $k$ parents, only requires size $O(N * 2^{k+1})$.

The Bayes' net also encodes additional conditional independence assumptions in its structure.

For example, the Bayes' net $X \to Y \to Z \to W$ encodes the joint distribution:

$$
P(X,Y,Z,W) = P(X) P(Y|X) P(Z|Y) P(W|Z)
$$

This structure implies other conditional independence assumptions, e.g. that $Z$ is conditionally independent of $X$ given $Y$, i.e. $P(Z|Y) = P(Z|X,Y)$.

More generally we might ask: given two nodes, are they independent given certain evidence and the structure of the graph (i.e. assignments of intermediary nodes)?

We can use the _D-separation_ algorithm to answer this question.

First, we consider three configurations of _triples_ as base cases, which we can use to deal with more complex networks. That is, any Bayes' net can be decomposed into these three triple configurations.

A simple configuration of nodes in the form of $X \to Y \to Z$ is called a _causal chain_ and encodes the joint distribution $P(x,y,z) = P(x) P(y|x) P(z|y)$.

$X$ is not guaranteed to be (unconditionally) independent of $Z$.

However, is $X$ guaranteed to be conditionally independent of $Z$ given $Y$?

From the definition of conditional probability, we know that:

$$
P(z|x,y) = \frac{P(x,y,z)}{P(x,y)}
$$

With the Bayes' net, we can simplify this (the numerator comes from the joint distribution the graph encodes, as demonstrated previously, and the denominator comes from applying the product rule):

$$
P(z|x,y) = \frac{P(x)P(y|x)P(z|y)}{P(x)P(y|x)}
$$

Then, canceling a few things out:

$$
P(z|x,y) = P(z|y)
$$

So yes, $X$ is guaranteed to be conditionally independent of $Z$ given $Y$ (i.e. once $Y$ is observed). We say that evidence along the chain "blocks" the influence.

Another configuration of nodes is a _common cause_ configuration:

$$
\begin{aligned}
Y &\to X \\
Y &\to Z
\end{aligned}
$$

The encoded joint distribution is $P(x,y,z) = P(y)P(x|y)P(z|y)$.

Again, $X$ is not guaranteed to be (unconditionally) independent of $Z$.

Is $X$ guaranteed to be conditionally independent of $Z$ given $Y$?

Again, we start with the definition of conditional probability:

$$
P(z|x,y) = \frac{P(x,y,z)}{P(x,y)}
$$

Apply the product rule to the denominator and replace the numerator with the Bayes' net's joint distribution:

$$
P(z|x,y) = \frac{P(y)P(x|y)P(z|y)}{P(y)P(x|y)}
$$

Yielding:

$$
P(z|x,y) = P(z|y)
$$

So again, yes, $X$ is guaranteed to be conditionally independent of $Z$ given $Y$.

Another triple configuration is the _common effect_ configuration (also called _v-structures_):

$$
\begin{aligned}
X &\to Z \\
Y &\to Z
\end{aligned}
$$

$X$ and $Y$ _are_ (unconditionally) independent here.

However, is $X$ guaranteed to be conditionally independent of $Y$ given $Z$?

No - observing $Z$ puts $X$ and $Y$ in competition as the explanation for $Z$ (this is called _causal competition_). That is, having observed $Z$, we think that $X$ or $Y$ was the cause, but not both, so now they are dependent on each other (if one happened, the other didn't, and vice versa).

Consider the following Bayes' net:

TODO draw this out
$$
\begin{aligned}
L &\to R \\
R &\to D \\
R &\to T \\
B &\to T
\end{aligned}
$$

Where our random variables are rain $R$, dripping roof $D$, low pressure $L$, traffic $T$, baseball game $B$.

The relationships assumed here are: low pressure fronts cause rain, rain or a baseball game causes traffic, and rain causes your friend's roof to drip.

Given that you observe traffic, the probability that your friend's roof is dripping goes up - since perhaps the traffic is caused by rain, which would cause the roof to drip. This relationship is encoded in the graph the path between $T$ and $D$.

However - if we observe that it is raining, then observation of traffic has no more effect on $D$ - intuitively, this makes sense - we already know it's raining, so seeing traffic doesn't tell us more about the roof dripping. In this sense, observing $R$ "blocks" the path between $T$ and $D$.

One exception here is the v-structure with $R,B,T$. Observing that a baseball game is happening affects our belief about it raining _only_ if we have observed $T$. Otherwise, they are independent. So v-structures are "reversed" in some sense.

That is, we must observe $T$ to _activate_ the path between $R$ and $B$.

Thus we make the distinction between _active_ triples, in which information "flows" as it did with the path between $T$ and $D$ and between $R$ and $B$ when $T$ is observed, and _inactive_ triples, in which this information is "blocked".

Active triples are chain and common cause configurations in which the central node is _not_ observed and common effect configurations in which the central node _is_ observed, _or_ common effect configurations in which some child node of the central node is observed.

An example for the last case:

$$
\begin{aligned}
X &\to Z \\
Y &\to Z \\
Z &\to A \to B \to C
\end{aligned}
$$

If $Z$, $A$, $B$ or $C$ are observed, then the triple is active.

Inactive triples are chain and common cause configurations in which the central node _is_ observed and common effect configurations in which the central node is _not_ observed.

So now, if we want to know if two nodes $X$ and $Y$ are conditionally independent given some evidence variables $\{Z\}$, we check all undirected paths from $X$ to $Y$ and see if there are any active paths (by checking all its constituent triples). If there are none, then they are conditionally independent, and we say that they are _d-separated_. Otherwise, conditional independence is not guaranteed. This is the _d-separation_ algorithm.

You can apply d-separation to a Bayes net and get a complete list of conditional independences that are necessarily true given certain evidence. This tells you the set of probability distributions that can be represented.

### Inference in Bayes' nets

Given a query, i.e. a joint probability distribution we are interested in getting a value for, we can infer an answer for that query from a Bayes' net.

The simplest approach is _inference by enumeration_ in which we extract the conditional probabilities from the Bayes' net and appropriately combine them together.

For example, if we want to know $P(x|y,z)$ we could go through our Bayes' net's conditional probability tables and do something like:

TODO example

But this is very inefficient, especially because variables that aren't in the query require us to enumerate over all possible values for them. We lose most of the benefit of having this compact representation of joint distributions.

An alternative approach is _variable elimination_, which is still NP-hard, but faster than enumeration.

Variable elimination requires the notion of _factors_. Here are some factors:

- a joint distribution: $P(X,Y)$, which is just all entries $P(x,y)$ for all $x,y$ and sums to 1.

Example:

$$
P(T,W)
$$

| T    | W    | P   |
|------|------|-----|
| hot  | sun  | 0.4 |
| hot  | rain | 0.1 |
| cold | sun  | 0.2 |
| cold | rain | 0.3 |

- a selected joint: $P(x,Y)$, i.e. we fix $X=x$, then look at all entries $P(x,y)$ for all $y$, and sums to $P(x)$. This is a "slice" of the joint distribution.

Example:

$$
P(\text{cold}, W)
$$

| T    | W    | P   |
|------|------|-----|
| cold | sun  | 0.2 |
| cold | rain | 0.3 |

- a single conditional: $P(Y|x)$, i.e. we fix $X=x$, then look at all entries $P(y|x)$ for all $y$, and sums to 1.

Example:

$$
P(W|\text{cold})
$$

| T    | W    | P   |
|------|------|-----|
| cold | sun  | 0.4 |
| cold | rain | 0.6 |

- a family of conditionals: $P(X,Y)$, i.e. we have multiple conditions, all entries $P(x|y)$ for all $x, y$, and sums to $|Y|$.

Example:

$$
P(W|T)
$$

| T    | W    | P   |
|------|------|-----|
| hot  | sun  | 0.8 |
| hot  | rain | 0.2 |
| cold | sun  | 0.4 |
| cold | rain | 0.6 |

- a specified family: $P(y|X)$, i.e. we fix $y$ and look at all entries $P(y|x)$ for all $x$. Can sum to anything;

Example:

$$
P(\text{rain}|T)
$$

| T    | W    | P   |
|------|------|-----|
| hot  | rain | 0.2 |
| cold | rain | 0.6 |


In general, when we write $P(Y_1, \dots, Y_N | X_1, \dots, X_M)$, we have a factor, i.e. a multi-dimensional array for which the values are all instantiations $P(y_1, \dots, y_N|x_1, \dots, x_M)$.

Any assigned/instantiated $X$ or $Y$ is a dimension missing (selected) from the array, which leads to smaller factors - when we fix values, we don't have to consider every possible instantiation of that variable anymore, so we have less possible combinations of variable values to consider.

For example, if $X$ and $Y$ are both binary random variables, if we don't fix either of them we have four to consider ($(X=0,Y=0), (X=1,Y=0), (X=0,Y=1), (X=1,Y=1)$) . If we fix, say $X=1$, then we only have two to consider ($(X=1,Y=0), (X=1,Y=1)$).


Consider a simple Bayes' net:

$$
R \to T \to L
$$

Where $R$ is whether or not it is raining, $T$ is whether or not there is traffic, and $L$ is whether or not we are late for class.

We are given the following factors for this Bayes' net:

$P(R)$

| R  | P   |
|----|-----|
| +r | 0.1 |
| -r | 0.9 |

$P(T|R)$

| R  | T  | P   |
|----|----|-----|
| +r | +t | 0.8 |
| +r | -t | 0.2 |
| -r | +t | 0.1 |
| -r | -t | 0.9 |

$P(L|T)$

| T  | L  | P   |
|----|----|-----|
| +t | +l | 0.3 |
| +t | -l | 0.7 |
| -t | +l | 0.1 |
| -t | -l | 0.9 |

For example, if we observe $L=+l$, so we can fix that value and shrink the last factor $P(L|T)$:

$P(+l|T)$

| T  | L  | P   |
|----|----|-----|
| +t | +l | 0.3 |
| -t | +l | 0.1 |

We can _join_ factors, which gives us a new factor over the union of the variables involved.

For example, we can join on $R$, which involves picking all factors involving $R$, i.e. $P(R)$ and $P(T|R)$, giving us $P(R,T)$. The join is accomplished by computing the entry-wise products, e.g. for each $r,t$, compute $P(r,t) = P(r) P(t|r)$:

$P(R,T)$

| R  | T  | P    |
|----|----|------|
| +r | +t | 0.08 |
| +r | -t | 0.02 |
| -r | +t | 0.09 |
| -r | -t | 0.81 |

After completing this join, the resulting factor $P(R,T)$ replaces $P(R)$ and $P(T|R)$, so our Bayes' net is now:

$$
(R,T) \to L
$$

We can then join on $T$, which involves $P(L|T)$ and $P(R,T)$, giving us $P(R,T,L)$:

$P(R,T,L)$

| R  | T  | L  | P     |
|----|----|----|-------|
| +r | +t | +l | 0.024 |
| +r | +t | -l | 0.056 |
| +r | -t | +l | 0.002 |
| +r | -t | -l | 0.018 |
| -r | +t | +l | 0.027 |
| -r | +t | -l | 0.063 |
| -r | -t | +l | 0.081 |
| -r | -t | -l | 0.729 |

Now we have this joint distribution, and we can use the _marginalization_ operation (also called _elimination_) on this factor - that is, we can sum out a variable to shrink the factor. We can only do this if the variable appears in only one factor.

For example, say we still had our factor $P(R,T)$ and we wanted to get $P(T)$. We can do so by summing out $R$:

$P(T)$

| T  | P     |
|----|-------|
| +t | 0.17  |
| -t | 0.83  |

So we can take our full joint distribution $P(R,T,L)$ and get $P(T,L)$ by elimination (in particular, by summing out $R$):

$P(T,L)$

| T  | L  | P     |
|----|----|-------|
| +t | +l | 0.051 |
| +t | -l | 0.119 |
| -t | +l | 0.083 |
| -t | -l | 0.747 |

Then we can further sum out $T$ to get $P(L)$:

$P(L)$

| L  | P     |
|----|-------|
| +l | 0.134 |
| -l | 0.866 |

This approach is equivalent to inference by enumeration (building up the full joint distribution, then taking it apart to get to the desired quantity).

However, we can use these operations (join and elimination) to find "shortcuts" to the desired quantity (i.e. marginalize early without needing to build the entire joint distribution first). This method is _variable elimination_.

For example, we can compute $P(L)$ in a shorter route:

- join on $R$, as before, to get $P(R,T)$
- then eliminate (sum out) $R$ from $P(R,T)$ to get $P(T)$
- then join on $T$, i.e. with $P(T)$ and $P(L|T)$, giving us $P(T,L)$
- the eliminate $T$, giving us $P(L)$

In contrast, the enumeration method required:

- join on $R$ to get $P(R,T)$
- join on $T$ to get $P(R,T,L)$
- eliminate $R$ to get $P(T)$
- eliminate $T$ to get $P(L)$

The advantage of variable elimination is that we never build a factor of more than two variables (i.e. the full joint distribution $P(R,T,L)$), thus saving time and space. The largest factor typically has the greatest influence over the computation complexity.

In this case, we had no evidence (i.e. no fixed values) to work with. If we had evidence, we would first shrink the factors involving the observed variable, and the evidence would be retained in the final factor (since we can't sum it out once it's observed).

For example, say we observed $R=+r$.

We would take our initial factors and shrink those involving $R$:

$P(+r)$

| R  | P   |
|----|-----|
| +r | 0.1 |

$P(T|+r)$

| R  | T  | P   |
|----|----|-----|
| +r | +t | 0.8 |
| +r | -t | 0.2 |

And we would eventually end up with:

$P(+r, L)$

| R  | L  | P     |
|----|----|-------|
| +r | +l | 0.026 |
| +r | -l | 0.074 |

And then we could get $P(L|+r)$ by normalizing $P(+r, L)$:

$P(L|+r)$

| L  | P    |
|----|------|
| +l | 0.26 |
| -l | 0.74 |

More concretely, the general variable elimination algorithm is such:

- start with a query $P(Q|E_1 = e_1, \dots, E_k = e_k)$, where $Q$ are your query variables
- start with initial factors (i.e. local conditional probability tables instantiated by the evidence $E_1, \dots, E_k$, i.e. shrink factors involving the evidence)
- while there are still hidden variables (i.e. those in the net that are not $Q$ or any of the evidence $E_1, \dots, E_k$)
  - pick a hidden variable $H$
  - join all factors mentioning $H$
  - eliminate (sum out) $H$
- then join all remaining factors and normalize. The resulting distribution will be $P(Q | e_1, \dots, e_k)$.

The order in which you eliminate variables affects computational complexity in that some orderings generate larger factors than others. Again, the factor size is what influences complexity, so you want to use orderings that produce small factors.

For example, if a variable is mentioned in many factors, you generally want to avoid computing that until later on (usually last). This is because a variable mentioned in many factors means joining over many factors, which will probably produce a very large factor.

We can encode this in the algorithm by telling it to choose the next hidden variable that would produce the smallest factor (since factor sizes are relatively easy to compute without needing to actually produce the factor, just look at the number and sizes of tables that would have to be joined).

Unfortunately there isn't always an ordering with small factors, so variable elimination is great in many situations, but not all.

##### Polytrees

A _polytree_ is a directed graph with no undirected cycles. For polytrees we can always find an ordering that is efficient (generates small factors). We can apply cutset conditioning to a Bayes' net, i.e. choose a set of variables such that if they are removed, only a polytree remains, and then we can apply variable elimination. Here, "removing" a variable really means copying your Bayes' net so that there is one for each instantiation of that variable, then you combine the answers resulting from each of these copies.

#### Sampling

Another method for Bayes' net inference is _sampling_. This is an _approximate_ inference method, but it can be much faster. Here, "sampling" essentially means "repeated simulation".

The basic idea:

- draw $N$ samples from a sampling distribution $S$
- compute an approximate posterior probability
- with enough samples, this converges to the true probability $P$

Sampling from a given distribution:

1. Get sample $u$ from a uniform distribution over $[0,1]$
2. Convert this sample $u$ into an outcome for the given distribution by having each outcome associated with a sub-interval of $[0,1)$ with sub-interval size equal to the probability of the outcome

For example, if we have the following distribution:

| C     | P(C) |
|-------|------|
| red   | 0.6  |
| green | 0.1  |
| blue  | 0.3  |

Then we can map $u$ to $C$ in this way:

$$
c =
\begin{cases}
\text{red} & \text{if} 0 \leq u < 0.6 \\
\text{green} & \text{if} 0.6 \leq u < 0.7 \\
\text{blue} & \text{if} 0.7 \leq u < 1
\end{cases}
$$

There are many different sampling strategies for Bayes' nets:

- prior sampling
- rejection sampling
- likelihood weighting
- Gibbs sampling

In practice, you typically want to use either likelihood weighting or Gibbs sampling.

##### Prior sampling

We have a Bayes' net, and we want to sample the full joint distribution it encodes, but we don't want to have to build the full joint distribution.

Imagine we have the following Bayes' net:

$$
\begin{aligned}
P(C) \to P(R|C) \\
P(C) \to P(S|C) \\
P(R|C) \to P(W|S,R) \\
P(S|C) \to P(W|S,R)
\end{aligned}
$$

Where $C,R,S,W$ are binary variables (i.e. $C$ can be $+c$ or $-c$).

We start from $P(C)$ and sample a value $c$ from that distribution. Then we sample $r$ from $P(R|C)$ and $s$ from $P(S|C)$ conditioned on the value $c$ we sampled from $P(C)$. Then we sample from $P(W|S,R)$ conditioned on the sampled $r,s$ values.

Basically, we walk through the graph, sampling from the distribution at each node, and we choose a path through the graph such that we can condition on previously-sampled variables. This generates _one_ final sample across the different variables. If we want more samples, we have to repeat this process.

Prior sampling ($S_{PS}$) generates samples with probability:

$$
S_{PS}(x_1, \dots, x_n) = \prod_{i=1}^n P(x_i|\text{Parents}(X_i)) = P(x_1, dots, x_n)
$$

That is, it generates samples from the actual joint distribution the Bayes' net encodes, which is to say that this sampling procedure is _consistent_. This is worth mentioning because this isn't always the case; some sampling strategies sample from a different distribution and compensate in other ways.

Then we can use these samples to estimate $P(W)$ or other quantities we may be interested in, but we need many samples to get good estimates.

##### Rejection sampling

Prior sampling can be overkill, since we typically keep samples which are irrelevant to the problem at hand. We can instead use the same approach but discard irrelevant samples.

For instance, if we want to compute $P(W)$, we only care about values that $W$ takes on, so we don't need to keep the corresponding values for $C,S,R$. Similarly, maybe we are interested in $P(C|+s)$ - so we should only be keeping samples where $S=+s$.

This method is called _rejection sampling_ because we are rejecting samples that are irrelevant to our problem. This method is also consistent.

##### Likelihood Weighting

A problem with rejection sampling is that if the evidence is unlikely, we have to reject a lot of samples.

For example, if we wanted to estimate $P(C|+s)$ and $S=+s$ is generally very rare, then many of our samples will be rejected.

We could instead fix the evidence variables, i.e. when it comes to sample $S$, just say $S=+s$. But then our sample distribution is not consistent.

We can fix this by weighting each sample by the probability of the evidence (e.g. $S=+s$) given its parents (e.g. $P(+s|\text{Parents})$).

##### Gibbs sampling

With likelihood weighting, we consider the evidence only for variables sampled after we fixed the evidence (that is, that come after the evidence node in our walk through the Bayes' net). Anything we sampled before did not take the evidence into account. It's possible that what we sample before we get to our evidence is very inconsistent with the evidence, i.e. makes it very unlikely and gives us a very low weight for our sample.

With Gibbs sampling, we fix our evidence and then instantiate of all our other variables, $x_1, \dots, x_n$. This instantiation is arbitrary but it must be consistent with the evidence.

Then, we sample a new value for one variable at a time, conditioned on the rest, though we keep the evidence fixed. We repeat this many times.

If we repeat this infinitely many times, the resulting sample comes from the correct distribution, and it is conditioned on both the upstream (pre-evidence) and downstream (post-evidence) variables.

## Decision Networks

Decision networks are similar to Bayes' networks. Some nodes are random variables (these are essentially embedded Bayes' networks), some nodes are _action variables_, in which a decision is made, and some nodes are utility functions, which computes a utility for its parent nodes.

For instance, an action node could be "bring (or don't bring) an umbrella", and a random variable node could be "it is/isn't raining". These nodes may feed into a utility node which computes a utility based on the values of these nodes. For instance, if it is raining and we don't bring an umbrella, we will have a very low utility, compared to when it isn't raining and we don't bring an umbrella, for which we will have a high utility.

We want to choose actions that maximize the expected utility given observed evidence.

The general process for action selection is:

- instantiate all evidence
- set action node(s) each possible way
- calculate the posterior for all parents of the utility node, given the evidence
- calculated the expected utility for each action
- choose the maximizing action (it will vary depending on the observed evidence)

This is quite similar to expectimax/MDPs, except now we can incorporate evidence we observe.

TODO decision network example

### Value of information

More evidence helps, but typically there is a cost to acquiring it. We can quantify the value of acquiring evidence as the _value of information_ to determine whether or not it is more evidence is worth the cost. We can compute this with a decision network.

The value of information is simply the expected gain in the maximum expected utility given the new evidence.

For example, say someone hides 100 dollars behind one of two doors, and if we can correctly guess which door it is behind, we get the money.

There is a 0.5 chance that the money is behind either door.

In this scenario, we can use the following decision network:

$$
\begin{aligned}
\text{choose door} \to U \\
\text{money door} \to U
\end{aligned}
$$

Where $\text{choose door}$ is the action variable, $\text{money door}$ is the random variable, and $U$ is the utility node.

The utility function at $U$ is as follows:

| choose door | money door | utility |
|-------------|------------|---------|
| a           | a          | 100     |
| a           | b          | 0       |
| b           | a          | 0       |
| b           | b          | 100     |

In this current scenario, our maximum expected utility is 50. That is, choosing either door $a$ or $b$ gives us $100 \times 0.5 = 50$ expected utility.

How valuable is knowing which door the money is behind?

We can consider that if we know which door the money is behind, our maximum expected utility becomes 100, so we can quantify the value of that information as $100-50=50$, which is what we'd be willing to pay for that information.

In this scenario, we get _perfect information_, because we observe the evidence "perfectly" (that is, our friend tells us the truth and there's no chance that we misheard them).

More formally, the value of perfect information of evidence $E'$, given existing evidence $e$ (of which there might be none), is:

$$
\text{VPI}(E'|e) = (\sum_{e'} P(e'|e) \text{MEU}(e, e')) - \text{MEU}(e)
$$

Properties of VPI:

- nonnegative: $\forall E', e: \text{VPI}(E'|e) \geq 0$, i.e. is not possible for $\text{VPI}$ to be negative (proof not shown)
- nonadditive: $\text{VPI}(E_j, E_k|e) \neq \text{VPI}(E_j|e) + \text{VPI}(E_k|e)$ (e.g. consider observing the same evidence twice - no more information is added)
- order-independent: $\text{VPI}(E_j,E_k|e) = \text{VPI}(E_j|e) + \text{VPI}(E_k|e,E_j) = \text{VPI}(E_k|e) + \text{VPI}(E_j|e,E_k)$

Also: generally, if the parents of the utility node is conditionally independent of another node $Z$ given the current evidence $e$, then $\text{VPI}(Z | e) = 0$. Evidence has to affect the utility node's parents to actually affect the utility.

What's the value of _imperfect_ information? Well, we just say that "imperfect" information is perfect information of a noisy version of the variable in question.

For example, say we have a "light level" random variable that we observe through a sensor. Sensors always have some noise, so we add an additional random variable to the decision network (connected to the light level random variable) which corresponds to the sensor's light level measurement. Thus the sensor's observations are "perfect" in the context of the sensor random variable, because they are exactly what the sensor observed, though technically they are noisy in the context of the light level random variable.

### POMDPs

Partially-observed MDPs are MDPs in which the states are not (fully) observed. They include _observations_ $O$ and an _observation function_ $P(o|s)$ (sometimes notated $O(s,o)$; it gives a probability for an observation given a state).

When we take an action, we get an observation which puts us in a new _belief_ state (a distribution of possible states).

With POMDPs the state space becomes very large because there are many (infinite) probability distributions over a set of states.

As a result, you can't really run value iteration on POMDPs, but you can use approximate Q-learning or a truncated (limited lookahead) expectimax approach to approximate the value of actions.

In general, however, POMDPs are very hard/expensive to solve.

## HMMs and Particle Filtering

Often we have a sequence of observations and we want to use these observations to learn something about the underlying process that generated them. As such we need to introduce time or space to our models.

### Markov models

We can consider a Markov model as a chain-structured Bayes' Net, so our reasoning there applies here as well.

TODO Markov model chain example

Each node is a state in the sequence and each node is identically distributed (stationary) and depends on the previous state, i.e. $P(X_t|X_{t-1})$ (except for the initial state $P(X_1)$). This is essentially just a conditional independence assumption (i.e. that $P(X_t)$ is conditionally independent of $X_{t-2}, X_{t-3}, \dots, X_1$ given $X_{t-1}$).

The parameters of a Markov model are the _transition probabilities_ (or _dynamics_) and the initial state probabilities (i.e. the initial distribution $P(X_1)$.

This is the same as a MDP transition model, but there's no choice of action.

Say we want to know $P(X)$ at time $t$. A Markov model algorithm for solving this is the _forward algorithm_, which is just an instance of variable elimination (in the order $X_1, X_2, \dots$). A simplified version:

$$
P(x_t) = \sum_x_{t-1} P(x_t|x_{t-1}) P(x_{t-1})
$$

Assuming $P(x_1)$ is known.

$P(X_t)$ converges as $t \to \infty$, and it converges to the same values regardless of the initial state. This converged distribution, independent of the initial state, is called the _stationary distribution_. The influence of the initial state fades away as $t \to \infty$.

Formally, the stationary distribution satisfies:

$$
P_{\infty}(X) = P_{\infty+1}(X) = \sum_x P_{t+1|t}(X|x) P_{\infty}(x)
$$

Gibbs sampling is essentially a Markov model (hence it is a _Markov Chain_ Monte Carlo method) in which the stationary distribution is the conditional distribution we are interested in.

### Hidden Markov Models

A _hidden_ Markov model is one in which we don't directly observe the state. That is, there is a Markov chain where we don't see $X_t$ but rather we see some evidence/outputs/effects/etc $E_t$.

For example, imagine we are in a windowless room and we want to know if it's raining. We can't directly observe whether it's raining, but we can see if people have brought umbrellas with them.

An HMM is defined by:

- the initial distribution $P(X_1)$
- the transitions $P(X|X_{-1})$
- the emissions $P(E|X)$ (the probability of seeing evidence given the hidden state)

We introduce an additional conditional independence assumption - that the current observation is independent of everything else given the current state.

### Filtering/Monitoring

_Filtering_ or _monitoring_ is the task of tracking the belief state (the distribution) $B_t(X) = P_t(X_t | e_1, \dots, e_t)$ over time, as new evidence is observed.

We start with $B_1(X)$ with some initial setting (typically uniform), and update as new evidence is observed/time passes.

## Reference

CS188: Artificial Intelligence. Dan Klein, Pieter Abbeel. University of California, Berkeley (edX).
