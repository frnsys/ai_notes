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

## Reference

CS188: Artificial Intelligence. Dan Klein, Pieter Abbeel. University of California, Berkeley (edX).
