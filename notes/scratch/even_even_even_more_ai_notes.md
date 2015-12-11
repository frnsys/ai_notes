## Reactive planning

Reactive planning is an approach to planning in which no sequence of actions is planned in advance; rather, new actions are chosen on-the-fly. Thus they are well suited for real-time environments.

## Multi-task and multi-scale problems

A _multi-task_ domain is an environment where an agent performs two or more separate tasks.

A _multi-scale_ domain is a multi-task domain that satisfies the following:

- multiple structural scales: actions are performed across multiple levels of coordination
- interrelated tasks: there is not a strict separation across tasks and the performance in each tasks impacts other tasks
- actions are performed in real-time

More generally, multi-scale problems involve working at many different levels of detail.

For example, an AI for an RTS game must manage many simultaneous goals at the micro and macro level, and these goals and their tasks are often interwoven, and all this must be done in real-time.

## Case-Based Goal Formulation

With _case-based goal formulation_, a library of cases relevant to the problem is maintained (e.g. with RTS games, this could be a library of replays for that game). Then the agent uses this library to select a goal to pursue, given the current world state That is, the agent finds the state case $q$ (from a case library $L$) most similar to the current world state $s$:

$$
q = \argmin_{c \in L} \text{distance}(s, c)
$$

Where the distance metric may be domain independent or domain specific.

Then, the goal state $g$ is formulated by looking ahead $n$ actions from $q$ to a future state in that case $q'$, finding that difference, and adding that to the current world state $s$:

$$
g = s + (q' - q)
$$

The number of actions $n$ is called the _planning window size_. A small planning window is better for domains where plans are invalidated frequently.

## References

- Integrating Learning in a Multi-Scale Agent. Ben G. Weber. 2012.