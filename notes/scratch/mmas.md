# Massively Multi-Agent Systems (MMAS)

[A Distributed Platform for Global-Scale Agent-Based Models of Disease Transmission](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3898773/). Jon Parker, Joshua M. Epstein.

- parallel & distributed
- workload distributed across two layers:
    - a node layer (each node is a JVM)
    - each node distributes its workload across threads
- the population is distributed across computing resources
    - each partition is called a ModelBlock (MB)
    - each MB represents people who live close to one another in geographic space

the optimal distribution strategy may be problem-dependent, but you can usually do well using either a round-robin or random allocation scheme.

you can also implement dynamic load balancing (e.g. moving MB from one node to another) but there is a very high overhead. Random or round-robin allocation can help avoid dramatic load imbalances.

not all agents are simulated - only active ones are.

>  Consider this analogy: a row of 6 billion contiguous dominoes is given. Some domino (the index case) is toppled, and a contagion of falling dominoes ensues. You wish to simulate the progress of this wave. It would be absurd to loop through the entire list of all 6 billion dominoes at every simulated time-step. Each trip through the list would examine billions of dominoes when only a handful are changing state at any one time. It is far more efficient to maintain a list of active (i.e., falling) dominoes and operate only on this set. Proper implementation of this active set modeling scheme requires that each domino correctly determine which dominoes it will effect when active. This allows active dominoes to promote other dominoes to the active set at the appropriate time. Predictably, when a domino completes its fall it will remove itself from the active set.

each ModelBlock maintains a priority queue where events are placed (e.g. events where an agent does something, or something happens to an agent, etc). It goes through this queue, executing events as they come.

---

Agent Based Modeling, Large Scale Simulations. Hazel R. Parry.

| Element                    | Least complex ->                              | Most complex                                                                         |
|----------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------|
| Spatial structure          | Aspatial or lattice of sells (1d, 2d, or 3d+) | Continuous space                                                                     |
| Internal state             | Simple representation (boolean true or false) | Complex representation (many states from an enumerable set) or fuzzy variable values |
| Agent heterogeneity        | No                                            | Yes                                                                                  |
| Interactions               | Local and fixed (w/in a neighborhood)         | Multiple different ranges and stochastic                                             |
| Synchrony of model updates | Synchronous update (e.g. time steps)          | Not synchronous: asynchrony due to state-transition rules or b/c event-driven        |

Also, agents may also interact with the environment (in addition to each other).

Data may be mapped to different nodes in a few ways:

- _cyclic mapping_ involves cycling through each node and assigning array elements to each node in turn (same as round-robin from above)
- _block mapping_ involves partitioning array elements as evenly as possible into blocks of consecutive elements

The right way depends on the application and it can greatly help with load balancing.

Since computational demands may change over time during the simulation (perhaps some agents become more active, for exaple), so dynamic load balancing can help.

One method is _Adaptive Actor Architecture_, where agents are redistributed as nodes become overloaded according to their communication patterns (moved to be closer to agents they communicate with frequently). This introduces additional overhead so should only be applied where a lot of communication is going on between particular agents.

Asynchronous updating may be tricky because some nodes may block other nodes, we need to ensure that messages are received in the right order.

A deadlock is when two or more processes are waiting for communication from one of the other processes. This halts the entire simulation.

One way to avoid this is to use non-blocking message passing, i.e. work on a node continues even if the message hasn't transmitted yet.

---

[Time Management in High Level Architecture](http://www.cc.gatech.edu/computing/pads/PAPERS/Time_mgmt_High_Level_Arch.pdf). Richard M. Fujimoto.

Different kinds of time:

- _physical time_: the time of the system being modeled, e.g. 2/14/2015, 5:30pm, etc.
- _simulation time_: the simulator's representation of time, e.g. 10 time steps
- _wallclock time_: the "real world" time as the simulation is executed, e.g. we started running it at 12pm, it ran for two hours

A challenge in "federated simulation" (simulation with multiple nodes) is keeping events synchronized across nodes, i.e. such that they are all executed/interpreted in the correct order.

The best way to do this depends on how time works in the simulation:

- _event driven_: use timestamps and process in timestamp order (TSO, as opposed to receive order, RO)
- _time stepped_: centrally keep track of when nodes have completed the current time step then simultaneously call the next time step across nodes

With timestamp order, a node must be guaranteed that it will receive no more messages containing an earlier timestamp, so that it knows it can go ahead processing the queued events without needing to rollback. Alternatively, you can use _optimistic event processing_ which processes messages in timestamp order without guarantee there are no earlier timestamped messages pending, but has some means of recovering if there is some conflict (i.e. rollback). A rollback can get complicated, as it may require the node that is rolling back to recall events it has already sent out, which may trigger further rollbacks on other nodes, etc (this is called the Time Warp method).

_Simultaneous events_ (i.e. those with identical timestamps) can be executed in an arbitrary order or according to some tie-breaking field in the event.

Time step simulations are easier because they can pause between timestamps to wait for all pending events to be sent to their destinations.