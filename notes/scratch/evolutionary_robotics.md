notes from University of Vermont's [Evolutionary Robotics course](https://www.youtube.com/playlist?list=PLAuiGdPEdw0jySMqCxj2-BQ5QKM9ts8ik) (CS206, Spring 2016), taught by Josh Bongard

General approach:

1. create a task environment
2. create the robot
3. create a neural net as the robot's "brain"
4. use an evolutionary algorithm to optimize the neural net for the task

Some previously evolved behavior for one purpose may later find new use for another purpose. This is called _exaptation_.

_Embodied_ cognition involves learning through interaction - the agent manipulates its environment, observes the consequence, and learns from that. A "body" here refers to a tool that can affect and be affected by the world. The general idea behind embodied cognition is that the way you process information is affected by your body.

_Situated_ cognition is similar to embodied cognition, which says that the way you process information is affected by the fact that you are physically situated in the world (i.e. you have a bunch of sensors).

A _complete agent_ is one that is both situated and embodied. There are some important properties that distinguish complete agents from other kinds of agents:

- they are subject to the laws of physics
- they generate sensory stimulation (through behavior)
- they affect the environment (through behavior)

|              | Disembodied      | Embodied          |
|--------------|------------------|-------------------|
| Not situated | Computers        | Industrial robots |
| Situated     | Embedded devices | People, robots    |

