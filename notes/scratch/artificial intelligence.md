
# Artificial Intelligence

My notes from MIT 6.034 (Fall 2010): Artificial Intelligence, taught by Patrick H. Winston.

---

## Introduction

### AI is about:

> algorithms enabled by
> constraints exposed by
> representations that support
> models targeted at
> thinking and perception and action
> tied together in loops.

Questions to ask:
How to best represent a problem? How to to model systems to solve problems, explain the past, and predict the future?

AI is not purely about reasoning. Perception, language, etc are also important for problem solving.

Example: How many African countries does the equator cross?

Without prior knowledge, you can't purely reason your way to the correct answer. But upon seeing a map, the answer is immediately clear.

Imagination is crucial to problem solving as well. With imagination you are able to simulate events which may not have ever happened or may never have experienced and may not even be possible. All by integrating existing knowledge in different ways.

### The Rumpelstiltskin Principle

The "_Generate & Test_" method is a problem solving method where you:

1. Generate some potential solutions
2. Test the solutions and see which one is correct

It's very simple - what's important is that this method, which was previously intuitive and thus likely not given much thought - now has a _name_, which allows us to talk about it.

Giving something a name gives you some power over it in that now you can examine and discuss it. This is the _Rumpelstiltskin Principle_.

### "Simple" =/= "Trivial"

"Trivial" implies that something is simple _and_ of little worth or interest.

Just because something is simple does not mean it is not powerful. Often the opposite, in fact.

---

## Reasoning - Goal Trees and Problem Solving

Problem reduction has other names: "and/or tree" or "goal tree".
It refers to decomposing a problem into simpler problems, which can be represented as a goal tree, which details the possible paths to solving the original problem. In addition regular nodes, nodes in the tree can be "and", where all branches must solved, or "or" nodes, where only one of the branches needs to be solved.

![A goal tree](assets/goal_tree.svg)

You need some method for picking a branch at or nodes, but regardless of the method, it is possible that the selected branch leads to a dead end and that one of the alternative paths should be attempted.

Problems can be reduced via "safe" transformations, which reliably lead to the true solution, or by "heuristic" transformations, which are approximations which do not necessarily lead to the true solution. But sometimes they are necessary for reducing the problem further. A system using this approach requires *knowledge* about what transformations are available.

---

## Goal Trees and Rule-Based "Expert" Systems

### A block-moving program

A block-moving program was demoed, which was able to arrange blocks on a table according to a desired state specified by the user. It generates apparently complex behavior using a few simple rules and procedures along with recursion. It is even capable of stating the reasons behind its actions - answering questions like "why did you move block B1 on top of B2?" or "how did you clear the top of block B3?" - this apparent reasoning is also a result of these simple rules.

For example, say we have the current state:

     ______   ______
    |      | |      |
    |  BX  | |  BY  |
    |______| |______|
    |      | |      |
    |  B1  | |  B2  |
    |______|_|______|___________________________

We want the program to place block B1 onto B2.

This problem can be reduced to a simpler set, which then forms the solution procedure.

For example:

- Move B1 on top of B2.
    - Pick up B1.
        - Clear the top of B1.
            - Find space to place BX.
            - Pick up BX.
            - Place BX on the empty space.
    - Place B1 on top of B2.
        - Clear the top of B2.
            - Find space to place BY.
            - Pick up BY.
            - Place BY on the empty space.

This is a goal tree (in particular, an "and tree", since all branches must be completed).

The program can answer "why" questions by moving up the tree: "Why did you clear the top of B1?", with the answer "So I could pick up B1."
It can answer "how" questions by moving down the tree: "How did you pick up B1?", with the answer "By clearing the top of B1."

### Herb Simon's Ant

Imagine you look at the path of an ant.

![The complex path of an ant](assets/ant_01.svg)

It looks complex. You may attribute that complexity to the ant.

But when you look closer, you see the ant is just avoiding pebbles on the beach.

![The complex path from simple rules](assets/ant_02.svg)

The apparent complexity of the behaviour is a consequence of the complexity of the environment.

That is,

$$
complexity(behavior) = max(complexity(program), complexity(environment))
$$

or

$$
complexity(behavior) = max(complexity(program), complexity(problem))
$$

### Rule-based "expert" systems

Around the mid 1980's, rule-based "expert" systems emerged, spurred by interest in commercial applications of AI.

The general approach is about encapsulating knowledge into simple rules (i.e. if-this-then-that).

"Expert" is in quotes because it is more of a marketing term - it can't really be said that these systems are experts in the same way we'd call a human an expert.

There are two main systems, both of which are _deductive_. That is, they work with facts to produce new facts by proving them.

#### Forward-Chaining rule-based expert systems

This uses rules (facts) to get to some conclusion, which can be represented as a goal tree.

For example, if we have an expert system for identifying animals, given some features we may ask "what animal is this?".

![An example forward-chaining rule-based system](assets/rule_based.svg)

Note: $Rn$ denotes rule $n$. Also notice there are some "and" nodes.

#### Backward-chaining rule-based expert systems

This is similar to forward-chaining, but moves backwards from a hypothesis and proves it via facts.

For example, if again we have an expert system for identifying animals, given an animal we may ask, "is this a cheetah?"

#### Knowledge engineering

The selection and extraction of knowledge from human experts, and their subsequent representation as rules in the system, is known as _knowledge engineering_.

There are a few heuristics regarding knowledge engineering:

- Look at specific cases and try to generalize to rules.
- Ask about things which appear the same but are handled differently. You may learn new distinctions and expand your domain vocabulary.
- Build the system and see where it fails - you're missing a rule for that case!

Knowledge engineering is relevant to _human_ learning as well! That is, those heuristics can help you become a better learner in general.

---

## Learning

Two kinds of learning, broadly:

- _Regularity_ - learning based on observations of regularity (patterns), involves processing a lot of information
    - Nearest neighbors
    - Neural nets
    - Boosting
- _Constraint_
    - One-shot learning
    - Explanation-based learning

---

## Learning: Near Misses, Felicity Conditions

Say we are learning what an arch is. We are presented with a few configurations and told what is and isn't an arch.

The first example is our initial model. We can graphically represent the relationship between the bricks as shown in the graph.

The second example is a _near miss_ - it is not an arch but it only differs in one aspect (the other two bricks do not support the other block). From this difference we learn that an arch _requires_ that support relationship; thus this heuristic is called a _require link_.

![A near miss](assets/arch_01.svg)

We are presented with another near miss - this time the lower two blocks have a touch relationship, and learn that these two blocks must not touch - thus we have a _forbid link_ heuristic.

![A near miss](assets/arch_02.svg)

Now we are presented with an example (as opposed to a near miss) - this is still an arch. The brick on top is colored differently, so we encode the color property and _extend_ the valid values it can take on (hence this is called the _extend set_ heuristic).

![An example](assets/arch_03.svg)

We are presented with another example in which we still have an arch when the top brick is blue. We're going to assume our universe has only three colors - red, white, blue - so the top brick can effectively have any color. The asterisk (wildcard) indicates it can take on any value. This heuristic is called a _drop link_ because technically we drop the color link altogether, though we keep it here to remember that we've learned it.

![An example](assets/arch_04.svg)

We are presented with a final example in which the top brick is now a wedge, and we still have an arch. Again, we encode the type of block in our representation, which we hadn't been before. We're going to go a bit further and assume that bricks and wedges belong to the category of blocks, which further belong to the category of toys, and that any of these are valid tops to form an arch. This heuristic is called "_climb tree_".

![An example](assets/arch_05.svg)

Examples allow you to _generalize_, whereas near misses allow you to _specialize_.

Because this kind of learning is accomplished off of a very small number of examples, it is called _one-shot_ learning.


### Misc. tips

An important part in learning is knowing how to describe things, how to represent things. When you a learning something, talking yourself through can be helpful because it forces you to try and describe things explicitly.

How to sell your ideas:

- have some visual symbol for people to associate with the idea, as a visual handle
- have some slogan as a verbal handle
- have something surprising
- have something salient
- have a story

---

## Representations: Classes, Trajectories, Transitions

One of the unique features of humans is that we can think in stories. We can be presented with a snippet of something and expand it out into a broader narrative without really trying, bringing in assumptions and other knowledge and drawing out implications and interpretations.

With what language do we think? What is our "inner language"? Having answers for these questions may shed some light on how we can emulate our intelligence with machines.

### Semantic Nets

You can create a _semantic network_ of relationships, for instance, amongst characters in a story and how they relate or interact with one another.

The edges between nodes are called _combinators_, and these edges can be connected to each other as well - this process of connecting edges together is called _reification_. For instance, With the network `Macbeth->kills->Duncan` and `Macbeth->murders->Duncan`, you may have an edge from `kills` to `murder` because one implies the another.

With a complex semantic network you may want to identify parts of the network, which you can do so with _frames_, which is a _localization_ layer on top of the semantic net. A frame specifies what kind of features describe it. So you can put a frame over the murder relation, like `Agent:Macbeth, Victim:Duncan`. You could also include a "weapon" feature or something.

There is, however, a problem of _parasitic semantics_, that is we project meaning onto the machine which it is not able to understand, i.e. further implications from the existing net. For the relation `Macbeth->murders->Duncan`, we as humans may recognize that there is some implication of motive, that someone murders someone not without reason - but a machine doesn't know that.

These concepts: combinators, reification, and localization are some components which seem to underlie how we think about things.

### Classes

We classify things on a scale of general -> basic -> specific. For instance, general might be "a tool", basic might be "a hammer", and specific might be a "ball peen hammer". A lot of our knowledge is in the "basic" range. General might be too broad to know many interesting things, and specific may just be variations or nuance on knowledge in the basic category.

### Transition

Consider a car which crashes into a wall:

|                  | Before hit   | During hit | After hit       |
|------------------|--------------|------------|-----------------|
| Speed of car     | not changing | disappears | does not appear |
| Distance to wall | decreasing   | disappears | does not change |
| Condition of car | not changing | changes    | does not change |

As the car approaches the wall, its speed does not change, its distance to the wall decreases, and its condition does not change.
When it hits the wall, it's speed disappears (it stops), its distance to the wall disappears (it has reached the wall), and its condition changes (now it's damaged).
After the hit, there is no speed (it has already stopped), its distance to the wall has not changed, and its condition is still damaged.

We have a language of change - in particular of change _causing_ change, which we call _transition_ - which seems to consist of ten key concepts:

- decreasing
- increasing
- changing
- appears
- disappears

And the not variation of each.

### Trajectory

An object may be moving along a trajectory, having started at a source ("from") and going towards a destination ("to")  Its motion may be caused by an agent ("by"), perhaps with a co-agent ("with"), perhaps using an instrument ("with"), perhaps involving some conveyance ("by"), perhaps for some beneficiary ("for").

This set of language describes a _trajectory frame_ ("frame" as in the semantic net mentioned before); if the object is not moving then it may still involve these things but is referred to as a _role frame_ instead.


### Bringing these together

Consider the sentence:

> Pat comforted Chris.

We can construct a role frame for this sentence:

- Agent: Pat
- Action: ? (we don't have a firm image of what the actual action is, i.e. what Pat did to comfort Chris)
- Object: Chris
- Result: Transition frame (see below)

In terms of the action - we may _hallucinate_ more than we are actually being told, e.g. "comfort" may invoke images of Pat hugging Chris or something, but none of that is really known or unambiguously implied by the term.

The result is a transition frame, which we can construct as:

- Object: Chris
- Mood: increases

You could represent this sentence in a different way. You could frame is as a trajectory:

- Agent: Pat
- Object: comfort
- Destination: Chris

The important thing to recognize here is that things can be represented and conceptualized in many different ways; this is how we can creatively solve problems.

There is some implication here of a _sequence_ of events, which also seems crucial to how we think - that is we often think about things with a concept of sequence. For instance, it's very hard to start telling a story in the middle, it's much easier if you start from the beginning or at least the beginning of a scene.


### Story Libraries

As part of our knowledge we have a library of "stories" which are frames we can use to make sense of and describe things. These frames are hierarchical: we may have an "event" frame, in which we might describe a time and a place, and a child frame might be a "disaster", in which we might describe the number of fatalities and costs of the damage, and a child frame of the "disaster" frame might be the "earthquake" frame, in which we might add on new descriptors for "magnitude" and the "fault line", and it might have a sibling "hurricane" frame, in which instead we might add on descriptors for "category" and "name", and so on.


---

## Architectures: GPS, SOAR, Subsumption, Society of Mind

### General Problem Solver (GPS)

- Start in some current state $C$
- Want to get to some goal state $S$
- Measure the symbolic difference $d$ between these points
- Select some operator $o$, based on $d$ which will move you from your current state to some intermediary state $I$, between $C$ and $S$.
- Then you have a distance $d_2$ from $I$ to $S$ which used to come up with another operator $o_2$ which takes you to another intermediary position $I_2$.
- And so on

For example,

- You're at school ($C$) and you want to visit home ($S$).
- If the distance $d$ is sufficiently large, the first operator might be taking an airplane.
- But you can't just get on an airplane from where you are so there might be another intermediary step in which you take a car to the airport. And before that you need to walk to the car. And so on. So there's some recursion here.

The idea is that this abstract representation of problem solving can be useful for modeling intelligence.


### SOAR (State Operator and Result)

Consists of:

- A long-term memory
- A short-term memory
- Connections to the outside world (e.g. a vision and an action system)

Most of the work happens in the short-term memory, meant to emulate those systems in humans.

This system uses a lot of assertions and rules with an elaborate preferences subsystem for resolving situations with two competing rules.

SOAR also defines problem spaces which are then searched through to find a solution. This can be recursive in that if it can't think of what to do next, that becomes its own problem space.

The key concept here is around getting things into a symbolic form.


### Subsumption

Instead of having individual systems for dealing with, for instance, vision, reasoning, and action, you can organize behaviors into layers of abstraction: e.g. `avoid->wander->explore->seek` each with its own vision, reasoning, and action systems which can borrow from the layer beneath without affecting it (the benefit of abstracted layers).

The proposed system has no representation, so instead of constructing models about the world, you just react to the world immediately. For example, you don't need to have a map of a room in memory, you just react to objects as you encounter them. The mechanisms in this system are just finite state machines.

The Roomba robot operates off of these principles.


### Emotion Machine (Marvin Minsky)

Includes several layers of thinking:

- Self-conscious thinking
- Self-reflecting thinking
- Reflective thinking
- Deliberative thinking (corresponds to SOAR and GPS)
- Learned reaction (corresponds to subsumption)
- Instinctive reaction (corresponds to subsumption)


### Genesis system

Centered around language, which has two roles:

- guide, marshal, and interact with the perception systems
- enable the description of events
    - which enables you to tell and understand stories
        - which enables you to understand macro and micro culture


---

## Final lecture

### What does AI offer that is different from other fields trying to understand intelligence

- A language for procedures (e.g. programming metaphors which are unavailable to other fields interested in psychology)
- New ways to make models (i.e. programs)
- Enforced details (i.e. when you have to implement something, you can't neglect details)
- Opportunities to experiment (e.g. you can't remove some knowledge from a person to see how they do without it, but you can with a program)
- Upper bounds (e.g. what's the minimum necessary for a program to understand a story?)

### How do you do artificial intelligence (and engineering/science in general)

- Define or describe a competence
- Select or invent a representation, picking one which allows you to...
- Understand constraints and regularities
- Select methods
- Implement and experiment

And avoid "mechanism envy", e.g. sticking to a favorite method without thinking about if it's appropriate.

### Recommended follow-up courses

- 6.868: Minsky, Society of Mind
- 6.863: Berwik, Language
- 6.048: Berwik, Evolution
- 6.945: Sussman, Large Scale Symbolic Systems
- 6.803: Winston: Human Intelligence Enterprise

See <http://ocw.mit.edu/courses/find-by-number/>

### Patrick Winston's powerful ideas

- The idea of powerful idea
- The right representations make you smarter
- Sleep makes you smarter
- You cannot learn unless you almost know
- We think with mouths, eyes, and hands
- The Strong Story Hypothesis: "The mechanisms that enable us humans to tell and understand stories are the evolutionary augmentations that separate our intelligence from that of other primates."
- All great ideas are simple (Be careful about confusing simplicity with triviality)
