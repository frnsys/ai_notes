notes from:

<https://www.youtube.com/watch?v=o_ZuWbX-CyE>  

May 21, 2010

Human Behavioral Biology

Lecture 22: Emergence and Complexity

Professor Robert Sapolsky

Stanford University

First see: [Chaos and Reductionism](`Chaos and Reductionism.md`)

Cellular automata are a good way of seeing how the systems described in the previous lecture (see link above) manifest.

![](assets/Emergence%20and%20Complexity/4c0da44a58aae6d72d3c63562a07174a.png)  

They operate on simple rules which determine how the following generation is derived from the current generation (each generation is a row of cells, which are binary; they can be on/filled in or off), and give rise to periodic or aperiodic deterministic systems. They begin with an initial configuration of cells for the starting generation. Minor differences in this starting state can lead to wildly different patterns (the butterfly effect).

Through these systems you can get very structured patterns. But in most cases, the patterns break down and stop after awhile - that is, the majority of the time, these systems go extinct.

There are few starting states that succeed (don't go extinct), and the mature states they produce all look very similar to each other. That is, a lot of different starting conditions can end up in a similar state. This is _convergence_. This convergence makes it very difficult to learn of the starting state from these later generations.

A slight change in a starting state can result in something totally different (an example of the butterfly effect).

Shifting the starting state of the above example gives something similar, but different:

![](assets/Emergence%20and%20Complexity/902e51db459c05e471081c81eda5be72.png)  

And another starting state leads to a very stable, unchanging sequence:

![](assets/Emergence%20and%20Complexity/ff3bbd25a9296eee646c1ccbcc3aa4f9.png)  

(this is a bad image since the camera operator didn't focus on the screen, but you get the idea)

Another shift in starting state dies out very quickly and just leads to empty rows after a few generations.

And changing the starting states again may lead back to those patterns:

![](assets/Emergence%20and%20Complexity/3bea256a3733dc1ea76602183cc83c55.png)  

In this case, it came very close to becoming extinct (where one generation was almost entirely empty).

The point here is that, given a starting state, you can't say a priori that, oh this will lead to a structured pattern, or this will go extinct, etc. You have to see how it plays out/go through it step by step.

A pattern you see in these systems is that an asymmetric starting state often leads to more dynamic patterns than symmetric starting states.

You see these in living systems, where unrelated organisms in distance locations but similar environments converge on a few similar forms and adaptations (and others go extinct).

So _emergence_ is the idea that, in these complex systems, a small set of simple rules can code for the vast diversity, for example, that we see in the natural world.

An example of a neural network:

![](assets/Emergence%20and%20Complexity/neuralnetwork.png)  

On the bottom layer you have Hubel-and-Wiesel-type neurons, which can recognize a specific Impressionist painter's work (of course, these neurons do not actually exist). Within the neural network, you can see that the neuron that all of these bottom neurons feed into is capable of recognizing Impressionist paintings in general, whereas the two flanking neurons which have only two Impressionist neurons feeding into them is also capable of this, but less accurate due to its deficiency of information.

As an example, say you're trying to remember the name of that painting movement, and you remember, Degas was one of them, and the Degas neuron lights up, which lights up the Impressionism neuron, but maybe only a little, and you remember, Monet was one too, and that neuron lights up and further activates the Impressionism neuron, and then you remember, Renoir was one too, and that neuron lights up and further activates the Impressionism neuron, and then you remember - oh, it's Impressionism.

So there's this network of neurons which code for and connect all sorts of experiences and memories and so on, which is also what allows us to relate similar ideas together, since they are connected in some way. Creativity could be framed as a broader network of connections which allows more disparate relations to be made.

When you get to layers of the cortex beyond the first few, you get to the "associational" cortex, and you start to see this in action. You see multimodal, multi-responsive neurons where a lot of different things stimulate them embedded in these networks.

Alzheimer's is the weakening of these networks; that is, it is the degradation of these links. Alzheimer's patients don't necessarily forget things, they have a very hard time remembering things via association. You can give a lot of clues to the answer of something, such as "Who is the president?", which will activate neural networks, but will fail to activate the one coding for the answer to that question. But once they get the answer, they recognize it and are able to recall things connected to that answer - that network was there, but it was disconnected from others, and thus much harder to activate.

Going back to the bifurcation of the circulatory system, instead of having genes encode for each branching, there are "fractal" genes, i.e. "scale free" genes, which code for simple _rules_ around when bifurcation happens.

Consider this: in the human body, no cell is more than five cells away from a blood vessel. But the circulatory system is less than 5% of the body. It doesn't seem to make sense that it could be everywhere but occupy so little space.

If you look at a fractal like Koch's snowflake:

![](assets/Emergence%20and%20Complexity/057a64c2aadc008d6ba749efba2ee11a.jpeg)

via <http://www.rhynelandscape.com/wp-content/uploads/2013/06/Koch-Snowflake-Fractal.jpg> 

You can see how fractals can accomplish the impossible. With Koch's snowflake, you take a triangle, and halfway through each side you create another triangle, and you apply this rule to the resultant triangles ad nauseum. What you see is that, paradoxically, the resulting fractal has an infinite perimeter but occupies a finite area.  

Some things don't even need to be explicitly coded for, through rules or otherwise. Rather, they can emerge as results of the properties or physical constraints of the system which they are in.

The "wisdom of the crowds" phenomenon is an example of emergence, where the aggregate familiarity or expertise of a crowd provides a solution better than one that any individual of the crowd can provide. Note that for this to work the crowd must have some expertise around the problem being solved.

Ants are an example of emergence. If you put one ant on a table and watch it, it looks like it just wanders aimlessly. If you put down a few, it still looks like just a small aimless group. It isn't until you put maybe thousands of ants that more complex coordinated behaviors start to emerge. And this organized behavior spontaneously emerges as a result of simple _nearest-neighbor_ rules that the ants use. No single ant "knows" what the blueprint of the complete organized system is, nor does it need to know, since its behaviors are independent of it. That organization is a natural consequence of these simple rules.

The Traveling Salesman problem is a very difficult problem for which there is no formal mathematic solution. It can be "solved" by using very powerful computers and brute-forcing it. But a more reasonable approach is to use "swarm intelligence", which involves creating generations of hundreds of thousands of "ants" each wt

ith the rule where they randomly go to a city and to another city, etc, but they also leave a pheromone trail. The addition the rule is that, if a pheromone trail is encountered, follow that trail.

The amount of pheromone available to each ant is limited, which means that shorter paths will be thicker (since it runs out over longer paths). Further, the pheromone trails evaporate after awhile. So thicker paths will also stick around longer, and thus have a greater likelihood of being encountered and followed (which then further reinforces them). After some generations the ants eventually solve this problem.

This is actually how ants work in the real world too, to find the shortest paths to food.

Note that this is _not_ wisdom of the crowds because the ants don't have any familiarity with the problem, they probably aren't even conscious that they're solving a problem. It's an emergent feature.

With bees, a similar behavior emerges. When a bee finds a good resource, it comes back to the colony and does a dance. The angle of the dance and amount of wiggling indicates the angle to the resource and the duration of flight to get there. But there's another aspect - the duration of the dance. The higher quality the resource, the longer the bees perform the dance. Like with the ants encountering the pheromone trails, there are random encounters within the colony, where a bee runs into a bee doing a dance and then flies out to the specified resource. The longer the dance, the higher the probability another bee happens upon it, thus, higher quality resources have greater amounts of bees going to them, and in this way, an optimization feature emerges from the rules of the dance.

Attraction and repulsion are also a common form of rules. For example, cities tend to organize based on simple rules about how some establishments attract and/or repulse other establishments.

The emergence of amino acids from primordial ooze is another example - molecules operate on basic rules of attraction and repulsion. Several experiments have shown that rational structures spontaneously form after awhile from a "soup" of uniformly distributed molecules.

It seems a large reason for human's intellectual prowess is due to quantity. The neurons we have are not much different than those in other organisms, even a fly, but we outperform these other organisms because the sheer number of neurons we have provide greater opportunity for more complex emergent behavior. When examining the 2% genetic difference between chimps and humans, there was surprisingly little difference around brains; the main difference was around cell division and increasing the quantity of neurons.

