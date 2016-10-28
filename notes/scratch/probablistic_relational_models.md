> despite their success, Bayesian networks are often inadequate for representing large and complex domains. A Bayesian network for a given domain involves a prespecified set of random variables, whose relationship to each other is fixed in advance. Hence, a Bayesian network cannot be used to deal with domains where we might encounter a varying number of entities in a variety of configurations. This limitation of Bayesian networks is a direct consequence of the fact that they lack the concept of an "object" (or domain entity). Hence, they cannot represent general principles about multiple similar objects which can then be applied in multiple contexts.
>
> _Probabilistic relational models (PRMs)_ extend Bayesian networks with the concepts of objects, their properties, and relations between them.

PRMs specify a template for a probability distribution over some collection of entities and relationships; this template has two components:

- a relational component (the relational schema; e.g. `A VERB B`)
- a probabilistic component

TODO finish reading the chapter :)

## References

- [Probabilistic Relational Models](http://ai.stanford.edu/~koller/Papers/Getoor+al:SRL07.pdf). Lise Getoor, Nir Friedman, Daphne Koller, Avi Pfeffer, Ben Taskar.
