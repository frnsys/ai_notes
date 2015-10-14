Consider the following problem:

    7 * 13 = ?

This is solved very quickly by a computer (it gives 91).

Now consider the following _factoring_ problem :

    ? * ? = 91

This a bit more difficult for a computer to solve, though it will yield the correct answers (7 and 13).

If we consider extremely large numbers, a computer can still very quickly compute their product.

But, given a product, it will take a computer a very, very long time to compute their factors.

In fact, modern cryptography is based on the fact that computers are not good at finding factors for a number (in particular, prime factors).

This is because computers basically have to use brute force search to identify a factor; with very large numbers, this search space is enormous (it grows exponentially).

However, once we find a possible solution, it is easy to check that we are correct (e.g. just multiply the factors and compare the product).

There are many problems which are characterized in this way - they require brute force search to identify the _exact_ answer (there are often faster ways of getting _approximate_ answers), but once an answer is found, it can be easily checked if it is correct.

There are other problems, such as multiplication, where we can easily "jump" directly to the correct exact answer.

For the problems that require search, it is not known whether or not there is also a method that can "jump" to the correct answer.

Consider the "needle in the haystack" analogy. We could go through each piece of hay until we find the needle (brute force search). Or we could use a magnet to pull it out immediately. The question is open: does this magnet exist for problems like factorization?

Problems which we can quickly solve, like multiplication, are in a family of problems called "P", which stands for "polynomial time" (referring to the relationship between the number of inputs and how computation time increases).

Problems which can be quickly verified to be correct are in a family of problems called "NP", which stands for "nondeterministic polynomial time".

P is a subset of NP, since their answers are quickly verified, but NP also includes the aforementioned search problems. Thus, a major question is whether or not P = NP, i.e. we have the "magnet" for P problems, but is there also one for the rest of the NP problems? Or is searching the only option?

### NP-completeness

There are some NP problems which all NP problems can be reduced to. Such an NP problem is called _NP-complete_.

For example, any NP problem can be reduced to a clique problem (e.g. finding a clique of some arbitrary size in a graph); thus the clique problem is NP-complete. Any other NP problem can be reduced to a clique problem, and if we can find a way of solving the clique problem quickly, we can also all of those related problems quickly as well.

### NP-hard

Problems which are _NP-hard_ are at least as hard as NP problems, so this includes problems which may not even be in NP.


## References

- Beyond Computation: The P vs NP Problem. Michael Sipser, MIT. Tuesday, October 3, 2006. <https://www.youtube.com/watch?v=msp2y_Y5MLE>
