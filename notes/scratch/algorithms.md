A _correct_ algorithm is one that always terminates in the correct answer, no matter the inputs.

The number of operations for an algorithm is typically a function of its inputs, typically denoted $n$.

## Major algorithm design paradigms:

- __Divide and conquer__: break up the problem into subproblems which can be recursively solved, then combine the results of the subproblems in some way to form the solution for the original problem.
- __Greedy__: (to do)
- __Dynamic programming__: (to do)


## Merge sort

A canonical example of divide and conquer, which improves over other sorting algorithms such as:

- _selection sort_: each pass over the array identifies the minimum element of the unsorted elements
- _insertion sort_: iterates over each element in the list, inserting it into the proper position of the elements it has already looked at (i.e. the "prefix", or head, or front, of the array)
- _bubble sort_: look at pairs of elements in the array and swap those out of order until the entire array is sorted

TODO merge sort graphic

With merge sort, we take an unsorted array and split in half, then recursively apply merge sort to each half. Then the sorted halves are re-combined ("merged") to give the final sorted array.

This merge step is accomplished by iterating over the two sorted halves in parallel and comparing their values, copying over the smallest of the two to the final sorted array.

Given two sorted halves `A` and `B`, with our final sorted array `C`, The basics of the merging subroutine of the algorithm looks like:

    for k in range(n):
        if A[i] < B[j]:
            C[k] = A[i]
            i += 1
        else if B[j] < A[i]:
            C[k] = B[j]
            j += 1

This is a simplified version, of course, which does not capture end cases (e.g. if `A` is shorter than `B`) nor does it capture when `A[i] == B[j]`.

Let's consider how many operations there are of this (simplified) merge portion of merge sort.

- increment `k`
- compare `A[i]` and `B[j]`
- set `C[k]`
- increment `i` or `j`

So each loop has $4n$ operations.

We also have two other operations where we initialize `i` and `j`.

So in total, this simplified merge subroutine has $4n + 2$ operations.

You could argue that we need an additional operation in the loop for comparing `k` to `n` to see if the loop should terminate (in which case we would have $5n + 2$), but in the end, these small differences don't mean much.

We can assume that $n \geq 1$, so we can say that the runtime for this simplified merge subroutine is $\leq 6n$.


$\log_2 n$ can be thought of as the number of times you divide a value by 2 until you get below or equal to 1. This captures the recursive component of the algorithm, i.e. for there are $\log_2 n$ levels of recursion since it recurses by dividing its input into two.

Intuitively, each level reduces the input by a factor of 2. For instance, if $n=8$, then the first split would have inputs of $n=4$, then the next split would have inputs of $n=2$, then the last split would have inputs of $n=1$. That's three levels (the first level doesn't count, as it is considered the external call to the function), i.e. $\log_2 8 = 3$.

Each level has $2^j$ subproblems with inputs of length $\frac{n}{2^j}$ (e.g. if $n=8$, the first level, $i=1$, has $2^1=2$ different subproblems with inputs of length $\frac{8}{2^1} = 4$).

So the complete analysis would works as follow:

For a level $j$, there are $2^j$ subproblems, and each subproblem really only runs the merge subroutine, which we have already approximated as $6m$ operations, where $m$ is the input size. We know that each of $j$'s subproblems has an input of size $\frac{n}{2^j}$, that is, each subproblem has $6 \frac{n}{2^j}$ operations. So the level as a total has (as an upper bound) the following number of operations:

$$
2^j 6 \frac{n}{2^j}
$$

Which then simplifies:

$$
\require{cancel}
\cancel{2^j} 6 \frac{n}{\cancel{2^j}} = 6n
$$

So the number of operations is independent of the level's depth!

Thus, the total number of operations is just the number of levels times the number of operations for a level:

$$
6n (\log_2(n) + 1) = 6n \log_2 n + 6n
$$

(We include a $+1$ for the final merge step)

### Kinds of algorithmic analysis

- worst-case analysis: the upper bound running time that is true for any arbitrary input of length $n$
- average-case analysis: assuming that all input are equally likely, the average running time
- benchmarks: runtime on an agreed-upon set of "typical" inputs

Average-case analysis and benchmarks requires some domain knowledge about what inputs to expect. When you want to do a more "general purpose" analysis, worst-case analysis is preferable.

### Other algorithmic analysis notes

- Constant factors are typically ignored - this simplifies things, these constants can vary according to a lot of different factors (architecture, compiler, programmer), and in the end, it doesn't have much of an effect on the analysis.
- We focus on __asymptotic analysis__; that is, we focus on large input sizes.
    - Algorithms which are inefficient for large $n$ may be better on small $n$ when compared to algorithms that perform well for large $n$. For example, insertion sort has an upper bound runtime of $\frac{n^2}{2}$, which, for small $n$ (e.g. $n < 90$), is better than merge sort. This is because constant factors are more meaningful with small inputs. Anyways, with small $n$, it often doesn't really matter what algorithm you use, since the input is so small, there are unlikely to be significant performance differences, so analysis of small input sizes is not very valuable (or interesting).

Thus we define a "fast" algorithm as one in which the worst-case running time grows slowly with input size.

## Asymptotic Analysis

With asymptotic analysis, we suppress constant factors and lower-order terms, since they don't matter much for large inputs, and because the constant factors can vary quite a bit depending on the architecture, compiler, programmer, etc.

For example, we'd take our previous upper bound for merge sort, $6n \log_2 n + 6n$ and rewrite it as just $n \log n$ ($\log$ typically implies $\log_2$).

Then we say the running time for merge sort is $O(n \log n)$, said "big-oh of $n \log n$", the $O$ implies that we have dropped the constant factors and lower-order terms.

### Loop examples

Consider the following algorithm for finding an element in an array:

```python
def func(i, arr):
    for el in arr:
        if el == i:
            return True
    return False
```

This has the running time of $O(n)$ since, in the worst case, it checks every item.

Now consider the following:

```python
def func2(i, arr):
    return func(i, arr), func(i, arr)
```

This still has the running time of $O(n)$, although it has twice the number of operations (i.e. $\sim 2n$ operations total), we drop the constant factor $2$.

Now consider the following algorithm for checking if two arrays have a common element:

```python
def func3(arr1, arr2):
    for el1 in arr1:
        for el2 in arr2:
            if el1 == el2:
                return True
    return False
```

This has a runtime of $O(n^2)$, which is called a __quadratic time__ algorithm.

The following algorithm for checking duplicates in an array also has a runtime of $O(n^2)$, again due to dropping constant factors:

```python
def func4(arr):
    for i, el1 in enumerate(arr):
        for el2 in arr[i:]:
            if el1 == el2:
                return True
    return False
```

### Big-Oh formal definition

Say we have a function $T(n)$, $n \geq 0$, which is usually the worst-case running time of an algorithm.

We say that $T(n) = O(f(n))$ if and only if there exist constants $c, n_0 > 0$ such that $T(n) \leq c f(n)$ for all $n /geq n_0$.

That is, we can multiply $f(n)$ by some constant $c$ such that there is some value $n_0$, after which $T(n)$ is always below $c f(n)$.

For example: we demonstrated that $6n \log_2 n + 6n$ is the worst-case running time for merge sort. For merge sort, this is $T(n)$. We described merge sort's running time in big-oh notation with $O(n \log n)$. This is appropriate because there exists some constant $c$ we can multiply $n \log n$ by such that, after some input size $n_0$, $c f(n)$ is always larger than $T(n)$. In this sense, $n_0$ defines a sufficiently large input.

As a simple example, we can prove that $2^{n+10} = O(2^n)$.

So the inequality is:

$$
2^{n+10} \leq c 2^n
$$

We can re-write this:

$$
2^10 2^n \leq c 2^n
$$

Then it's clear that if we set $c=2^10$, this inequality holds, and it happens to hold for all $n$, so we can just set $n_0 = 1$. Thus $2^{n+10} = O(2^n)$ is in fact true.

### Big-Omega notation

$T(n) = \Omega(f(n))$ if and only if there exist constants $c, n_0 > 0$ such that $T(n) \geq c f(n)$ for all $n \geq n_0$.

That is, we can multiply $f(n)$ by some constant $c$ such that there is some value $n_0$, after which $T(n)$ is always _above_ $c f(n)$.

### Big-Theta notation

$T(n) = \Theta(f(n))$ if and only if $T(n) = O(f(n))$ _and_ $T(n) = \Omega(f(n))$. That is, $T(n)$ eventually stays sandwiched between $c_1 f(n)$ and $c_2 f(n)$ after some value $n_0$.

### Little-Oh notation

Stricter than big-oh, in that this must be true for all positive constants.

$T(n) = o(f(n))$ if and only if for _all_ constants $c>0$, there exists a constant $n_0$ such that $T(n) \leq c f(n)$ for all $n \geq n_0$.

## The divide-and-conquer paradigm

This consists of:

- Divide the problem into smaller subproblems. You do not have to literally divide the problem in the algorithm's implementation; this may just be a conceptual step.
- Compute the subproblems using recursion.
- Combine the subproblem solutions into the problem for the original problem.

---

Other algorithmic design methods:

- Randomization: involves some randomness

Primitives: there are some algorithms that are so fast that they are considered "primitives", i.e. to be used as building blocks for more complex algorithms.

## References

- Algorithms: Design and Analysis, Part 1. Tim Roughgarden. Stanford/Coursera.