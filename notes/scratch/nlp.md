$$
\providecommand{\argmax}{\operatorname*{argmax}}
\providecommand{\argmin}{\operatorname*{argmin}}
$$

Some (higher-level) NLP problems:

- machine translation
- (structured) information extraction
- summarization
- natural language interfaces
- speech recognition

NLP is hard because there can be a great deal of ambiguity in language - sentences that are exactly the same can mean different things depending on context.

This ambiguity occurs at many levels:

- the _acoustic_ level: e.g. mixing up similar-sounding words
- the _syntactic_ level: e.g. multiple plausible grammatical parsings of a sentence
- the _semantic_ level: e.g. some words can mean multiple things ("bank" as in a river or a financial institution); this is called _word sense ambiguity_
- the _discourse_ (multi-clause) level: e.g. unclear what a pronoun is referring to

There are lower-level NLP problems which include:

- part-of-speech tagging
- parsing
- word-sense disambiguation
- named entity recognition
- etc

## Language Modeling

We have some finite vocabulary $V$. There is an infinite set of strings ("sentences") that can be produced from $V$, notated $V^{\dagger}$ (these strings have zero or more words from $V$, ending with the `STOP` symbol). These sentences may make sense, or they may not (e.g. they might be grammatically incorrect).

Say we have a training sample of $N$ example sentences in English. We want to learn a probability distribution $p$ over the possible set of sentences $V^{\dagger}$; that is, $p$ is a function that satisfies:

$$
\sum_{x \in V^{\dagger}} p(x) = 1, p(x) \geq 0 \text{for all} x \in V^{\dagger}
$$

The goal is for likely English sentences (i.e. "correct" sentences) to be more probable than nonsensical sentences.

### A naive method

For any sentence $x_1, \dots, x_n$, we notate the count of that sentence in the training corpus as $c(x_1, \dots, x_n)$.

Then we might simply say that:

$$
p(x_1, \dots, x_n) = \frac{c(x_1, \dots, x_n)}{N}
$$

However, this method assigns 0 probability to sentences that are not in the training corpus, thus leaving many plausible sentences unaccounted for.

### Trigram models

Based off of Markov processes.

#### Markov processes

Consider a sequence of random variables $X_1, X_2, \dots, X_n$, each of which can take any value in a finite set $V$. Assume that $n$ is fixed (for now, later we will let $n$ be a random variable as well).

We want to model $P(X_1=x_1, X_2=x_2, \dots, X_n=x_n)$, where $x_1, x_2, \dots, x_n \in V$.

First we can use the chain rule of probability:

$$
P(X_1=x_1, X_2=x_2, \dots, X_n=x_n) = P(X_1=x_1) \prod_{i=2}^n P(X_i=x_i|X_1=x_1,\dots,X_{i-1}=x_{i-1})
$$

Then we make the __first-order Markov assumption__:

$$
P(X_1=x_1) \prod_{i=2}^n P(X_i=x_i|X_1=x_1,\dots,X_{i-1}=x_{i-1}) = P(X_1=x_1) \prod_{i=2}^n P(X_i=x_i|X_{i-1}=x_{i-1})
$$

That is, for any $i \in {2 \dots n}$, for any $x_1, \dots, x_i$:

$$
P(X_i=x_i|X_1=x_1,\dots,X_{i-1}=x_{i-1}) = P(X_i=x_i|X_{i-1}=x_{i-1})
$$

That is, that any random variable depends only on the previous random variable, and is conditionally independent of all the random variables before that. This is called a __first-order Markov process__.

For a __second-order Markov process__, we instead assume that any random variable depends only on the previous _two_ random variables:

$$
P(X_1=x_1,X_2=x_2,\dots,X_n=x_n) = P(X_1=x_1)P(X_2=x_2|X_1=x_1)\prod_{i=3}^n P(X_i=x_i|X_{i-2}=x_{i-2},X_{i-1}=x_{i-1})
$$

Though this is usually condensed to:

$$
\prod_{i=1}^n P(X_i=x_i|X_{i-2}=x_{i-2},X_{i-1}=x_{i-1})
$$

And we define $x_{-1}, x_0$ as the special "start" symbol, $*$.

Now let's remove the assumption that $n$ is fixed and instead consider it as a random variable. We can just define $X_n=\text{STOP}$, where $\text{STOP}$ is a special symbol, $\text{STOP} \notin V$.

#### Trigram models

With a trigram model, we have a parameter $q(w|u,v)$ for each trigram (sequence of three words) $u,v,w$ such that $w \in V \cup \{\text{STOP}\}$ and $u,v \in V \cup \{*\}$.

For any sentence $x_1, \dots, x_n$, where $x_i \in V$ for $i=1 \dots (n-1)$ and $x_n = \text{STOP}$, the probability of the sentence under the trigram language model is:

$$
p(x_1, \dots, x_n) = \prod{i=1}^n q(x_i|x_{i-2},x_{i-1})
$$

With $x_{-1}, x_0$ as the special "start" symbol, $*$.

(This is just a second-order Markov process)

So then, how do we estimate the $q(w_i|w_{i-2},w_{i-1})$ parameters?

We could use the maximum likelihood estimate:

$$
q_{\text{ML}}(w_i|w_{i-2},w_{i-1}) = \frac{\text{Count}(w_{i-2},w_{i-1},w_i)}{\text{Count}(w_{i-2},w_{i-1})}
$$

However, this still has the problem of assigning 0 probability to trigrams that were not encountered in the training corpus.

There are also still many, many parameters to learn: if we have a vocabulary size $N=|V|$, then we have $N^3$ parameters in the model.

## Perplexity

__Perplexity__ is a measure of the quality of a language model.

Assume we have a set of $m$ test sentences, $s_1, s_2, \dots, s_m$.

We can compute the probability of these sentences under our learned model $p$:

$$
\prod_{i=1}^m p(s_i)
$$

Though typically we look at log probability instead:

$$
\sum^m_{i=1} \log p(s_i)
$$

The perplexity is computed:

$$
\begin{aligned}
\text{perplexity} &= 2^{-l} \\
l &= \frac{1}{M} \sum_{i=1}^m \log p(s_i)
\end{aligned}
$$

Where $M$ is the total number of words in the test data. Note that $\log$ is $\log_2$.

Lower perplexity is better (because a high log probability is better, which causes perplexity to be low).

## Parameter estimation in language models

### Linear interpolation

Above we defined the trigram maximum-likelihood estimate. We can do the same for bigram and unigram estimates:

$$
\begin{aligned}
q_{\text{ML}}(w_i|w_{i-1}) &= \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})} \\
q_{\text{ML}}(w_i) &= \frac{\text{Count}(w_i)}{\text{Count}()}
\end{aligned}
$$

These various estimates demonstrate the bias-variance trade-off - the trigram maximum-likelihood converges to a better estimate but requires a lot more data to do so; the unigram maximum-likelihood estimate converges to a worse estimate but does so with a lot less data.

With linear interpolation, we try to combine the strengths and weaknesses of each of these estimates:

$$
q(w_i|w_{i-2}, w_{i-1}) = \lambda_1 q_{\text{ML}}(w_i|w_{i-2},w_{i-1}) + \lambda_2 q_{\text{ML}}(w_i|w_{i-1}) + \lambda_3 q_{\text{ML}}(w_i)
$$

Where $\lambda_1 + \lambda_2 + \lambda_3 = 1, \lambda_i \geq 0 \forall i$.

That is, we compute a weighted average of the estimates.

For a vocabulary $V' = V \cup \{\text{STOP}\}$, $\sum_{w \in V'} q(w|u,v)$ defines a distribution, since it sums to 1.

How do we estimate the $\lambda$ values?

We can take out some our training data as validation data (say ~5%). We train the maximum-likelihood estimates on the training data, then we define $c'(w_1, w_2, w_3)$ as the count of a trigram in the validation set.

Then we define:

$$
L(\lambda_1, \lambda_2, \lambda_3) = \sum_{w_1, w_2, w_3} c'(w_1, w_2, w_3) \log q(w_3|w_1,w_2)
$$

And choose $\lambda_1, \lambda_2, \lambda_3$ to maximize $L$ (this ends up being the same as choosing $\lambda_1, \lambda_2, \lambda_3$ to minimize the perplexity).

In practice, however, the $\lambda$ values are allowed to vary.

We define a function $\Pi$ that partitions histories, e.g.

$$
\Pi(w_{i-2},w_{i-1}) =
\begin{cases}
1 & \text{if Count$(w_{i-1},w_{i-2}) = 0$} \\
2 & \text{if $1 \leq$ Count$(w_{i-1},w_{i-2}) \leq 2$} \\
3 & \text{if $3 \leq$ Count$(w_{i-1},w_{i-2}) \leq 5$} \\
4 & \text{otherwise}
\end{cases}
$$

These partitions are usually chosen by hand.

Then we vary the $\lambda$ values based on the partition:

$$
q(w_i|w_{i-2}, w_{i-1}) = \lambda_1^{\Pi(w_{i-2},w_{i-1})} q_{\text{ML}}(w_i|w_{i-2},w_{i-1}) + \lambda_2^{\Pi(w_{i-2},w_{i-1})} q_{\text{ML}}(w_i|w_{i-1}) + \lambda_3^{\Pi(w_{i-2},w_{i-1})} q_{\text{ML}}(w_i)
$$

Where $lambda_1^{\Pi(w_{i-2},w_{i-1})} + lambda_2^{\Pi(w_{i-2},w_{i-1})} + lambda^3{\Pi(w_{i-2},w_{i-1})} = 1$ and each are $\geq 0$.

### Discounting methods

Generally, these maximum likelihood estimates can be high, so we can define "discounted" counts, e.g. $\text{Count}*(x) = \text{Count}(x) - 0.5$ (the value to discount by can be determined on a validation set, like the $\lambda$ values from before). As a result of these discounted counts, we will have some probability mass left over, which is defined as:

$$
\alpha(w_{i-1}) = 1 - \sum_w \frac{\text{Count}*(w_{i-1}, w)}{\text{Count(w_{i-1})}}
$$

We can assign this leftover probability mass to words we have not yet seen.

We can use a __Katz Back-Off model__. First we will consider the bigram model.

We define two sets:

$$
\begin{aligned}
A(w_{i-1}) &= \{w : \text{Count}(w_{i-1},w) > 0\} \\
B(w_{i-1}) &= \{w : \text{Count}(w_{i-1},w) = 0\}
\end{aligned}
$$

Then the bigram model:

$$
q_{\text{BO}}(w_i|w_{i-1}) =
\begin{cases}
\frac{\text{Count}*(w_{i-1},w_i)}{\text{Count}(w_{i-1})} & \text{if $w_i \in A(w_{i-1})$} \\
\alpha(w_{i-1}) \frac{q_{\text{ML}}(w_i)}{\sum_{w \in B(w_{i-1})} q_{\text{ML}}(w)} & \text{if $w_i \in B(w_{i-1})$}
\end{cases}
$$

Where

$$
\alpha(w_{i-1}) = 1 - \sum_{w \in A(w_{i-1})} \frac{\text{Count}*(w_{i-1}, w)}{\text{Count(w_{i-1})}}
$$

Basically, this assigns the leftover probability mass to bigrams that were not previously encountered.

The Katz Back-Off model can be extended to trigrams as well:

$$
\begin{aligned}
A(w_{i-2},w_{i-1}) &= \{w : \text{Count}(w_{i-2},w_{i-1},w) > 0\} \\
B(w_{i-2},w_{i-1}) &= \{w : \text{Count}(w_{i-2},w_{i-1},w) = 0\} \\
q_{\text{BO}}(w_i|w_{i-2},w_{i-1}) &=
\begin{cases}
\frac{\text{Count}*(w_{i-2},w_{i-1},w_i)}{\text{Count}(w_{i-2},w_{i-1})} & \text{if $w_i \in A(w_{i-2},w_{i-1})$} \\
\alpha(w_{i-2},w_{i-1}) \frac{q_{\text{BO}}(w_i|w_{i-1})}{\sum_{w \in B(w_{i-2},w_{i-1})} q_{\text{BO}}(w|w_{i-1})} & \text{if $w_i \in B(w_{i-2},w_{i-1})$}
\end{cases} \\
\alpha(w_{i-2},w_{i-1}) &= 1 - \sum_{w \in A(w_{i-2},w_{i-1})} \frac{\text{Count}*(w_{i-2},w_{i-1}, w)}{\text{Count(w_{i-2},w_{i-1})}}
\end{aligned}
$$

## Tagging problems

A class of NLP problems in which we want to assign a tag to each word in an input sentence.

- __Part-of-speech tagging__: Given an input sentence, output a POS tag for each word. Like in many NLP problems, ambiguity makes this a difficult task.
- __Named entity recognition__: Given an input sentence, identify the _named entities_ in the sentence (e.g. a company, or location, or person, etc) and what type the entity is (other words are tagged as non-entities). Entities can span multiple words, so there will often be "start" and "continue" tags (e.g. for "Wall Street", "Wall" is tagged as "start company", and "Street" is tagged as "continue company").

There are two types of constraints in tagging problems:

- _local_: words with multiple meanings can have a bias (a "local preference") towards one meaning (i.e. one meaning is more likely than the others)
- _contextual_: certain meanings of a word are more likely in certain contexts

These constraints can sometimes conflict.

### Generative models

One approach to tagging problems (and supervised learning in general) is to use a _conditional model_ (often called a _discriminative model_), i.e. to learn the distribution $p(y|x)$ and select $\argmax_y p(y|x)$ as the label.

Alternatively, we can use a _generative model_ which instead learns the distribution $p(x,y)$. We often have $p(x,y) = p(y)p(x|y)$, where $p(y)$ is the _prior_ and $p(x|y)$ is the _conditional generative model_.

This is generative because we can use this to generate new sentences by sampling the distribution given the words we have so far.

We can apply Bayes' Rule as well to derive the conditional distribution as well:

$$
p(y|x) = \frac{p(y)p(x|y)}{p(x)}
$$

Where $p(x) = \sum_y p(y)p(x|y)$.

Again, we can select $\argmax_y p(y|x)$ as the label, but we can apply Bayes' Rule to equivalently get $\argmax_y \frac{p(y)p(x|y)}{p(x)}$. But note that $p(x)$ does not vary with $y$ (i.e. it is constant), so it does not affect the $\argmax$, and we can just drop it to get $\argmax_y p(x)p(x|y)$.

### Hidden Markov Models (HMM)

An example of a generative model.

We have an input sentence $x = x_1, x_2, \dots, x_n$ where $x_i$ is the $i$th word in the sentence.

We also have a tag sequence $y = y_1, y_2, \dots, y_n$ where $y_i$ is the tag for the $i$th word in the sentence.

We can use a HMM to define the joint distribution $p(x_1, x_2, \dots, x_n, y_1, y_2, \dots, y_n)$.

Then the most likely tag sequence for $x$ is $\argmax_{y_1,\dots,y_n} p(x_1, x_2, \dots, x_n, y_1, y_2, \dots, y_n)$.

#### Trigram HMMs

For any sentence $x_1, \dots, x_n$ where $x_i \in V$ for $i = 1, \dots, n$ and any tag sequence $y_1, \dots, y_{n+1}$ where $y_i \in S$ for $i = 1, \dots, n$ and $y_{n+1} = \text{STOP}$ (where $S$ is the set of possible tags, e.g. DT, NN, VB, P, ADV, etc), the joint probability of the sentence and tag sequence is:

$$
p(x_1, \dots, x_n, y_1, \dots, y_{n+1}) = \prod_{i=1}^{n+1} q(y_i|y_{i-2}, y_{i-1}) \prod_{i=1}^n e(x_i|y_i)
$$

Again we assume that $x_0 = x_{-1} = *$.

The parameters for this model are:

- $q(s|u,v)$ for any $s \in S \cup \{\text{STOP}\}, u, v \in S \cup \{*\}$
- $e(x|s)$ for any $s \in S, x \in V$, sometimes called "emission parameters"

The first product is the (second-order) Markov chain, quite similar to the trigram Markov chain used before for language modeling, and the $e(x_i|y_i)$ terms of the second product are what we have observed. Combined, these produce a hidden Markov model (the Markov chain is "hidden", since we don't observe the tag sequences, we only observe the $x_i$s).

#### Parameter estimation in HMMs

For the $q(y_i|y_{i-2},y_{i-1})$ parameters, we can again use a linear interpolation with maximum likelihood estimates approach as before with the trigram language model.

For the emission parameters, we can also use a maximum likelihood estimate:

$$
e(x|y) = \frac{\text{Count}(y, x)}{\text{Count}(y)}
$$

However, we again have the issue that $e(x|y) = 0$ for all $y$ if we have never seen $x$ in the training data. This will cause the entire joint probability $p(x_1, \dots, x_n, y_1, \dots, y_{n+1})$ to become 0.

How do we deal with low-frequency words then?

We can split the vocabulary into two sets:

- frequent words: occurring $\geq t$ times in the training data, where $t$ is some threshold (e.g. $t=5$)
- low-frequency words: all other words, including those not seen in the training data

Then map low-frequency words into a small, finite set depending on textual features, such as prefixes, suffixes, etc. For example, we may map all all-caps words (e.g. IBM, MTA, etc) to a word class "allCaps", and we may map all four-digit numbers (e.g. 1988, 2010, etc) to a word class "fourDigitNum", or all first words of sentences to a word class "firstWord", and so on.

### The Viterbi algorithm

We want to compute $\argmax_{y_1,\dots,y_n} p(x_1, x_2, \dots, x_n, y_1, y_2, \dots, y_n)$, but we don't want to do so via brute-force search. The search space is far too large, growing exponentially with $n$ (the search space's size is $|S|^n$).

A more efficient way of computing this is to use the __Viterbi algorithm__:

Define $S_k$ for $k=-1, \dots, n$ to be the set of possible tags at position $k$:

$$
\begin{aligned}
S_{-1} &= S_0 = \{*\} \\
S_k &= S \forall k \in \{1, \dots, n\}
\end{aligned}
$$

Then we define:

$$
r(y_{-1}, y_0, y_1, \dots, y_k) = \prod_{i=1}^k q(y_i|y_{i-2}, y_{i-1}) \prod_{i=1}^k e(x_i|y_i)
$$

This computes the probability from our HMM for a given sequence of tags, $y_{-1}, y_0, y_1, \dots, y_k$, but only up to the $k$th position.

We define a dynamic programming table: $\pi(k,u,v)$ as the maximum probability of a tag sequence ending in tags $u, v$ at position $k$, i.e:

$$
\pi(k,u,v) = \max_{(y_{-1}, y_0, y_1, \dots, y_k):y_{k-1}=u,y_k=v} r(y_{-1}, y_0, y_1, \dots, y_k)
$$

To clarify: $k \in \{1, \dots, n\}, u \in S_{k-1}, v \in S_k$.

For example: say we have the sentence "The man saw the dog with the telescope", which we re-write as "START START The Man saw the dog with the telescope". We'll set $S_k=\{D,N,V,P\}$ for $k \geq 1$ and $S_{-1}=S_0=\{*\}$.

If we want to compute $\pi(7,P,D)$, then $k=7$ so then fix the 7th term with the $D$ tag and the $k-1$ term with the $P$ tag. Then we consider all possible tag sequences (ending with $P, D$) up to the 7th term (e.g. $*, D, N, V, P, P, P, D$ and so on) and get the probability of the most likely sequence.

We can re-define the above recursively.

The base case is $\pi(0, *, *) = 1$ since we always have the two START tokens tagged as $*$ at the beginning.

Then, for any $k \in \{1, \dots, n\}$ for any $u \in S_{k-1}$ and $v \in S_k$:

$$
\pi(k, u, v) = \max_{w \in S_{k-2}} (\pi(k-1, w, u)q(v|w,u)e(x_k|v))
$$

The __Viterbi algorithm__ is just the application of this recursive definition while keeping backpointers to the tag sequences with max probability:

- For $k=1, \dots, n$
    - For $u \in S_{k-1}, v \in S_k$
        - $\pi(k, u, v) = \max_{w \in S_{k-2}} (\pi(k-1, w, u)q(v|w,u)e(x_k|v))$
        - $bp(k, u, v) = \argmax_{w \in S_{k-2}} (\pi(k-1, w, u)q(v|w,u)e(x_k|v))$
- Set $(y_{n-1}, y_n) = \argmax_{(u,v)} (\pi(n,u,v)q(\text{STOP}|u,v))$
- For $k=(n-2), \dots, 1, y_k = bp(k+2, y_{k_1}, y_{k+2})$
- Return the tag sequence $y_1, \dots, y_n$

It has the runtime $O(n|S|^3)$ because of the loop over $k$ value (for $k=1, \dots, n$, so this happens $n$ times), then its inner loops over $S$ twice (for $u \in S_{k-1}$ and for $v \in S_k$), with each loop searching over $|S|$.

## The parsing problem

The parsing problem takes some input sentence and outputs a __parse tree__ which describes the syntactic structure of the sentence.

The leaf nodes of the tree are the words themselves, which are each tagged with a part-of-speech. Then these are grouped into phrases, such as noun phrases (NP) and verb phrases (VP), up to sentences (S) (these are sometimes called __constituents__).

These parse trees can describe grammatical relationships such as subject-verb, verb-object, and so on.

TODO parse tree example

We can treat it as a supervised learning problem by using sentences annotated with parse trees (such data is usually called a "treebank").

### Context-free grammars (CFGs)

A formalism for the parsing problem.

A context-free grammar is a four-tuple $G=(N, \Sigma, R, S)$ where:

- $N$ is a set of non-terminal symbols
- $\Sigma$ is a set of terminal symbols
- $R$ is a set of rules of the form $X \to Y_1 Y_2 \dots Y_n$ for $n \geq 0, X \in N, Y_i \in (N \cup \Sigma)$
- $S \in N$ is a distinguished start symbol

An example CFG:

- $N = \{\text{S}, \text{NP}, \text{VP}, \text{PP}, \text{DT}, \text{Vi}, \text{Vt}, \text{NN}, \text{IN}\}$
- $S = \text{S}$
- $\Sigma = \{\text{sleeps}, \text{saw}, \text{woman}, \text{telescope}, \text{the}, \text{with}, \text{in}\}$
- $R$ is the following set of rules:
    - $\text{S} \to \text{NP VP}$
    - $\text{VP} \to \text{Vi}$
    - $\text{VP} \to \text{Vt NP}$
    - $\text{VP} \to \text{VP PP}$
    - $\text{NP} \to \text{DT NN}$
    - $\text{NP} \to \text{NP PP}$
    - $\text{PP} \to \text{IN NP}$
    - $\text{Vi} \to \text{sleeps}$
    - $\text{Vt} \to \text{saw}$
    - $\text{NN} \to \text{man}$
    - $\text{NN} \to \text{woman}$
    - $\text{NN} \to \text{telescope}$
    - $\text{DT} \to \text{the}$
    - $\text{IN} \to \text{with}$
    - $\text{IN} \to \text{in}$

Note:
- S = sentence
- VP = verb phrase
- NP = noun phrase
- PP = prepositional phrase
- DT = determiner
- Vi = intransitive verb
- Vt = transitive verb
- NN = noun
- IN = preposition

We can _derive_ sentences from this grammar.

A __left-most derivation__  is a sequence of strings $s_1, \dots, s_n$ where:

- $s_1 = S$, the start symbol
- $s_n \in \Sigma^*$; that is, $s_n$ consists only of terminal symbols
- each $s_i$ for $i=2, \dots, n$ is derived from $s_{i-1}$ by picking the left-most non-terminal $X$ in $s_{i-1}$ and replacing it with some $\beta$ where $X \to \beta$ is a rule in $R$.

Using the example grammar, we could do:

1. "S"
2. expand "S" to "NP VP"
3. expand "NP" (since it is the left-most symbol) to "D N", yielding "D N VP"
4. expand "D" (again, it is left-most) to "the", yielding "the N VP"
5. expand "N" (since the left-most symbol "the" is a terminal symbol) to "man", yielding "the man VP"
6. expand "VP" to "Vi" (since it is the last non-terminal symbol), yielding "the man Vi"
7. expand "Vi" to "sleeps", yielding "the man sleeps"
8. the sentence consists only of terminal symbols, so we are done.

Thus a CFG defines a set of possible derivations, which can be infinite.

We say that a string $s \in \Sigma^*$ is in the _language_ defined by the CFG if we can derive it from the CFG.

A string in a CFG may have multiple derivations - this property is called "ambiguity".

For instance, "fruit flies like a banana" is ambiguous in that "fruit flies" may be a noun phrase or it may be a noun and a verb.

### Probabilistic Context-Free Grammars (PCFGs)

PCFGs are CFGs in which each rule is assigned a probability, which helps with the ambiguity problem. We can compute the probability of a particular derivation as the product of the probability of its rules.

We notate the probability of a rule as $q(\alpha \to \beta)$. Note that we have individual probability distributions for the left-side of each rule, e.g. $\sum q(\text{VP} \to \beta) = 1, \sum q(\text{NP} \to \beta = 1$, and so on. Another way of saying this is these distributions are conditioned on the left-side of the rule.

These probabilities can be learned from data as well, simply by counting all the rules in a treebank and using maximum likelihood estimates:

$$
q_\text{ML}(\alpha \to \beta) = \frac{\text{Count}(\alpha \to \beta)}{\text{Count}(\alpha)}
$$

Given a PCFG, a sentence $s$, and a set of trees which yield $s$ as $\Tau(s)$, we want to compute $\argmax_{t \in \Tau(s)} p(t)$. That is, given a sentence, what is the most likely parse tree to have produced this sentence?

A challenge here is that $|\Tau(s)|$ may be very large, so brute-force search is not an option. We can use the __CKY algorithm__ instead.

First we will assume the CFG is in Chomsky normal form. A CFG is in _Chomsky normal form_ if the rules in $R$ take one of two forms:

- $X \to Y_1 Y_2$ for $X, Y_1, Y_2 \in N$
- $X \to Y$ for $X \in N, Y \in \Sigma$

In practice, any PCFG can be converted to an equivalent PCFG in Chomsky normal form by combining multiple symbols into single symbols (e.g. you can convert $\text{VP} \to \text{Vt NP PP}$ by defining a new symbol $\text{Vt-NP} \to \text{Vt NP}$ and then redefining $\text{VP} \to \text{Vt-NP PP}$).

First, let's consider the problem $\max_{t \in \Tau(s)} p(t)$.

Notation:

- $n$ = number of words in the sentence
- $w_i$ = the $i$th word in the sentence

We define a dynamic programming table $\pi[i,j,X]$ which is the maximum probability of a constituent with non-terminal $X$ spanning the words $i, \dots, j$ inclusive. We set $i, j \in 1, \dots, n$ and $i \leq j$.

We want to calculate $\max_{t \in \Tau(s) p(t)} = \pi[1,n,S]$, i.e. the max probability for a parse tree spanning the first through the last word of the sentence with the $S$ symbol.

We will use a recursive definition of $\pi$.

The base case is: for all $i = 1, \dots, n$ for $X \in N$, $\pi[i,i,X] = q(X \to w_i)$. If $X \to w_i$ is not in the grammar, then $q(X \to w_i) = 0$.

The recursive definition is: for all $i = 1, \dots, (n-1)$ and $j = (i+1), \dots, n$ and $X \in N$:

$$
\pi(i,j,X) = \max_{X \to YZ \in R, s \in \{i,\dots,(j-1)\}} q(X \to YZ)\pi(i,s,Y)\pi(s+1,j,Z)
$$

$s$ is called the "split point" because it determines where the word sequence from $i$ to $j$ (inclusive) is split.

The full CKY algorithm:

Initialization: For all $i \in \{i, \dots, n\}$, for all $X \in N$:

$$
\pi(i,i,X) =
\begin{cases}
q(X \to x_i) & \text{if} X \to x_i \in R\\
0 & \text{otherwise}
\end{cases}
$$

Then:

- For $l = 1, \dots, (n-1)$
    - For $i = 1, \dots, (n-l)$
        - Set $j = i+1$
        - For all $X \in N$, calculate:

$$
\begin{aligned}
\pi(i,j,X) &= \max_{X \to YZ \in R, s \in \{i,\dots,(j-1)\}} q(X \to YZ)\pi(i,s,Y)\pi(s+1,j,Z) \\
bp(i,j,X) &= \argmax_{X \to YZ \in R, s \in \{i,\dots,(j-1)\}} q(X \to YZ)\pi(i,s,Y)\pi(s+1,j,Z) \\
\end{aligned}
$$

This has the runtime $O(n^3 |N|^3)$ because the $l$ and $i$ loops $n$ times each, giving us $n^2$, then at the inner-most loop (for all $X \in N$) loops $|N|$ times, then $X \to YZ \in R$ has $|N|^2$ values to search through because these are $|N|$ choices for $Y$ and $|N|$ choices for $Z$. Then there are also $n$ choices to search through for $s$.

#### Weaknesses of PCFGs

PCFGs (as described above) don't perform very well; they have two main shortcomings:

- Lack of sensitivity to lexical information
    - that is, attachment is completely independent of the words themselves
- Lack of sensitivity to structural frequencies
    - for example, with the phrase "president of a company in Africa", "in Africa" can be attached to either "president" or "company". If we were to parse this phrase, we might come up with two trees described by exactly the same rule sets, the only difference is where the PP "in Africa" is attached to. Since they are exactly the same rule sets, they have the same probability, so the PCFG can't distinguish the two. However, statistically, the "close attachment" structure (i.e. generally the PP would attach to the closer object, in this case, "company") is more frequent, so it should be preferred.

#### Lexicalized PCFGs

Lexicalized PCFGs deal with the above weaknesses.

For a non-terminal rule, we specify one its children as the "head" of the rule, which is essentially the most "important" part of the rule (e.g. for the rule $\text{VP} \to \text{Vt} \text{NP}$, the verb $\text{Vt}$ is the most important semantic part and thus the head).

We define another set of rules which identifies the heads of our grammar's rules, e.g. "If the rule contains NN, NNS, or NNP, choose the rightmost NN, NNS, or NNP as the head".

Now when we construct the tree, we annotate each node with its headword (that is, the word that is in the place of the head of a rule).

For instance, say we have the following tree:

```
VP
├── Vt
│   └── questioned
└── NP
    ├── DT
    │   └── the
    └── NN
        └── witness
```

We annotate each node with its headword:

```
VP(questioned)
├── Vt(questioned)
│   └── questioned
└── NP(witness)
    ├── DT(the)
    │   └── the
    └── NN(witness)
        └── witness
```

We can revise our Chomsky Normal Form for lexicalized PCFGs by defining the rules in $R$ to have one of the following three forms:

- $X(h) \to_1 Y_1(h) Y_2(w)$ for $X, Y_1, Y_2 \in N$ and $h,w \in \Sigma$
- $X(h) \to_2 Y_1(w) Y_2(h)$ for $X, Y_1, Y_2 \in N$ and $h,w \in \Sigma$
- $X(h) \to h$ for $X \in N, h \in \Sigma$

Note the subscripts on $\to_1, \to_2$ which indicate which of the children is the head.

##### Parsing lexicalized PCFGs

That is, we consider rules with words, e.g. $\text{NN}(dog)$ is a different rule than $\text{NN}(cat)$. By doing so, we increase the number of possible rules to $O(|\Sigma|^2 |N|^3)$, which is a lot.

However, given a sentence $w_1, w_2, \dots, w_n$, at most $O(n^2 |N|^3)$ rules are applicable because we can disregard any rule that does not contain one of $w_1, w_2, \dots, w_n$; this makes parsing lexicalized PCFGs a bit easier (it can be done in $O(N^5 |N|^3)$ time rather than $O(n^3|\Sigma|^2 |N|^3)$ time, which is the runtime if we consider all possible rules).

##### Parameter estimatino in lexicalized PCFGs

In a lexicalized PCFGs, our parameters take the form:

$$
q(\text{S}(\text{saw}) \to_2 \text{NP}(\text{man}) \text{VP}(\text{saw}))
$$

We decompose this parameter into a product of two parameters:

$$
q(\text{S} \to_2 \text{NP VP}|\text{S},\text{saw})q(\text{man}|\text{S} \to_2 \text{NP VP}, \text{saw})
$$

The first term describes: given $\text{S}(\text{saw})$, what is the probability that it expands $\to_2 \text{NP VP}$?

The second term describes: given the rule $\text{S} \to_2 \text{NP VP}$ and the headword $\text{saw}$, what is the probability that $\text{man}$ is the headword of $\text{NP}$?

Then we used smoothed estimation for the two parameter estimates (we're using linear interpolation):

$$
q(\text{S} \to_2 \text{NP VP}|\text{S},\text{saw}) = \lambda_1 q_{\text{ML}}(\text{S} \to_2 \text{NP VP}|\text{S, saw}) + \lambda_2 q_{\text{ML}}(\text{S} \to_2 \text{NP VP}|\text{S})
$$

Again, $\lambda_1, \lambda_2 \geq 0, \lambda_1 + \lambda_2 = 1$.

To clarify:

$$
\begin{aligned}
q_{\text{ML}}(\text{S} \to_2 \text{NP VP}|\text{S, saw}) &= \frac{\text{Count}(\text{S(saw)} \to_2 \text{NP VP})}{\text{Count}(\text{S(saw)})} \\
q_{\text{ML}}(\text{S} \to_2 \text{NP VP}|\text{S}) &= \frac{\text{Count}(\text{S} \to_2 \text{NP VP})}{\text{Count}(\text{S})}
\end{aligned}
$$

Here is the linear interpolation for the second parameter:

$$
q(\text{man}|\text{S} \to_2 \text{NP VP},\text{saw}) = \lambda_3 q_{\text{ML}}(\text{man}|\text{S} \to_2 \text{NP VP},\text{saw}) + \lambda_4 q_{\text{ML}}(\text{man}|\text{S} \to_2 \text{NP VP}) + \lambda_5 q_{\text{ML}}(\text{man}|\text{NP})
$$

Again, $\lambda_3, \lambda_4, \lambda_5 \geq 0, \lambda_3 + \lambda_4 + \lambda_5 = 1$.

To clarify, $q_{\text{ML}}(\text{man}|\text{NP})$ describes: given $\text{NP}$, what is the probability that its headword is $\text{man}$?

This presentation of PCFGs do not deal with the close attachment issue as described earlier, though there are modified forms which do.

## Machine Translation

### Challenges in machine translation

- __lexical ambiguity__ (e.g. "bank" as financial institution, or as in a "river bank")
- differing __word orders__ (e.g. English is subject-verb-object and Japanese is subject-object-verb)
- __syntactic structure__ can vary across languages (e.g. "The bottle floated into the cave" when translated into Spanish has the literal meaning "the bottle entered the cave floating"; the verb "floated" becomes an adverb "floating" modifying "entered")
- __syntactic ambiguity__ (e.g. "John hit the dog with the stick" can have two different translations depending on whether "with the stick" attaches to "John" or to "hit the dog")
- __pronoun resolution__ (e.g. "The computer outputs the data; it is stored in ASCII" - what is "it" referring to?)

### Classical machine translation methods

Early machine translation methods used _direct_ machine translation, which involved translating word-by-word by using a set of rules for translating particular words. Once the words are translated, reordering rules are applied.

But such rule-based systems quickly become unwieldy and fail to encompass the variety of ways words can be used in languages.

There are also _transfer-based_ approaches, which have three phases:

1. Analysis: analyze the source language sentence (e.g. a syntactic analysis to generate a parse tree)
2. Transfer: convert the source-language parse tree to a target-language parse tree based on a set of rules
3. Generation: convert the target-language parse tree to an output sentence

Another approach is _interlingua-based_ translation, which involves two phases:

1. Analysis: analyze the source language sentence into a language-independent representation of its meaning
2. Generation: convert the meaning representation into an output sentence


### Statistical machine translation methods

If we have parallel corpora (parallel meaning that they "line up") for the source and target languages, we can use these as training sets for translation (that is, used a supervised learning approach rather than a rule-based one).

#### The Noisy Channel Model

The noisy channel model has two components:

- $p(e)$, the language model (trained from just the target corpus, could be, for example, a trigram model)
- $p(f|e)$, the translation model

Where $e$ is a target language sentence (e.g. English) and $f$ is a source language sentence (e.g. French).

We want to generate a model $p(e|f)$ which estimates the conditional probability of a target sentence $e$ given the source sentence $f$.

So we have the following, using Bayes' Rule:

$$
\begin{aligned}
p(e|f) &= \frac{p(e,f)}{p(f)} = \frac{p(e)p(f|e)}{\sum_e p(e)p(f|e)} \\
\argmax_e p(e|f) &= \argmax_e p(e)p(f|e)
\end{aligned}
$$

#### IBM translation models

##### IBM Model 1

We want to model $p(f|e)$, where $e$ is the source language sentence with $l$ words, and $f$ is the target language sentence with $m$ words.

We say that an _alignment_ $a$ identifies which source word each target word originated from; that is, $a = \{a_1, \dots, \a_m \}$ where each $a_j \in \{0, \dots, l\}$, and if $a_j=0$ then it does not align to any word.

There are $(l+1)^m$ possible alignments.

Then we define models for $p(a|e,m)$ (the distribution of possible alignments) and $p(f|a,e,m)$, giving:

$$
\begin{aligned}
p(f,a|e,m) &= p(a|e,m) p(f|a,e,m) \\
p(f|e,m) &= \sum_{a \in A} p(a|e,m)p(f|a,e,m)
\end{aligned}
$$

Where $A$ is the set of all possible alignments.

We can also use the model $p(f,a|e,m)$ to get the distribution of alignments given two sentences:

$$
p(a|f,e,m) = \frac{p(f,a|e,m)}{\sum_{a \in A}p(f,a|e,m)}
$$

Which we can then use to compute the most likely alignment for a sentence pair $f, e$:

$$
a^* = \argmax_a p(a|f,e,m)
$$

When we start, we assume that all alignments $a$ are equally likely:

$$
p(a|e,m) = \frac{1}{(l+1)^m}
$$

Which is a big simplification but provides a starting point.

We want to estimate $p(f|a,e,m)$, which is:

$$
p(f|a,e,m) = \prod_{j=1}^m t(f_j|e_{a_j})
$$

Where $t(f_j|e_{a_j})$ is the probability of the source word $e_{a_j}$ being aligned with $f_j$. These are the parameters we are interested in learning.

So the general generative process is as follows:

1. Pick an alignment $a$ with probability $\frac{1}{(l+1)^m}$
2. Pick the target language words with probability:

$$
p(f|a,e,m) = \prod_{j=1}^m t(f_j|e_{a_j})
$$

Then we get our final model:

$$
p(f,a|e,m) = p(a|e,m) p(f|a,e,m) = \frac{1}{(l+1)^m} \prod_{j=1}^m t(f_j|e_{a_j})
$$

##### IBM Model 2

An extension of IBM Model 1; it introduces alignment (also called _distortion_) parameters $q(i|j,l,m)$, which is the probability that the $j$th target word is connected to the $i$th source word. That is, we no longer assume alignments have uniform probability.

We define:

$$
p(a|e,m) = \prod_{j=1}^m q(a_j|j,l,m)
$$

where $a = \{a_1, \dots, a_m\}$.

This now gives us the following as our final model:

$$
p(f,a|e,m) = \prod_{i=1}^m q(a_j|j,l,m) t(f_j|e_{a_j})
$$

In overview, the generative process for IBM model 2 is:

1. Pick an alignment $a = \{a_1, a_2, \dots, a_m \}$ with probability:

$$
\prod_{j=1}^m q(a_j | j,l,m)
$$

2. Pick the target language words with probability:

$$
p(f,a|e,m) = \prod_{j=1}^m t(f_j|e_{a_j})
$$

Which is equivalent to the final model described above.

Then we can use this model to get the most likely alignment for any sentence pair:

Given a sentence pair $e_1, e_2, \dots, e_l$ and $f_1, f_2, \dots, f_m$:

$$
a_j = \argmax_{a \in \{0, \dots, l\}} q(a|j,l,m) t(f_j|e_a)
$$

For $j = 1, \dots, m$.

##### Estimating the $q$ and $t$ parameters

We need to estimate our $q(i|j,l,m)$ and $t(f|e)$ parameters. We have a parallel corpus of sentence pairs, a single example of which is notated $(e^{(k)}, f^{(k)})$ for $k = 1, \dots, n$.

Our training examples _do not_ have alignments annotated (if we did, we could just use maximum likelihood estimates, e.g. $t_{\text{ML}}(f|e) = \frac{\text{Count}(e,f)}{\text{Count}(e)}$ and $q_{\text{ML}}(j|i,l,m) = \frac{\text{Count}(j|i,l,m)}{\text{Count}(i,l,m)}$).

We can use the Expectation Maximization algorithm to estimate these parameters.

We initialize our $q$ and $t$ parameters to random values. Then we iteratively do the following until convergence:

1. Compute "counts" based on the data and our current parameter estimates
2. Re-estimate the parameters with these counts

The amount we increment counts by is:

$$
\delta (k,i,j) = \frac{q(j|i,l_k,m_k)t(f_i^{(k)}|e_j^{(k)})}{\sum_{j=0}^{l_k} q(j|i,l_k,m_k)t(f_i^{(k)}|e_j^{(k)})}
$$

The algorithm for updating counts $c$ is:

- For $k = 1, \dots, n$
  - For $i=1, \dots, m_k$, for $j=0, \dots, l_k$
    - $c(e_j^{(k)}, f_i^{(k)}) += \delta(k,i,j)$
    - $c(e_j^{(k)}) += \delta(k,i,j)$
    - $c(j|i,l,m) += \delta(k,i,j)$
    - $c(i,l,m) += \delta(k,i,j)$

Then recalculate the parameters:

$$
\begin{aligned}
t(f|e) &= \frac{c(e,f)}{c(e)} \\
q(j|i,l,m) &= \frac{c(j|i,l,m)}{c(i,l,m)}
\end{aligned}
$$


##  References

_Natural Language Processing_. Michael Collins. Columbia University/Coursera.