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

## References

_Natural Language Processing_. Michael Collins. Columbia University/Coursera.