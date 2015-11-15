
Typically when words are represented as vectors, it is as a one-hot representation, that is, a vector of length $|V|$ where $V$ is the vocabulary, with all elements 0 except for the one corresponding to the particular word being represented (that is, it is a sparse representation).

This can be quite unwieldy as it has dimensionality of $|V|$, which is typically quite large.

We can instead use neural networks to learn dense representations of words ("word embeddings") of a fixed dimension (the particular dimensionality is specified as a hyperparameter, there is not as of this time a theoretical understanding of how to choose this value) and can capture other properties of words (such as analogies).

Representing a sentence can be accomplished by concatenating the embeddings of its words, but this can be problematic in that typically fixed-size vectors are required, and sentences are variable in their word length.

A way around this is to use the _continuous bag of words_ (CBOW) representation, in which, like the traditional bag-of-words representation, we throw out word order information and combine the embeddings by summing or averaging them, e.g. given a set of word embeddings $v_1, \dots, v_k$:

$$
\text{CBOW}(v_1, \dots, v_k) = \frac{1}{k} \sum_{i=1}^k v_i
$$

An extension of this method is the weighted CBOW (WCBOW) which is just a weighted average of the embeddings.

How are these word embeddings learned? Typically, it is by training a neural network (specifically for learning the embeddings) on an auxiliary task. For instance, context prediction is a common embedding training task, in which we try to predict a word given its surrounding context (under the assumption that words which appear in similar contexts are similar in other important ways).

## Some NN loss functions

### Hinge (binary classification)

In binary classification, the network outputs a single scalar $\hat y$ in $[-1, 1]$ (note that this assumes we are not outputting probability of class membership), and the classification rule takes the sign of the output (i.e. negative, classify as 0, positive, classify as 1). The classification is correct if the sign of $\hat y$ and $y$ are the same (i.e., if $y\hat y > 0$).

The __hinge loss__, also called __margin loss__ or __SVM loss__, is:

$$
L_{\text{hinge(binary)}}(\hat y, y) = \max(0, 1 - y\hat y)
$$


### Hinge (multiclass classification)

In the multiclass classification case, $y$ is a one-hot vector for the correct class. The classification rule is to select the class with the highest score from the output vector $\hat y$: $\argmax_i \hat y_i$. Note that we still assume that the output scores are _not_ probabilities of class membership.

We'll say that $\hat y_t$ is the estimated score for the true class, and $\hat y_k$ as the highest scoring class where $k \neq t$.The multiclass hinge loss is:

$$
L_{\text{hinge(multiclass)}}(\hat y, y) = \max(0, 1 - (\hat y_t - \hat y_k))
$$

### Log loss

A variation of hinge loss; a "soft" hinge loss:

$$
L_{\log}(\hat y, y) = \log(1+ \exp(-(\hat y_t - \hat y_k)))
$$

### Categorical cross-entropy loss (negative log likelihood)

Here we assume that the output values are probabilities of class membership (typically achieved by applying the softmax activation function at the output layer). If $y$ is the true distribution for the input, we have:

$$
L_{\text{cross-entropy}}(\hat y, y) = - \sum_i y_i \log(\hat y_i)
$$

Otherwise, if $y$ is a one-hot vector, this is simplified:

$$
L_{\text{cross-entropy}}(\hat y, y) = - \log(\hat y_t)
$$


## CNNs for NLP

CBOW representations lose word-ordering information, which can be important for some tasks (e.g. sentiment analysis).

CNNs are useful in such situations because they avoid the need of going to, for instance, bigram methods. They can automatically learn important local structures (much as they do with image recognition).


## References

- A Primer on Neural Network Models for Natural Language Processing. Yoav Goldberg. October 5, 2015. <http://arxiv.org/abs/1510.00726>
