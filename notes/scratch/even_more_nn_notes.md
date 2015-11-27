## Conditional Random Fields (CRFs)


## Restricted Boltzmann machines

Restricted Boltzmann machines (RBMs) are a type of neural network used for unsupervised learning; it tries to extract meaningful features.

Such methods are useful for when we have a small supervised training set, but perhaps abundant unlabeled data. We can train an RBM (or another unsupervised learning method) on the unlabeled data to learn useful features to use with the supervised training set - this approach is called _semi-supervised_ learning.


## Autoencoders

Autoencoders are a type of neural network used for unsupervised learning; it tries to extract meaningful features.

It is a feed-forward neural network that is trained to _reproduce its input_. That is, the output layer is the same size as its input layer, and it tries to reconstruct its input at the output layer.

Generally the output of an autoencoder is notated $\hat x$.

The first half (i.e. from the input layer up to the hidden layer) of the autoencoder architecture is called the _encoder_, and the latter half (i.e. from the hidden layer to the output layer) is called the _decoder_.

Often the weights of the decoder, $W*$, are just the transpose of the weights of the encoder $W$, i.e. $W* = W^T$. We refer to such weights as _tied_ weights.

Essentially what happens is the hidden layer learns a compressed representation of the input (given that it is a smaller size than the input/output layers, this is called an _undercomplete_ hidden layer, the learned representation is called an _undercomplete_ representation), since it needs to be reconstructed by the decoder back to its original form.

Undercomplete hidden layers do a good job compressing data similar to its training set, but bad for other inputs.

On the other hand, the hidden layer may be larger than the input/output layers, in which case it is called an _overcomplete_ hidden layer and the learned representation of the input is an _overcomplete_ representation. There's no compression as a result, and there's not guarantee that anything meaningful will be learned (since it can essentially just copy the input).

However, overcomplete representation as a concept is appealing because if we are using this autoencoder to learn features for us, we may want to learn many features. So how can we learn useful overcomplete representations?

### Denoising autoencoders

A _denoising_ autoencoder is a way of learning useful overcomplete representations. The general idea is that we want the encoder to be robust to noise (that is, to be able to reconstruct the original input even in the presence of noise). So instead of inputting $x$, we input $\tilde x$, which is just $x$ with noise added (sometimes called a _corrupted_ input), and the network tries to reconstruct the noiseless $x$ as its output.

There are many ways this noise can be added, but two popular approaches:

- for each component in an input, set it to 0 with probability $v$
- adding Gaussian noise (mean 0, and some variance; this variance is a hyperparameter)


### Loss functions for autoencoders

Say our neural network is $f(x) = \hat x$.

For binary inputs, we can use cross-entropy (more precisely, the sum of Bernoulli cross-entropies):

$$
l(f(x)) = - \sum_k (x_k \log(\hat x_k)) + (1 - x_k)(\log(1-\hat x_k))
$$

For real-valued inputs, we can use the sum of squared differences (i.e. the squared euclidean distance):

$$
l(f(x)) = \frac{1}{2} \sum_k(\hat_k - x_k)^2
$$

And we use a linear activation function at the output.

### Loss function gradient in autoencoders

Note that if you are using tied weights, the gradient $\nabla_W l(f(x^{(t)}))$ is the sum of two gradients; that is, it is sum of the gradients for $W*$ and $W^T$.

### Contractive autoencoders

A _contractive_ autoencoder is another way of learning useful overcomplete representations. We do so by adding an explicit term in the loss that penalizes uninteresting solutions (i.e. that penalizes just copying the input).

Thus we have a new loss function, extended from an existing loss function:

$$
l(f(x^{(t)})) + \lambda ||\nabla_{x^{(t)}} h(x^{(t)})||_F^2
$$

Where $\lambda$ is a hyperparameter and $\nabla_{x^{(t)}} h(x^{(t)})$ is the Jacobian of the encoder, represented as $h(x^{(t)})$, and $||A||_F$ is the Frobenius norm:

$$
||A||_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2}
$$

Where $A$ is a $m \times n$ matrix. To put it another way, the Frobenius norm is the square root of the sum of the absolute squares of a matrix's elements; in this case, the matrix is the Jacobian of the encoder.

Intuitively, the term we're adding to the loss (the squared Frobenius norm of the Jacobian) increases the loss if we have non-zero partial derivatives with the encoder $h(x^{(t)})$ with respect to the input; this essentially means we want to encourage the encoder to throw away information (i.e. we don't want the encoder's output to change with changes to the input; i.e. we want the encoder to be invariant to the input).

We balance this out with the original loss function which, as usual, encourages the encoder to keep good information (information that is useful for reconstructing the original input).

By combining these two conflicting priorities, the result is that the encoder keeps only the good information (the latter term encourages it to throw all information away, the former term encourages it to keep only the good stuff). The $\lambda$ hyperparameter lets us tweak which of these terms to prioritize.


### Contractive vs denoising autoencoders

Both perform well and each has their own advantages.

Denoising autoencoders are simpler to implement in that they are a simple extension of regular autoencoders and do not require computing the Jacobian of the hidden layer.

Contractive autoencoders have a deterministic gradient (since no sampling is involved; i.e. no random noise), which means second-order optimizers can be used (conjugate gradient, LBFGs, etc), and can be more stable than denoising autoencoders.

### Deep autoencoders

Autoencoders can have more than one hidden layer but they can be quite difficult to train (e.g. with small initial weights, the gradient dies).

They can be trained with unsupervised layer-by-layer pre-training (stacking RBMs), or care can be taken in weight initialization.

#### Shallow autoencoders for pre-training

A shallow autoencoder is just an autoencoder with one hidden layer.

In particular, we can create a deep autoencoder by stacking (shallow) denoising autoencoders.

This typically works better than pre-training with RBMs.

Alternatively, (shallow) contractive autoencoders can be stacked, and they also work very well for pre-training.

## Sparse Coding

The sparse coding model is another unsupervised neural network.

The general problem is that for each input $x^{(t)}$, we want to find a latent representation $h^{(t)}$ such that:

- $h^{(t)}$ is sparse (has many zeros)
- we can reconstruct the original input $x^{(t)}$ as well as possible

Formally:

$$
\min_D \frac{1}{T} \sum_{t=1}^T \min_{h^{(t)}} \frac{1}{2} || x^{(t)} - D h^{(t)} ||_2^2 + \lambda || h^{(t)} ||_1
$$

Note that $D h^{(t)}$ is the reconstruction $\hat x^{(t)}$, so the term $|| x^{(t)} - D h^{(t)} ||_2^2$ is the reconstruction error. $D$ is the matrix of weights; in the context of sparse coding it is called a _dictionary_ matrix, and it is equivalent to an autoencoder's output weight matrix.

The term $|| h^{(t)} ||_1$ is a sparsity penalty, to encourage $h^{(t)}$ to be sparse, by penalizing its L1 norm.

We constraint the columns of $D$ to be of norm 1 because otherwise $D$ could just grow large, allowing $h^{(t)}$ to become small (i.e. sparse). Sometimes the columns of $D$ are constrained to be no greater than norm 1 instead of being exactly 1.


## References

- Neural Networks. Hugo Larochelle. 2013. Université de Sherbrooke. <https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH>
- Neural Networks for Machine Learning. Geoff Hinton. 2012. University of Toronto/Coursera.