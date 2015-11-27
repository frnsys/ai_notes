## Adaptive learning rates

Over the course of training, it is often better to gradually decrease the learning rate as you approach an optima so that you don't "overshoot" it.

### Separate adaptive learning rates

The appropriate learning rate can vary across parameters, so it can help to have different adaptive learning rates for each parameter.

For example, the magnitudes of gradients are often very different across layers (starting small early on, growing larger further on).

The fan-in of a neuron (number of inputs) also has an effect, determining the size of "overshoot" effects (the more inputs there are, the more weights are changed simultaneously, all to adjust the same error, which is what can cause the overshooting).

So what you can do is manually set a global learning rate, then for each weight multiply this global learning rate by a local gain, determined empirically per weight.

One way to determine these learning rates is as follows:

- start with a local gain $g_{ij} = 1$ for each weight $w_{ij}$
- increase the local gain if the gradient for that weight does not change sign
- use small additive increases and multiplicative decreases:

$$
g_{ij}(t) =
\begin{cases}
g_{ij}(t-1) + 0.05 & \text{if} (\frac{\partial E}{\partial w_{ij}} (t) \frac{\partial E}{\partial w_{ij}}(t-1)) > 0 \\
0.95 g_{ij}(t-1) & \text{otherwise}
\end{cases}
$$

This ensures that big gains decay rapidly when oscillations start.

Another tip: limit the gains to line in some reasonable range, e.g. $[0.1, 10]$ or $[0.01, 100]$

Note that these adaptive learning rates are meant for full batch learning or for very big mini-batches. Otherwise, you may encounter gradient sign changes that are just due to sampling error of a mini-batch.

These adaptive learning rates can also be combined with momentum by using agreement in sign between the current gradient for a weight and the velocity for that weight.

Note that adaptive learning rates deal only with axis-aligned effects.


## Preventing overfitting

- Get more data, if possible
- Limit your model's capacity so that it can't fit the idiosyncrasies of the data you have. With neural networks, this can be accomplished by:
  - limiting the number of hidden layers and/or number of units per layer
  - start with small weights and stop learning early (so the weights can't get too large)
  - weight decay: penalize large weights using penalties on their squared values (L2) or absolute values (L1)
  - adding Gaussian noise (i.e. $x_i |+ N(0, \sigma_i^2$) to inputs
- Average many different models
  - Use different models with different forms, or
  - Train model on different subsets of the training data ("bagging")
- Use a single neural network architecture, but learn different sets of weights, and average the predictions across these different sets of weights


## Setting hyperparameters

There are many hyperparameters to set with neural networks, such as:

- number of layers
- number of units per layer
- type of unit
- weight penalty
- learning rate
- momentum
- whether or not to use dropout
- etc

and it can be very difficult to choose good ones.

You could do a naive grid search and just try all possible combinations of hyperparameters, which is infeasible because it blows up in size.

You could randomly sample combinations as well, but this still has the problem of repeatedly trying hyperparameter values which may have no effect.

Instead, we can apply machine learning to this problem and try and learn what hyperparameters may perform well based on the attempts thus far. In particular, we can try and predict regions in the hyperparameter space that might do well. We'd want to also be able to be explicit about the uncertainty in our prediction.

We can use Gaussian process models to do so. The basic assumption of these models is that similar inputs give similar outputs.

However, what does "similar" mean? Is 200 hidden units "similar" to 300 hidden units or not? Fortunately, such models can also learn this scale of similarity for each hyperparameter.

These models predict a Gaussian distribution of values for each hyperparameter (hence the name).

A method for applying this:

- keep track of the best hyperparameter combination so far
- pick a new combination of hyperparameters such that the expected improvement of the best combination is big

So we might try a new combination, and it might not do that well, but we won't have replaced our current best.

This method for selecting hyperparameters is called _Bayesian (hyperparameter) optimization_, and is a better approach than by picking hyperparameters by hand (less prone to human error).


## References

- Neural Networks for Machine Learning. Geoff Hinton. 2012. University of Toronto/Coursera.
