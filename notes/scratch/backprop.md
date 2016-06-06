(trying a clearer explanation of backprop)


terms:

- $J$ = cost function
- $w_i$ = weights for layer $i$
- $b_i$ = biases for layer $i$
- $f_i$ = activation function for layer $i$
- $l_i = w_i f_{i-1}(l_{i-1}) + b_i$
- $n$ = number of layers, i.e. $i=n$ is the output layer
- $d$ = number of input dimensions

The goal of backprop is to compute weight and bias updates, i.e. to compute $\frac{\partial J}{\partial w_i}$ and $\frac{\partial J}{\partial b_i}$ for all $i \in [1, n]$. We basically do so through applying the chain rule of derivatives.

For a layer $i$ we want to compute $\frac{\partial J}{\partial l_i}$ because we can easily compute $\frac{\partial J}{\partial w_i}$ and $\frac{\partial J}{\partial b_i}$ from it:

$$
\begin{aligned}
\frac{\partial J}{\partial w_i} &= \frac{\partial J}{\partial l_i} \frac{\partial l_i}{\partial w_i} \\
\frac{\partial J}{\partial b_i} &= \frac{\partial J}{\partial l_i} \frac{\partial l_i}{\partial b_i}
\end{aligned}
$$

Also note that (these derivatives are quite easy to work out on your own):

$$
\begin{aligned}
\frac{\partial l_i}{\partial w_i} &= f_{i-1}(l_{i-1}) \\
\frac{\partial l_i}{\partial b_i} &= 1
\end{aligned}
$$

To clarify, $\frac{\partial l_i}{\partial w_i} = f_{i-1}(l_{i-1})$ just means that it equals the output of the previous layer.

With these in mind, we can carry out backpropagation.

We start with the output layer, i.e. with $i=n$.

$$
\begin{aligned}
\frac{\partial J}{\partial l_n} &= \frac{\partial J}{\partial f_n(l_n)} \frac{\partial f_n(l_n)}{\partial l_n} \\
&= \frac{\partial J}{\partial f_n(l_n)} f_n'(l_n)
\end{aligned}
$$

Note that $\frac{\partial J}{\partial f_n(l_n)}$ is just the derivative of the cost function with respect to the network's predicted values, i.e. $J'(h(x))$.

With $\frac{\partial J}{\partial l_n}$ we can compute the updates for $w_n$ and $b_n$ using the relationship shown above.

Now we move backwards a layer to compute $\frac{\partial J}{\partial l_{n-1}}$. Starting with the full chain rule version:

$$
\frac{\partial J}{\partial l_{n-1}} = \frac{\partial J}{\partial f_n(l_n)} \frac{\partial f_n(l_n)}{\partial l_n} \frac{\partial l_n}{\partial f_{n-1}(l_{n-1})} \frac{\partial f_{n-1}(l_{n-1})}{\partial l_{n-1}}
$$

But we can simplify this a bit, especially because we've already computed $\frac{\partial J}{\partial l_n}$:

$$
\begin{aligned}
\frac{\partial J}{\partial l_{n-1}} &= \frac{\partial J}{\partial l_n} \frac{\partial l_n}{\partial f_{n-1}(l_{n-1})} \frac{\partial f_{n-1}(l_{n-1})}{\partial l_{n-1}} \\
&= \frac{\partial J}{\partial l_n} \frac{\partial l_n}{\partial f_{n-1}(l_{n-1})} f_{n-1}'(l_{n-1})
\end{aligned}
$$

Also note that because:

$$
l_n = w_n f_{n-1}(l_{n-1}) + b_n
$$

Then (again, this is easy to show with basic derivative rules):

$$
\frac{\partial l_n}{\partial f_{n-1}(l_{n-1})} = w_n
$$

Therefore:

$$
\frac{\partial J}{\partial l_{n-1}} = \frac{\partial J}{\partial l_n} w_n f_{n-1}'(l_{n-1})
$$

Then we can again go from this to $\frac{\partial J}{\partial w_{n-1}}$ and $\frac{\partial J}{\partial b_{n-1}}$ using the relationship described earlier.

We can generalize what we just did for any layer $i$:

$$
\frac{\partial J}{\partial l_i} = \frac{\partial J}{\partial l_{i+1}} w_{i+1} f_i'(l_i)
$$

And then use the relationship described earlier to go from this to $\frac{\partial J}{\partial w_i}$ and $\frac{\partial J}{\partial b_i}$.

---

To summarize:

Compute for the output layer, i.e. $i=n$:

$$
\frac{\partial J}{\partial l_n} = J'(h(x)) f_n'(l_n)
$$

Then for all other layers $i \neq n$ (except for the input layer, that has no parameters):

$$
\frac{\partial J}{\partial l_i} = \frac{\partial J}{\partial l_{i+1}} w_{i+1} f_i'(l_i)
$$

Then, for all layers $i \in [1, n]$, compute the weight and bias updates:

$$
\begin{aligned}
\text{weight update} &= \frac{\partial J}{\partial l_i} \frac{\partial l_i}{\partial w_i} = \frac{\partial J}{\partial l_i} f_{i-1}(l_{i-1}) \\
\text{bias update} &= \frac{\partial J}{\partial l_i} \frac{\partial l_i}{\partial b_i} = \frac{\partial J}{\partial l_i}
\end{aligned}
$$

Note that $f_0(l_0) = X$ (the input layer's output is just the input $X$).
