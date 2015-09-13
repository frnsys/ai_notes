
# Mathematical Concepts

## Solving analytically vs numerically

Often you may see a distinction made between solving a problem _analytically_ (sometimes _algebraeically_ is used) and solving a problem _numerically_.

Solving a problem analytically means you can exploit properties of the objects and equations, e.g. through methods from calculus and so on, and you can avoid substituting numerical values for the variables you are manipulating. If a problem may be solved analytically, the resulting solution is called a _closed form_ solution (or the _analytic_ solution) and is an exact solution.

Not all problems can be solved analytically; generally more complex mathematical models have no closed form solution. Such problems need to be _approximated_ numerically, which involves evaluating the equations many times by substituting different numerical values for variables. The result is an approximate solution.

## Linear vs nonlinear models

You'll often see a caveat with algorithms that they only work for linear models or it touted that it applies to nonlinear models as well.

A _linear model_ is a model which takes the general form:

$$
y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
$$

Note that this function does not need to produce a literal line (that is, the "linear" constraint does not apply to the predictor variables $x_1, \dots, x_n$). For instance, the function $y = x^2$ is linear. "Linear" rather refers to the parameters; i.e. the function must be "linear in the parameters", meaning that the parameters $\beta_0, \dots, \beta_n$ themselves must form a line (or its equivalent in whatever dimensional space you're working in).

A _nonlinear model_ includes parameters such as $\beta^2$ or $\beta_0 \beta_1$ (that is, multiple parameters in the same term) which is _not_ linear.
