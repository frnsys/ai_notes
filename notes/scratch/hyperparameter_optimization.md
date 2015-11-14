Bayesian optimization:

Basic idea: Model the generalization performance of an algorithm as a smooth function of its hyperparameters and then try to find the maxima.

It has two parts:

- Exploration: evaluate this function on sets of hyperparameters where the outcome is most uncertain
- Exploitation: evaluate this function on sets of hyperparameters which seem likely to output high values

Which repeat until convergence.

This is faster than grid search by making "educated" guesses as to where the optimal set of hyperparameters might be, as opposed to brute-force searching through the entire space.

- <https://chronicles.mfglabs.com/learning-to-learn-or-the-advent-of-augmented-data-scientists-20873282e181>

---

You can also use evolutionary algorithms to search the hyperparameter space.
