import random
import numpy as np


def true_func(X):
    params = np.array([0.5, 0.1])
    return np.dot(X, params)


n_dim = 2
n_samples = 10


def population(size=20):
    # create an initial population
    return np.random.random((size, n_dim))


def fitness(pop, X, y):
    # compute fitness of the population's individuals
    fitnesses = []
    for indiv in pop:
        fitnesses.append(-np.mean(np.square(y - np.dot(X, indiv))))
    return np.hstack(fitnesses)


def breed(pop, fitnesses, n_top=10, random_selection=0.05, mutate_chance=0.01):
    # breed the best selected individuals
    sorted = np.argsort(fitnesses)
    best = sorted[-n_top:]
    n_children = len(pop) - len(best)
    breeders = pop[best]

    # to avoid local minima, randomly select others
    extras = []
    for unfit_idx in sorted[:-n_top]:
        if random.random() < random_selection:
            extras.append(unfit_idx)
    breeders = np.vstack([breeders, pop[extras]])

    # mutate breeders
    mutate_rolls = np.random.random(len(breeders))
    mutate_idx = np.where(mutate_rolls < mutate_chance)[0]

    # select random components to mutate
    rand_cols = np.random.randint(0, breeders.shape[1], mutate_idx.size)

    # set randomly mutated values
    breeders[mutate_idx, rand_cols] = np.random.random(mutate_idx.size)


    children = []
    idx = list(range(len(breeders)))
    dims = np.array([list(range(n_dim))])
    for _ in range(n_children):
        # randomly choose parents
        p_idx = np.random.choice(idx, 2, replace=False)

        # the child is formed from randomly selected elements from each parent
        parents = breeders[p_idx]
        child = parents[np.random.randint(0,2,n_dim), dims]
        children.append(child)

    new_pop = np.vstack([breeders] + children)
    return new_pop


if __name__ == '__main__':
    generations = 1000
    X = np.random.random((n_samples, n_dim))
    y = true_func(X)
    pop = population()

    for i in range(generations):
        fitnesses = fitness(pop, X, y)
        pop = breed(pop, fitnesses)

        # TODO this hangs after awhile...gets a lot slower?
        if i % 100 == 0:
            print('generation {}:'.format(i), np.mean(fitnesses))