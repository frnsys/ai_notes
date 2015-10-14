import numpy as np
from sklearn import datasets



def make_linear(n_samples=2000, n_dimensions=3, intercept=True, noise_std=1., det=False):
    """
    Generates inputs, outputs and parameters for a linear function
    """
    rnd = np.random.RandomState(seed=1 if det else None)

    X = rnd.rand(n_samples, n_dimensions)

    n_params = n_dimensions
    if intercept:
        n_params += 1

    # Create random parameters for X's dimensions, plus one for x_0.
    theta = rnd.rand(n_dimensions + 1)
    y = theta[0] + np.dot(theta[1:], X.T) + rnd.normal(0, noise_std, n_samples)
    return X, y, theta


def make_blobs(n_samples=2000, n_dimensions=3, n_classes=2, std=1., det=False):
    """
    Make Gaussian blobs for classification
    """
    return datasets.make_blobs(n_samples=n_samples,
                               n_features=n_dimensions,
                               centers=n_classes,
                               cluster_std=std,
                               random_state=1 if det else None)


def make_moons(n_samples=2000, noise_std=0.1, det=False):
    """
    Make non-linearly separable binary classification data
    """
    return datasets.make_moons(n_samples=n_samples,
                               noise=noise_std,
                               random_state=1 if det else None)


def make_circles(n_samples=2000, noise_std=0.1, det=False):
    """
    Make non-linearly separable binary classification data
    """
    return datasets.make_circles(n_samples=n_samples,
                                 noise=noise_std,
                                 random_state=1 if det else None)
