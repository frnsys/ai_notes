"""
loss functions
"""

import numpy as np


def cross_entropy(y, y_hat):
    """
    Intuition: log(x)->- (decreases) as x->0; log(x)->0 as x->1.
    Each row in y_hat is a one-hot vector (i.e. one value is 1, the rest are 0)
    Thus y_hat*np.log(y) drops all values in np.log(y) except for where the one-hot vector equals 1.

    For example:

    y = [0,1,0,0]
    y_hat = [0.1,0.7,0.1,0.1]

    np.log(y_hat)
    >>> [-2.30258509, -0.35667494, -2.30258509, -2.30258509]

    y*np.log(y_hat)
    >>> [0, -0.35667494, 0, 0]

    We take the average of this for all training examples,
    and multiply by -1 so that our cost function is worse the higher it is (and thus we try to minimize it).
    """
    errs = y*np.log(y_hat)
    return -np.mean(errs)


if __name__ == '__main__':
    y = np.array([
        [1.,0.,0.,0.],
        [0.,0.,1.,0.],
        [0.,1.,0.,0.],
        [0.,0.,0.,1.],
    ])

    y_hat = np.array([
        [0.01,0.50,0.01,0.48],
        [0.01,0.01,0.97,0.01],
        [0.01,0.60,0.01,0.38],
        [0.01,0.18,0.01,0.80],
    ])

    # assert that each row is a distribution
    assert(np.array_equal(np.sum(y_hat, axis=1), np.array([1.,1.,1.,1.])))
    print(cross_entropy(y, y_hat))