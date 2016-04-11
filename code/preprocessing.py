import numpy as np


def zero_center(X):
    return X - np.mean(X, axis=0)


def normalize(X):
    return X/np.std(X, axis=0)

def pca_whitening(X):
    """from
    <http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html>
    by Xiu-Shen Wei"""
    X = zero_center(X)

    # covariance matrix
    cov = np.dot(X.T, X)/X.shape[0]

    # compute the SVD factorization
    # of the data covariance matrix
    U, S, V = np.linalg.svd(cov)

    # decorrelate data
    X_rot = np.dot(X, U)

    # whitening:
    # divide by the eigenvalues,
    # which are square roots of the singular values
    eps = 1e-5 # prevent division by zero
    X_white = X_rot/np.sqrt(S+eps)
    return X_white
