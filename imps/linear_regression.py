"""
Example implementation of multivariate linear regression using gradient descent.

- X = feature vectors
- y = labels/target variable
- theta = parameters
- hyp = hypothesis (actually, the vector computed from the hypothesis function)
"""
import numpy as np


def cost_function(X, y, theta):
    """
    This isn't used, but shown for clarity
    """
    m = y.size
    hyp = np.dot(X, theta)
    sq_err = sum(pow(hyp - y, 2))
    return (0.5/m) * sq_err


def coordinate_descent(X, y, theta, alpha=0.01, iterations=10000):
    m = y.size
    for i in range(iterations):
        hyp = np.dot(X, theta)
        for i, p in enumerate(theta):
            temp = X[:,i]
            err = (hyp - y) * temp
            cost_function_derivative = (1.0/m) * err.sum()
            theta[i] = theta[i] - alpha * cost_function_derivative
    return theta


def gradient_descent(X, y, theta, alpha=0.01, iterations=10000):
    m = y.size
    for i in range(iterations):
        hyp = np.dot(X, theta)
        loss = hyp - y
        gradient = np.dot(X.T, loss)/m
        theta = theta - alpha * gradient
    return theta


def least_squares(X, y):
    return np.linalg.lstsq(X, y)[0]


if __name__ == '__main__':
    def true_function(X):
        """
        Computes true outputs and true parameters (randomly generated)
        """
        # Create random parameters for X's dimensions, plus one for x_0.
        true_theta = np.random.rand(X.shape[1] + 1)
        return true_theta[0] + np.dot(true_theta[1:], X.T), true_theta

    # Create some random data
    # (cheating a little b/c we aren't adding any noise)
    n_samples = 2000
    n_dimensions = 5
    X = np.random.rand(n_samples, n_dimensions)
    y, true_theta = true_function(X)

    # Add a column of 1s for x_0
    ones = np.ones((n_samples, 1))
    X = np.hstack([ones, X])

    # Initialize parameters to 0
    theta = np.zeros((n_dimensions+1))

    # Split data
    X_train, y_train = X[:-1], y[:-1]
    X_test, y_test = X[-1:], y[-1:]

    # Estimate parameters
    #theta = gradient_descent(X_train, y_train, theta, alpha=0.01, iterations=10000)
    theta = least_squares(X, y)

    # Predict
    print('true', y_test)
    print('pred', np.dot(X_test, theta))

    print('true theta', true_theta)
    print('pred theta', theta)