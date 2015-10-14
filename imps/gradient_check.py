import numpy as np


def check_gradient(cost_func, grad_func, theta, h, eps=0.0001):
    g = grad_func(theta)
    basis_vec = np.zeros(theta.shape)

    for i in theta:
        e_i = basis_vec.copy()
        e_i[i] = 1

        theta_plus = theta.copy()
        theta_plus += e_i * eps

        theta_minus = theta.copy()
        theta_minus -= e_i * eps

        cost_plus = cost_func(theta_plus)
        cost_minus = cost_func(theta_minus)

        numerical_gradient = (cost_plus - cost_minus)/(2*eps)
        if not np.isclose(numerical_gradient, g):
            return False
    return True
