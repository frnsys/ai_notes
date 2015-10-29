import numpy as np
import unittest


def probs(X):
    n = len(X)
    o, c = np.unique(X, return_counts=True)
    p = c/n
    return o, p


def entropy(X):
    o, p = probs(X)
    return -sum(p_i*np.log2(p_i) for p_i in p)


def mutual_information(X, Y):
    assert(len(X) == len(Y))

    mi = 0.
    n = len(X)
    o_X, p_X = probs(X)
    o_Y, p_Y = probs(Y)

    for x, p_x in zip(o_X, p_X):
        for y, p_y in zip(o_Y, p_Y):
            p_xy = len(np.where(
                np.in1d(
                    np.where(X == x)[0],
                    np.where(Y == y)[0]
                )
            )[0])/n
            if p_xy == 0:
                continue
            mi += p_xy * np.log2(p_xy/(p_x*p_y))
    return mi


def information_variation(X, Y):
    return entropy(X) + entropy(Y) - (2*mutual_information(X, Y))


class Tests(unittest.TestCase):
    def test_probs(self):
        o, p = probs([1,1,1,2])
        np.testing.assert_array_equal(o, [1,2])
        np.testing.assert_array_equal(p, [0.75,0.25])

    def test_entropy(self):
        h = entropy([1,1,1])
        self.assertEqual(h, 0.)

        h = entropy([1,2,3,4])
        self.assertEqual(h, 2.)


    def test_mutual_information(self):
        mi = mutual_information([7,7,7,3],[0,1,2,3])
        self.assertEqual(mi, 0.81127812445913294)

        mi = mutual_information([0,1,2,3],[0,1,2,3])
        self.assertEqual(mi, 2.)

    def test_information_variation(self):
        vi = information_variation([7,7,7,3],[0,1,2,3])
        self.assertEqual(vi, 1.1887218755408671)

        vi = information_variation([0,1,2,3],[0,1,2,3])
        self.assertEqual(vi, 0.)