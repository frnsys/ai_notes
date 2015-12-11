## k-Nearest Neighbors (kNN)

A very simple nonparametric classification algorithm in which you take the $k$ closest neighbors to a point ("closest" depends on the distance metric you choose) and each neighbor constitutes a "vote" for its label. Then you assign the point the label with the most votes.

$k$ can be chosen heuristically: generally you don't want it to be so high that the votes become noisy (in the extreme, if you have $n$ datapoints and set $k=n$, you will just choose the most common label in the dataset), and you want to chose it so that it is coprime with the number of classes (that is, they share no common divisors except for 1). This prevents ties.

Alternatively, you can apply an optimization algorithm to choose $k$.

Some distances that you can use include Euclidean distance, Manhattan distance (also known as the city block distance or the taxicab distance), Minkowski distance (a generalization of the Manhattan and Euclidean distances), and Mahalanobis distance.

Minkowski-type distances assume that data is symmetric; that in all dimensions, distance is on the same scale. Mahalanobis distance, on the other hand, takes into account the standard deviation of each dimension.

kNN can work quite quickly when implemented with something like a k-d tree.

## References

- Thoughtful Machine Learning. Matthew Kirk. 2015.