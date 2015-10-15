(These are notes from the reference below)

When approaching a clustering problem, you often do not know much about the data beforehand (e.g. what number of clusters to expect).

Criteria for exploratory clustering:

- Err on the side of caution - do not assign data points that look noisy to clusters
- Intuitive parameters - should be clear what effect parameters have on the clustering outcome
- Stability - clusters should not vary too much on multiple runs of the algorithm
- Performance - the algorithm should scale well to large datasets


## K-Means

- Err on the side of caution: K-Means assigns _every_ point to a cluster; i.e. it does not have a concept of noise. It also expects clusters to be globular, so it does not handle more exotic cluster shapes (e.g. moon-shaped clusters).
- Intuitive parameters: K-Means' main parameter is $k$, the number of clusters to expect. This is problematic because you often do not know how many clusters may be in your data. The "elbow" method tries to estimate a value for $k$ by measuring cluster "goodness" for a variety of $k$ values and identifying some "elbow" where the metric sees a big improvement. In practice, this method seldom works well.
- Stability: K-Means has random centroid initializations and as such can result in different clusters across runs.
- Performance: K-Means is relatively simple and scales well to large datasets.

## Affinity Propagation

Data points "vote" on their preferred "exemplar", which yields a set of exemplars as the initial cluster points. Then we just assign each point to the nearest exemplar.

Affinity Propagation is one of the few clustering algorithms which supports non-metric dissimilarities (i.e. the dissimilarities do not need to be symmetric or obey the triangle inequality).

- Err on the side of caution: Like K-Means, Affinity Propagation assigns every point to a cluster and also assumes that clusters are globular.
- Intuitive parameters: You do not need to specify the number of clusters. The two parameters are _preference_ and _damping_, which often require a bit of tweaking to get right.
- Stability: Affinity Propagation is deterministic.
- Performance: does not scale well; the support for non-metric dissimilarities precludes it from many optimizations that other algorithms can take advantage of.

## Spectral Clustering

Spectral Clustering generates a graph of the datapoints, with edges as the distances between the points. Then the Laplacian of the graph is produced:

Given the adjacency matrix $A$ and the degree matrix $D$ of a graph $G$ of $n$ vertices, the Laplacian matrix $L_{n \times n}$ is simply $L = D - A$.

As a reminder:

- the adjacency matrix $A$ is an $n \times n$ matrix where the element $A_{i,j}$ is 1 if an edge exists between vertices $i$ and $j$ and 0 otherwise.
- the degree matrix $D$ is an $n \times n$ diagonal matrix where the element $D_{i,i}$ is the degree of vertex $i$.

Then the eigenvectors of the Laplacian are computed to find an embedding of the graph into Euclidean space. Then some clustering algorithm (typically K-Means) is run on the data in this transformed space.

With K-Means as the clustering algorithm:

- Err on the side of caution: The space transformation of the data means that the original clusters do not need to be globular, since they are globular in the transformed space. However, every point is still assigned to a cluster.
- Intuitive parameters: Still need to specify the number of clusters.
- Stability: Still have random initialization of centroids.
- Performance: The graph transformation component makes this slower than regular K-Means.

## Agglomerative Clustering

(explained elsewhere in my notes)

- Err on the side of caution: No longer any globular assumption, but still no concept of noise.
- Intuitive parameters: We still must specify the number of clusters to find (or some other way of cutting the resulting hierarchy)
- Stability: Stable across runs.
- Performance: Can scale fairly well to large datasets, depending on the implementation.

## DBSCAN

DBSCAN transforms the space according to density, then identifies for dense regions as clusters by using single linkage clustering. Sparse points are considered noise.

- Err on the side of caution: No globular assumption, and sparse points are classified as noise. It does not handle variable density clusters well so it sometimes splits them up into separate clusters.
- Intuitive parameters: The epsilon parameter specifies the distance at which to cut clusters and a "min samples" parameter determines how the space transformation happens. Together, they basically specify the minimum density to consider a cluster.
- Stability: DBSCAN is stable across runs but not so when varying its parameters.
- Performance: Scales well to large datasets.

## HDBSCAN

HDBSCAN is an improvement upon DBSCAN which can handle variable density clusters.

- Err on the side of caution: Same as DBSCAN with the advantage of supporting variable density clusters.
- Intuitive parameters: The epsilon parameter is replaced with a "min cluster size" parameter which is intuitive, though "min samples" remains.
- Stability: HDBSCAN is stable across runs and more so than DBSCAN when varying its parameters.
- Performance: Can scale well depending on the implementation.


## References

- Comparing Python Clustering Algorithms. Leland McInnes. <http://nbviewer.ipython.org/github/lmcinnes/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb>
