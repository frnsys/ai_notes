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


---

## HDBSCAN

HDBSCAN uses single-linkage clustering, and a concern with single-linkage clustering is that some errant point between two clusters may accidentally act as a bridge between them, such that they are identified as a single cluster. HDBSCAN avoids this by first transforming the space in such a way that sparse points (these potentially troublesome noise points) are pushed further away.

To do this, we first define a distance called the __core distance__, $\text{core}_k(x)$, which is point $x$'s distance from its $k$th nearest neighbor.

Then we define a new distance metric based on these core distances, called __mutual reachability distance__. The mutual reachability distance $d_{\text{mreach}-k}$ between points $a$ and $b$ is the furthest of the following points: $\text{core}_k(a), \text{core}_k(b), d(a,b)$, where $d(a,b)$ is the regular distance metric between $a$ and $b$. More formally:

$$
d_{\text{mreach}-k}(a, b) = \max(\text{core}_k(a), \text{core}_k(b), d(a,b))
$$

For example, if $k=5$:

![](assets/hdbscan_distance_01.svg)

Then we can pick another point:

![](assets/hdbscan_distance_02.svg)

And another point:

![](assets/hdbscan_distance_03.svg)

Say we want to compute the mutual reachability distance between the blue $b$ and green $g$ points.

First we can compute $d(b, g)$:

![](assets/hdbscan_distance_04.svg)

Which is larger than $\text{core}_k(b)$, but both are smaller than $\text{core}_k(g)$. So the mutual reachability distance between $b$ and $g$ is $\text{core}k(g)$:

![](assets/hdbscan_distance_05.svg)

On the other hand, the mutual reachability distance between the red and green points is equal to $d(r, g)$ because that is larger than either of their core distances.

We build a distance matrix out of these mutual reachability distances; this is the transformed space. We can use this distance matrix to represent a graph of the points.

We want to construct a minimum spanning tree out of this graph.

As a reminder, a _spanning tree_ of a graph is any subgraph which contains all vertices and is a tree (a tree is a graph where vertices are connected by only one path; i.e. it is a connected graph - all vertices are connected - but there are no cycles).

The weight of a tree is the sum of its edges' weights. A minimum spanning tree is a spanning tree with the least (or equal to least) weight.

The minimum spanning tree of this graph can be constructed using Prim's algorithm.

From this spanning tree, we then want to create the cluster hierarchy. This can be accomplished by sorting edges from closest to furthest and iterating over them, creating a merged cluster for each edge.

(A note from the original post which I don't understand yet: "The only difficult part here is to identify the two clusters each edge will join together, but this is easy enough via a union-find data structure.")

Given this hierarchy, we want a set of flat clusters. DBSCAN asks you to specify the number of clusters, but HDBSCAN can independently discover them. It does require, however, that you specify a minimum cluster size.

In the produced hierarchy, it is often the case that a cluster splits into one large subcluster and a few independent points. Other times, the cluster splits into two good-sized clusters. The minimum cluster size makes explicit what a "good-sized" cluster is.

If a cluster splits into clusters which are at or above the minimum cluster size, we consider them to be separate clusters. Otherwise, we don't split the cluster (we treat the other points as having "fallen out of" the parent cluster) and just keep the parent cluster intact. However, we keep track of which points have "fallen out" and at what distance that happened. This way we know at which distance cutoffs the cluster "sheds" points. We also keep track at what distances a cluster split into its children clusters.

Using this approach, we "clean up" the hierarchy.

We use the distances at which a cluster breaks up into subclusters to measure the _persistence_ of a cluster. Formally, we think in terms of $\lambda = \frac{1}{\text{distance}}$.

We define for each cluster a $\lambda_{\text{birth}}$, which is the distance at which this cluster's parent split to yield this cluster, and a $\lambda_{\text{death}}$, which is the distance at which this cluster itself split into subclusters (if it does eventually split into subclusters).

Then, for each point $p$ within a cluster, we define $\lambda_p$ to be when that point "fell out" of the cluster, which is either somewhere in between $\lambda_{\text{birth}}, \lambda_{\text{death}}$, or, if the point does not fall out of the cluster, it is just $\lambda_{\text{death}}$ (that is, it falls out when the cluster itself splits).

The _stability_ of a cluster is simply:

$$
\sum_{p \in \text{cluster}} (\lambda_p - \lambda_{\text{birth}})
$$

Then we start with all the leaf nodes and select them as clusters. We move up the tree and sum the stabilities of each cluster's child clusters. Then:

- If the sum of cluster's child stabilities _greater_ than its own stability, then we set its stability to be the sum of its child stabilities.
- If the sum of a cluster's child stabilities is _less_ than its own stability, then we select the cluster and unselect its descendants.

When we reach the root node, return the selected clusters. Points not in any of the selected clusters are considered noise.

As a bonus: each $\lambda_p$ in the selected clusters can be treated as membership strength to the cluster if we normalize them.

## References

- How HDBSCAN Works. Leland McInnes. <http://nbviewer.jupyter.org/github/lmcinnes/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>
