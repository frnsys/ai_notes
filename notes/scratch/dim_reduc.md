## Dimensionality Reduction using Eigendecomposition

[Dimensionality Reduction: Why we take Eigenvectors of the Similarity Matrix? ](https://www.youtube.com/watch?v=3k9hwRCcT30). Michael Lin.

We have a similarity matrix $W$ (by definition of similarity metrics, it is symmetric positive definite).

Then we compute an eigenvector decomposition of $W$ to get $V D V^T$.

Here, $V$ and $D$ are $N \times N$ (thus so is $V^T$) for $N$ datapoints.

Because $W$ is symmetric positive definite, the columns of $V$ will be mutually orthogonal eigenvectors of $W$ (sorted from lowest eigenvalue to highest, left to right). The eigenvalues form the diagonal of the matrix $D$ (and zeros everywhere else), with the lowest eigenvector at $0,0$ and increasing down the diagonal.

We can truncate $V$ down to some arbitrary number of columns (e.g. the last two); this gives us the datapoint embeddings for the reduced space (e.g. if we take the last two columns, we are given the datapoint embeddings for a 2D space)

