from sklearn.cluster import KMeans


def run_kmeans(
    X,
    K,
    random_state=42,
    n_init=30
):
    """
    Run the KMeans baseline.

    Parameters
    ----------
    X : np.ndarray
        Input data with shape:
        n_samples x n_features

    K : int
        Number of clusters.

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of KMeans initializations.

    Returns
    -------
    result : dict
        Dictionary containing predicted labels.
    """

    model = KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=random_state
    )

    labels = model.fit_predict(X)

    return {
        "labels": labels,
        "method": "KMeans"
    }



import numpy as np
from sklearn.cluster import KMeans


def run_specgem(X, K, seed=42):
    """
    Reproduce the classical-PCA baseline labeled
    SpecGEM in Jin & Wang (2016).

    Returns predicted cluster labels.
    """
    X = np.asarray(X, dtype=float)

    # Normalize each feature using sample standard deviation
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    W = (X - means) / stds

    # Obtain the first K-1 unit-norm left singular vectors
    U, singular_values, Vt = np.linalg.svd(
        W, full_matrices=False
    )
    embedding = U[:, :K - 1]

    # Cluster the rows of the spectral embedding
    labels = KMeans(
        n_clusters=K,
        n_init=30,
        random_state=seed
    ).fit_predict(embedding)

    return labels


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_specgem_spectral(X, K, seed=42):
    """
    SpecGEM-style spectral embedding with KMeans,
    following the implementation in Untitled.ipynb.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    K : int
        Number of clusters.
    seed : int
        Random state for KMeans.

    Returns
    -------
    dict
        Predicted labels and spectral embedding details.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Step 1: Standardize features
    X_spec = StandardScaler().fit_transform(X)

    # Step 2: Construct subject similarity matrix
    similarity = X_spec @ X_spec.T
    W = np.sqrt(np.maximum(similarity, 0))

    # Step 3: Construct normalized Laplacian
    degrees = W.sum(axis=1)
    D_inv_sqrt = np.diag(
        1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    )

    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # Step 4: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    order = np.argsort(eigenvalues)
    nu = eigenvalues[order]
    U = eigenvectors[:, order]

    # Step 5: Spectral kernel eigenvalues
    lambdas = np.maximum(0, 1 - nu)

    # Step 6: Eigen-gap dimension selection
    delta = lambdas[:-1] - lambdas[1:]

    gap_threshold = (
        -0.00016 + 2.7 / n + 2.3 / p
    )

    valid = np.where(delta > gap_threshold)[0]

    if len(valid) > 0:
        d = int(valid[-1] + 1)
    else:
        d = K

    # Step 7: Spectral embedding
    Phi = U[:, :d] * np.sqrt(lambdas[:d])

    # Step 8: KMeans clustering
    model = KMeans(
        n_clusters=K,
        n_init=30,
        random_state=seed
    )

    labels = model.fit_predict(Phi)

    return {
        "labels": labels,
        "embedding": Phi,
        "dimension": d,
        "gap_threshold": gap_threshold
    }


import numpy as np
import umap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_umap(X, K, seed=42):
    """
    UMAP + KMeans baseline following Algorithm 7
    and Appendix C.5 of Ma et al. (2026).

    Returns predicted labels and UMAP embedding.
    """

    X = np.asarray(X, dtype=float)

    # Step 1: Standardize the input features
    W = StandardScaler().fit_transform(X)

    # Step 2: Compute the subject Gram matrix
    G = W @ W.T

    # Step 3: Obtain the top K+2 eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    d = K + 2
    init_embedding = eigenvectors[:, -d:][:, ::-1].copy()

    # Step 4: UMAP with custom spectral initialization
    reducer = umap.UMAP(
        n_neighbors=8,
        n_components=d,
        metric="cosine",
        angular_rp_forest=True,
        init=init_embedding,
        random_state=seed
    )

    embedding = reducer.fit_transform(W)

    # Step 5: KMeans clustering
    labels = KMeans(
        n_clusters=K,
        n_init=30,
        random_state=seed
    ).fit_predict(embedding)

    return {
        "labels": labels,
        "embedding": embedding
    }
