
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


def scfs_spectral_clustering(X, K, seed=42):
    """
    Perform spectral clustering using the top K left singular vectors.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    K : int
        Number of clusters.
    seed : int
        Random seed.

    Returns
    -------
    labels : ndarray
        Predicted cluster labels.
    """

    X = np.asarray(X, dtype=float)

    n_samples = X.shape[0]

    if not 1 < K <= n_samples:
        raise ValueError("K must be between 2 and n_samples.")

    # Obtain the top K left singular vectors.
    # Randomized SVD avoids computing a full decomposition
    # of a high-dimensional gene-expression matrix.
    U, singular_values, Vt = randomized_svd(
        X,
        n_components=K,
        n_iter=7,
        random_state=seed
    )

    # Cluster samples in the spectral embedding.
    kmeans = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=100,
        max_iter=200,
        random_state=seed
    )

    labels = kmeans.fit_predict(U)

    return labels


def scfs_feature_selection(X, labels, tau=0.9):
    """
    Calculate SC-FS feature-selection scores using
    provisional cluster labels.

    A lower score means that less of the feature's
    variation remains unexplained within clusters.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Standardized input matrix.
    labels : ndarray of shape (n_samples,)
        Initial pseudo-labels.
    tau : float
        Selection threshold.

    Returns
    -------
    selected_features : ndarray
        Indices of selected features.
    scores : ndarray
        Within-cluster / total variation scores.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.shape[0] != len(labels):
        raise ValueError("X and labels must have matching sample counts.")

    if not 0 <= tau <= 1:
        raise ValueError("tau must be between 0 and 1.")

    # Total sum of squares for every feature.
    overall_mean = X.mean(axis=0)
    tss = np.sum((X - overall_mean) ** 2, axis=0)

    # Within-cluster residual sum of squares.
    rss = np.zeros(X.shape[1], dtype=float)

    for cluster in np.unique(labels):

        cluster_data = X[labels == cluster]

        cluster_mean = cluster_data.mean(axis=0)

        rss += np.sum(
            (cluster_data - cluster_mean) ** 2,
            axis=0
        )

    # Features with zero variation are uninformative.
    scores = np.ones(X.shape[1], dtype=float)

    valid = tss > 0

    scores[valid] = rss[valid] / tss[valid]

    # Use the strict threshold applied in the R code.
    selected_features = np.where(
        (scores < tau) & valid
    )[0]

    return selected_features, scores


def scfs_lloyd(X, labels, K, max_iter=None):
    """
    Refine existing cluster assignments using Lloyd iterations.

    This uses batch assignment updates. It is a practical
    Python implementation but may differ from the authors'
    sequential-update R implementation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
        Initial cluster assignments.
    K : int
        Number of clusters.
    max_iter : int or None
        Maximum number of refinement iterations.

    Returns
    -------
    labels : ndarray
        Refined cluster assignments.
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int).copy()

    n_samples = X.shape[0]

    if max_iter is None:
        max_iter = max(
            1,
            int(np.ceil(3 * np.log(n_samples)))
        )

    for _ in range(max_iter):

        centers = np.empty(
            (K, X.shape[1]),
            dtype=float
        )

        # Compute cluster centers.
        for k in range(K):

            members = X[labels == k]

            if len(members) == 0:
                # Leave an empty cluster center as NaN;
                # it will not attract assignments.
                centers[k] = np.nan
            else:
                centers[k] = members.mean(axis=0)

        # Distances from each sample to each center.
        distances = np.sum(
            (X[:, None, :] - centers[None, :, :]) ** 2,
            axis=2
        )

        distances[:, np.isnan(centers).any(axis=1)] = np.inf

        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

    return labels


def run_scfs(X, K, variant="SC-FS1", tau=0.9, seed=42):
    """
    Run SC-FS clustering and feature selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.
    K : int
        Number of clusters.
    variant : str
        "SC-FS1": feature selection + spectral reclustering.
        "SC-FS2": SC-FS1 followed by Lloyd refinement.
    tau : float
        Feature-selection threshold.
    seed : int
        Random seed.

    Returns
    -------
    labels : ndarray
        Final predicted cluster labels.
    selected_features : ndarray
        Indices of influential features.
    scores : ndarray
        Feature-selection scores for all original features.
    """

    if variant not in ("SC-FS1", "SC-FS2"):
        raise ValueError(
            "variant must be 'SC-FS1' or 'SC-FS2'."
        )

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if K > min(X.shape):
        raise ValueError(
            "K cannot exceed the number of samples or features."
        )

    # Stage 1: Standardize the input features.
    X_scaled = StandardScaler().fit_transform(X)

    # Stage 2: Obtain initial pseudo-labels.
    initial_labels = scfs_spectral_clustering(
        X_scaled,
        K,
        seed=seed
    )

    # Stage 3: Select informative features.
    selected_features, scores = scfs_feature_selection(
        X_scaled,
        initial_labels,
        tau=tau
    )

    # Fallback when fewer than K features are selected.
    if len(selected_features) < K:

        nonconstant = np.where(
            np.var(X_scaled, axis=0) > 0
        )[0]

        if len(nonconstant) == 0:
            raise ValueError(
                "No nonconstant features are available."
            )

        n_fallback = min(10 * K, len(nonconstant))

        ranked = nonconstant[
            np.argsort(scores[nonconstant])
        ]

        selected_features = ranked[:n_fallback]

    # Keep only selected features.
    X_selected = X_scaled[:, selected_features]

    # Stage 4: Recluster the selected-feature matrix.
    labels = scfs_spectral_clustering(
        X_selected,
        K,
        seed=seed
    )

    # Stage 5: Optional Lloyd refinement.
    if variant == "SC-FS2":

        labels = scfs_lloyd(
            X_selected,
            labels,
            K
        )

    return labels, selected_features, scores
