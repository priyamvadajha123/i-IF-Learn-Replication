import math
import numpy as np

from scipy import stats
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# Step 1: Standardize data
# ============================================================

def standardize_data(X):
    """
    Standardize each feature using the same scaling
    as the authors' implementation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix:
        n_samples x n_features

    Returns
    -------
    X_scaled : np.ndarray
        Standardized data.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = (
        X_scaled
        / np.sqrt(X.shape[0])
        * np.sqrt(X.shape[0] - 1)
    )

    return X_scaled


# ============================================================
# Step 2: KS score for each real feature
# ============================================================

def compute_ks_scores(X):
    """
    Compute one KS statistic for every feature.

    Each feature is compared against a standard
    normal distribution.
    """

    p = X.shape[1]

    ks_scores = np.zeros(p)

    for j in range(p):

        ks_stat, _ = kstest(
            X[:, j],
            "norm"
        )

        ks_scores[j] = ks_stat

    return ks_scores


# ============================================================
# Step 3: Simulate KS null distribution
# ============================================================

def simulate_ks_null(
    n,
    rep=100000,
    random_state=42
):
    """
    Simulate the KS statistic under the null hypothesis.

    Important:
    The authors first generate a normal sample and then
    standardize that sample before calculating the KS statistic.
    """

    # Keep reproducibility while following the authors'
    # np.random.randn style.
    rng = np.random.RandomState(random_state)

    ks_values = np.zeros(rep)

    for i in range(rep):

        # Generate a pure normal-noise feature
        x = rng.randn(n)

        # Authors standardize each simulated feature
        x = (
            x - np.mean(x)
        ) / np.std(x)

        # Compute KS statistic manually,
        # following the authors' implementation
        pi = np.sort(
            stats.norm.cdf(x)
        )

        kk = np.arange(
            0,
            n + 1
        ) / n

        ks_stat = np.max(
            np.abs(
                kk[1:n + 1] - pi
            )
        )

        ks_stat = max(
            ks_stat,
            np.max(
                np.abs(
                    kk[:n] - pi
                )
            )
        )

        ks_values[i] = ks_stat

    return ks_values


# ============================================================
# Step 4: Adjust the real KS scores
# ============================================================

def adjust_ks_scores(
    ks_scores,
    null_scores
):
    """
    Rescale the KS statistics from the real data so that
    their mean and standard deviation match the simulated
    KS null distribution.

    This follows the authors' implementation.
    """

    ks_scores = np.asarray(
        ks_scores
    )

    null_scores = np.asarray(
        null_scores
    )

    mean_real = np.mean(
        ks_scores
    )

    std_real = np.std(
        ks_scores
    )

    mean_null = np.mean(
        null_scores
    )

    std_null = np.std(
        null_scores
    )

    adjusted_scores = (
        (ks_scores - mean_real)
        / std_real
        * std_null
        + mean_null
    )

    return adjusted_scores


# ============================================================
# Step 5: Empirical p-values
# ============================================================

def compute_ks_pvalues(
    adjusted_scores,
    null_scores
):
    """
    Compute empirical p-values exactly in the style
    used by the authors.

    For each adjusted KS score:
        p = 1 - fraction of simulated KS scores <= score
    """

    p_values = []

    for score in adjusted_scores:

        count = np.sum(
            null_scores <= score
        )

        p_value = (
            1
            - count / len(null_scores)
        )

        p_values.append(
            p_value
        )

    return np.asarray(
        p_values
    )


# ============================================================
# Step 6: Higher Criticism Thresholding
# ============================================================

def higher_criticism_threshold(
    p_values,
    n,
    p
):
    """
    Calculate the Higher Criticism threshold using
    the same formula and restrictions as the authors.

    Returns
    -------
    j_hat : int
        Threshold used as the number of influential features.
    """

    sorted_p = np.sort(
        p_values
    )

    hc_scores = np.zeros(
        p
    )

    for j in range(
        1,
        p + 1
    ):

        difference = (
            j / p
            - sorted_p[j - 1]
        )

        denominator = math.sqrt(
            max(
                math.sqrt(n)
                * difference,
                0
            )
            + j / p
        )

        hc_scores[j - 1] = (
            math.sqrt(p)
            * difference
            / denominator
        )

    # Authors only consider p-values above log(p)/p
    # and ranks below p/2.
    valid_indices = np.where(
        np.logical_and(
            sorted_p
            > math.log(p) / p,
            np.arange(
                1,
                p + 1
            )
            < p / 2
        )
    )[0]

    if valid_indices.size == 0:
        return None

    j_hat = valid_indices[
        np.argmax(
            hc_scores[
                valid_indices
            ]
        )
    ]

    return j_hat


# ============================================================
# Main IFPCA procedure
# ============================================================

def run_ifpca(
    X,
    K,
    threshold=None,
    null_rep=100000,
    random_state=42
):
    """
    Run IFPCA following the authors' implementation.

    Steps
    -----
    1. Standardize X.
    2. Compute KS statistic for every feature.
    3. Simulate the KS null distribution.
    4. Adjust the real KS statistics.
    5. Compute empirical p-values.
    6. Use HCT to decide how many features to retain.
    7. Keep the strongest KS features.
    8. Compute PCA embedding through XX^T.
    9. Keep top K - 1 eigenvectors.
    10. Run KMeans.

    Returns
    -------
    labels : np.ndarray
        Predicted clusters.

    selected_features : np.ndarray
        Indices of influential features.

    embedding : np.ndarray
        PCA representation used for KMeans.

    ks_scores : np.ndarray
        Original KS statistics.

    p_values : np.ndarray
        Adjusted KS empirical p-values.
    """

    # --------------------------------------------------------
    # 1. Standardize
    # --------------------------------------------------------

    X_scaled = standardize_data(
        X
    )

    n, p = X_scaled.shape

    # --------------------------------------------------------
    # 2. Simulate null KS distribution
    # --------------------------------------------------------

    null_scores = simulate_ks_null(
        n=n,
        rep=null_rep,
        random_state=random_state
    )

    # --------------------------------------------------------
    # 3. KS statistics on real features
    # --------------------------------------------------------

    ks_scores = compute_ks_scores(
        X_scaled
    )

    # --------------------------------------------------------
    # 4. Adjust KS statistics
    # --------------------------------------------------------

    adjusted_scores = adjust_ks_scores(
        ks_scores,
        null_scores
    )

    # --------------------------------------------------------
    # 5. Empirical p-values
    # --------------------------------------------------------

    p_values = compute_ks_pvalues(
        adjusted_scores,
        null_scores
    )

    # --------------------------------------------------------
    # 6. HCT threshold
    # --------------------------------------------------------

    if threshold is None:

        threshold = higher_criticism_threshold(
            p_values,
            n,
            p
        )

    if threshold is None or threshold <= 0:

        raise ValueError(
            "HCT did not select a valid number "
            "of influential features."
        )

    # --------------------------------------------------------
    # 7. Select strongest features
    # --------------------------------------------------------

    selected_features = np.argsort(
        adjusted_scores
    )[-threshold:]

    X_selected = X_scaled[
        :,
        selected_features
    ]

    # --------------------------------------------------------
    # 8. PCA embedding using XX^T
    # --------------------------------------------------------

    gram_matrix = (
        X_selected
        @ X_selected.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        gram_matrix
    )

    embedding = eigenvectors[
        :,
        np.argsort(
            eigenvalues
        )[::-1][:K - 1]
    ]

    # --------------------------------------------------------
    # 9. KMeans
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=K,
        n_init=30,
        random_state=42
    )

    labels = kmeans.fit_predict(
        embedding
    )

    return (
        labels,
        selected_features,
        embedding,
        ks_scores,
        p_values
    )