
import random
import numpy as np

from scipy import stats
from scipy.stats import f_oneway, norm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_distances

from .ifpca import (
    run_ifpca,
    standardize_data,
    higher_criticism_threshold,
)


# ============================================================
# Step 1: ANOVA F-statistics
# ============================================================

def compute_anova_scores(X, labels):
    """
    Compute an ANOVA F-statistic for every feature
    using the current pseudo-labels.

    Parameters
    ----------
    X : np.ndarray
        Data matrix: n_samples x n_features

    labels : np.ndarray
        Current pseudo-cluster labels.

    Returns
    -------
    f_scores : np.ndarray
        One F-statistic per feature.

    p_values : np.ndarray
        Corresponding ANOVA p-values.
    """

    n_features = X.shape[1]

    f_scores = []
    p_values = []

    unique_labels = np.unique(labels)

    for j in range(n_features):

        groups = [
            X[labels == label, j]
            for label in unique_labels
        ]

        f_stat, p_value = f_oneway(*groups)

        if np.isnan(f_stat):
            f_stat = 0

        if np.isnan(p_value):
            p_value = 1.0

        f_scores.append(f_stat)
        p_values.append(p_value)

    return (
        np.array(f_scores),
        np.array(p_values)
    )


# ============================================================
# Step 2: Adjust F-statistics
# ============================================================

def adjust_f_scores(f_scores, df1, df2):
    """
    Adjust empirical F-statistics using quartile
    matching with the theoretical F-distribution.
    """

    q1_data = np.percentile(f_scores, 25)
    q2_data = np.percentile(f_scores, 50)
    q3_data = np.percentile(f_scores, 75)

    q1_theory = stats.f.ppf(0.25, df1, df2)
    q2_theory = stats.f.ppf(0.50, df1, df2)
    q3_theory = stats.f.ppf(0.75, df1, df2)

    f_adjusted = (
        (f_scores - q2_data)
        / (q3_data - q1_data)
    ) * (
        q3_theory - q1_theory
    ) + q2_theory

    return f_adjusted


# ============================================================
# Helper: Replace infinity values
# ============================================================

def replace_infinite_values(arr):
    """
    Replace positive and negative infinity with
    the largest and smallest finite values.
    """

    arr = np.array(arr, dtype=float)

    finite = arr[np.isfinite(arr)]

    max_finite = finite.max()
    min_finite = finite.min()

    arr[arr == np.inf] = max_finite
    arr[arr == -np.inf] = min_finite

    return arr


# ============================================================
# Step 3: Reliability calculation
# ============================================================

def compute_reliability_weight(
    reliability_pvalues,
    n,
    c_constant=0.6
):
    """
    Compute Higher-Criticism-style reliability
    scores for currently selected features.
    """

    sorted_p = np.sort(reliability_pvalues)

    s = len(sorted_p)

    j_values = np.arange(1, s + 1)

    scores = (
        np.sqrt(s)
        * (j_values / n - sorted_p)
        / np.sqrt(sorted_p * (1 - sorted_p))
    )

    return (
        sorted_p,
        scores
    )


# ============================================================
# Step 4: Final reliability weight
# ============================================================

def final_reliability_weight(
    reliability_scores,
    c_constant=0.6
):
    """
    Convert the reliability statistic into the
    final information weight used by i-IF-Learn.
    """

    two_thirds = int(
        len(reliability_scores) * 2 / 3
    )

    stable_scores = reliability_scores[-two_thirds:]

    s = len(stable_scores)

    b = np.sqrt(
        2 * np.log(np.log(s))
    )

    c = (
        2 * np.log(np.log(s))
        + 0.5 * np.log(np.log(np.log(s)))
        - np.log(4 * np.pi) / 2
    )

    score_max = np.max(stable_scores)

    w1 = np.exp(
        -np.exp(c - b * score_max)
    )

    w = (
        1
        - (1 - w1)
        / (1 - w1 + c_constant)
    )

    return (
        w,
        w1,
        score_max
    )


# ============================================================
# Step 5: Embedding
# ============================================================

def create_embedding(
    X_selected,
    n_components,
    method="PCA"
):
    """
    Create a PCA or Laplacian Eigenmap embedding
    of the selected features.
    """

    if method.lower() == "pca":

        gram_matrix = (
            X_selected @ X_selected.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            gram_matrix
        )

        order = np.argsort(eigenvalues)[::-1]

        embedding = eigenvectors[
            :, order[:n_components]
        ]

    elif method.lower() == "laplacian":

        cosine_dist = cosine_distances(
            X_selected
        )

        gamma = 1

        affinity_matrix = np.exp(
            -gamma * cosine_dist ** 2
        )

        model = SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",
            n_neighbors=8
        )

        embedding = model.fit_transform(
            affinity_matrix
        )

    else:

        raise ValueError(
            "method must be 'PCA' or 'laplacian'"
        )

    return embedding


# ============================================================
# Step 6: Random F-statistic reference
# ============================================================

def generate_random_f_reference(
    X,
    K,
    num_selected_features,
    component,
    method="PCA",
    num_trials=200,
    random_seed=42
):
    """
    Generate the random F-statistic reference
    distribution used to estimate reliability.

    For each trial:
    1. Randomly select features.
    2. Embed those features.
    3. Run KMeans.
    4. Compute ANOVA F-statistics.
    """

    np.random.seed(random_seed)
    random.seed(random_seed)

    n, p = X.shape

    scaler = StandardScaler()

    Z = scaler.fit_transform(X)

    Z = (
        Z * np.sqrt((n - 1) / n)
    )

    random_f_values = []

    for trial in range(num_trials):

        random_indices = random.sample(
            range(p),
            num_selected_features
        )

        X_random = Z[:, random_indices]

        embedding = create_embedding(
            X_random,
            n_components=component,
            method=method
        )

        kmeans = KMeans(
            n_clusters=K,
            n_init=30,
            random_state=42
        )

        random_labels = kmeans.fit_predict(
            embedding
        )

        random_scores, _ = compute_anova_scores(
            X,
            random_labels
        )

        random_f_values.extend(
            random_scores[random_indices]
        )

    return np.array(random_f_values)


# ============================================================
# Main i-IF-Learn
# ============================================================

def run_i_if_learn(
    X,
    K,
    method="PCA",
    seed=42,
    max_iter=10,
    convergence_threshold=0.10,
    random_trials=200,
    ks_repetitions=100000
):
    """
    Run the i-IF-Learn algorithm.

    method="PCA"
        -> i-IF-PCA

    method="laplacian"
        -> i-IF-Lap

    Returns
    -------
    result : dict

        labels
            Final cluster labels.

        features
            Final selected features.

        history
            Per-iteration statistics, cluster
            labels, and selected feature indices.

        method
            Embedding method used.

        initial_labels
            Pseudo-labels from IF-PCA initialization.

        initial_features
            Features from IF-PCA initialization.
    """

    np.random.seed(seed)
    random.seed(seed)

    # ========================================================
    # Stage 1: IF-PCA initialization
    # ========================================================

    (
        labels,
        current_features,
        _,
        _,
        ks_pvalues
    ) = run_ifpca(
        X,
        K,
        null_rep=ks_repetitions,
        random_state=seed
    )

    # NEW: Preserve initial IF-PCA predictions and features.
    # These are saved before iterative refinement begins.
    initial_labels = labels.copy()
    initial_features = current_features.copy()

    # Same standardized matrix used throughout
    X_scaled = standardize_data(X)

    n, p = X.shape

    # Degrees of freedom for F distribution
    df1 = K - 1
    df2 = n - K

    # K + 2 dimensions during iterative embedding
    component = K + 2

    # ========================================================
    # Create random F reference
    # ========================================================

    random_f_reference = generate_random_f_reference(
        X=X,
        K=K,
        num_selected_features=len(current_features),
        component=component,
        method=method,
        num_trials=random_trials,
        random_seed=seed
    )

    history = []

    # ========================================================
    # Iterative refinement
    # ========================================================

    for iteration in range(1, max_iter + 1):

        # ----------------------------------------------------
        # 1. F-statistic using current pseudo-labels
        # ----------------------------------------------------

        f_scores, _ = compute_anova_scores(
            X,
            labels
        )

        f_scores = replace_infinite_values(
            f_scores
        )

        # ----------------------------------------------------
        # 2. Adjust F-statistics
        # ----------------------------------------------------

        f_adjusted = adjust_f_scores(
            f_scores,
            df1,
            df2
        )

        # Convert adjusted F statistics to p-values
        f_pvalues = (
            1 - stats.f.cdf(
                f_adjusted,
                df1,
                df2
            )
        )

        # ----------------------------------------------------
        # 3. Estimate reliability of pseudo-labels
        # ----------------------------------------------------

        selected_f_scores = f_scores[
            current_features
        ]

        reliability_pvalues = np.array([
            np.mean(
                f_score < random_f_reference
            )
            for f_score in selected_f_scores
        ])

        (
            _,
            reliability_scores
        ) = compute_reliability_weight(
            reliability_pvalues,
            n=n
        )

        (
            w,
            _,
            _
        ) = final_reliability_weight(
            reliability_scores
        )

        # ----------------------------------------------------
        # 4. Convert F and KS p-values to normal scores
        # ----------------------------------------------------

        z_f = norm.ppf(f_pvalues)
        z_ks = norm.ppf(ks_pvalues)

        z_f = replace_infinite_values(z_f)
        z_ks = replace_infinite_values(z_ks)

        # ----------------------------------------------------
        # 5. Combine supervised and unsupervised evidence
        # ----------------------------------------------------

        combined_scores = (
            w * z_f
            + (1 - w) * z_ks
        )

        z_scores = (
            combined_scores
            / (
                w ** 2
                + (1 - w) ** 2
            )
        )

        # ----------------------------------------------------
        # 6. HCT feature selection
        # ----------------------------------------------------

        new_pvalues = norm.cdf(
            z_scores
        )

        threshold = higher_criticism_threshold(
            new_pvalues,
            n=n,
            p=p
        )

        if threshold is None or threshold <= 0:

            raise ValueError(
                f"HCT failed at iteration {iteration}"
            )

        # Smallest scores are strongest
        new_features = np.argsort(
            z_scores
        )[:threshold]

        # ----------------------------------------------------
        # 7. Embed selected features
        # ----------------------------------------------------

        X_selected = X_scaled[
            :,
            new_features
        ]

        embedding = create_embedding(
            X_selected,
            n_components=component,
            method=method
        )

        # ----------------------------------------------------
        # 8. KMeans
        # ----------------------------------------------------

        kmeans = KMeans(
            n_clusters=K,
            n_init=30,
            random_state=42
        )

        new_labels = kmeans.fit_predict(
            embedding
        )

        # ----------------------------------------------------
        # 9. Convergence
        # ----------------------------------------------------

        changed_features = np.setdiff1d(
            new_features,
            current_features
        )

        change_ratio = (
            len(changed_features)
            / len(current_features)
        )

        # NEW: Save the labels and features after each
        # iteration, alongside the existing statistics.
        history.append({
            "iteration": iteration,
            "weight": w,
            "n_features": len(new_features),
            "change_ratio": change_ratio,
            "labels": new_labels.copy(),
            "features": new_features.copy()
        })

        print(
            f"Iteration {iteration}: "
            f"features={len(new_features)}, "
            f"w={w:.4f}, "
            f"change={change_ratio:.4f}"
        )

        # Update for next iteration
        labels = new_labels
        current_features = new_features

        if change_ratio < convergence_threshold:

            print(
                f"Converged after {iteration} iterations."
            )
            break

    # NEW: Return the initial results in addition
    # to the existing final results and history.
    return {
        "labels": labels,
        "features": current_features,
        "history": history,
        "method": method,
        "initial_labels": initial_labels,
        "initial_features": initial_features
    }
