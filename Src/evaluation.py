import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


# ============================================================
# Clustering accuracy
# ============================================================

def clustering_accuracy(y_true, y_pred):
    """
    Compute clustering accuracy after finding the best
    one-to-one mapping between predicted cluster labels
    and true class labels.

    This is needed because cluster IDs are arbitrary.
    For example, predicted cluster 0 may correspond to
    true class 2.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    confusion = np.zeros(
        (len(true_labels), len(pred_labels)),
        dtype=int
    )

    # Count how many samples belong to each
    # true-label / predicted-label combination
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            confusion[i, j] = np.sum(
                (y_true == true_label)
                & (y_pred == pred_label)
            )

    # Find the best mapping between predicted
    # cluster IDs and true class labels
    row_ind, col_ind = linear_sum_assignment(
        -confusion
    )

    correct = confusion[
        row_ind,
        col_ind
    ].sum()

    accuracy = correct / len(y_true)

    return accuracy


# ============================================================
# Evaluate clustering
# ============================================================

def evaluate_clustering(y_true, y_pred):
    """
    Compute the two main clustering metrics used
    in the i-IF-Learn paper:

    1. Clustering Accuracy
    2. Adjusted Rand Index (ARI)
    """

    accuracy = clustering_accuracy(
        y_true,
        y_pred
    )

    ari = adjusted_rand_score(
        y_true,
        y_pred
    )

    return {
        "Accuracy": accuracy,
        "ARI": ari
    }


# ============================================================
# Create one standardized result row
# ============================================================

def make_result_row(
    dataset,
    method,
    y_true,
    y_pred,
    selected_features=None,
    runtime=None
):
    """
    Create one standardized experiment result row.

    This will make it easier to combine results from
    different datasets and methods into one DataFrame.
    """

    metrics = evaluate_clustering(
        y_true,
        y_pred
    )

    return {
        "Dataset": dataset,
        "Method": method,
        "Accuracy": metrics["Accuracy"],
        "ARI": metrics["ARI"],
        "Selected Features": selected_features,
        "Runtime (s)": runtime
    }