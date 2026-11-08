from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Dataset registry
# ============================================================

MICROARRAY_DATASETS = {
    "brain": {
        "x_file": "brain.x.txt",
        "y_file": "brain.y.txt",
        "sep": r"\s+",
        "transpose_y": True,
    },
    
"breast": {
    "x_file": "breast.x.txt",
    "y_file": "breast.y.txt",
    "sep": ","
},

    "colon": {
        "x_file": "colon.x.txt",
        "y_file": "colon.y.txt",
        "sep": r"\s+",
        "transpose_y": True,
    },
    "leukemia": {
        "x_file": "leukemia.x.txt",
        "y_file": "leukemia.y.txt",
        "sep": r"\s+",
        "transpose_y": True,
    },
    "lung1": {
        "x_file": "lung1.x.txt",
        "y_file": "lung1.y.txt",
        "sep": ","
    },
    "lung2": {
        "x_file": "lung2.x.txt",
        "y_file": "lung2.y.txt",
        "sep": ",",
        "transpose_y": True,
    },
    "lymphoma": {
        "x_file": "lymphoma.x.txt",
        "y_file": "lymphoma.y.txt",
        "sep": r"\s+",
        "transpose_y": True,
    },
    "prostate": {
        "x_file": "prostate.x.txt",
        "y_file": "prostate.y.txt",
        "sep": r"\s+",
        "transpose_y": True,
    },
    "srbct": {
        "x_file": "srbct.x.txt",
        "y_file": "srbct.y.txt",
        "sep": r"\s+",
        "transpose_y": True,
    },
    "sucancer": {
        "x_file": "su.x.txt",
        "y_file": "su.y.txt",
        "sep": ","
            },
}


SCRNA_DATASETS = {
    "camp1": {
        "x_file": "camp1-x-filter.txt",
        "y_file": "camp1-y.txt",
    },
    "camp2": {
        "x_file": "camp2-x-filter.txt",
        "y_file": "camp2-y.txt",
    },
    "darmanis": {
        "x_file": "darmanis-x-filter.txt",
        "y_file": "darmanis-y.txt",
    },
    "deng": {
        "x_file": "deng-x-filter.txt",
        "y_file": "deng-y.txt",
    },
    "goolam": {
        "x_file": "goolam-x-filter.txt",
        "y_file": "goolam-y.txt",
    },
    "grun": {
        "x_file": "grun-x-filter.txt",
        "y_file": "grun-y.txt",
    },
    "li": {
        "x_file": "li-x-filter.txt",
        "y_file": "li-y.txt",
    },
    "patel": {
        "x_file": "patel-x-filter.txt",
        "y_file": "patel-y.txt",
    },
}


# ============================================================
# Helper functions
# ============================================================

def _load_microarray_labels(path, config):
    """
    Load microarray labels using the same formats shown
    in the original author example code.
    """

    if config.get("label_mode") == "fourth_column":
        y = pd.read_csv(
            path,
            sep=r"\s+",
            header=None
        ).iloc[:, 3].values

    elif config.get("transpose_y", False):
        y = pd.read_csv(
            path,
            sep=config["sep"],
            header=None
        ).T.values[0]

    else:
        y = pd.read_csv(
            path,
            sep=config["sep"],
            header=None
        ).values.ravel()

    return np.asarray(y)


def _ensure_samples_by_features(X, y):
    """
    Ensure X has shape:
        n_samples x n_features

    If one dimension of X matches the number of labels,
    use that dimension as the sample dimension.
    """

    if X.shape[0] == len(y):
        return X

    if X.shape[1] == len(y):
        return X.T

    raise ValueError(
        f"Could not determine orientation. "
        f"X shape={X.shape}, len(y)={len(y)}"
    )


# ============================================================
# Main loader
# ============================================================

def load_dataset(
    name,
    microarray_dir=None,
    scrna_dir=None,
):
    """
    Load one dataset from the i-IF-Learn paper.

    Parameters
    ----------
    name : str
        Dataset name, for example:
        'leukemia', 'srbct', 'breast',
        'goolam', 'deng', 'li', etc.

    microarray_dir : str or Path, optional
        Folder containing the 10 microarray datasets.

    scrna_dir : str or Path, optional
        Folder containing the 8 scRNA-seq datasets.

    Returns
    -------
    X : np.ndarray
        Shape = n_samples x n_features

    y : np.ndarray
        True labels.

    K : int
        Number of clusters/classes.
    """

    name = name.lower().strip()

    # --------------------------------------------------------
    # Microarray datasets
    # --------------------------------------------------------

    if name in MICROARRAY_DATASETS:

        if microarray_dir is None:
            raise ValueError(
                "microarray_dir must be provided "
                "for microarray datasets."
            )

        base = Path(microarray_dir)
        config = MICROARRAY_DATASETS[name]

        x_path = base / config["x_file"]
        y_path = base / config["y_file"]

        X_raw = pd.read_csv(
            x_path,
            sep=config["sep"],
            header=None
        ).values

        y = _load_microarray_labels(
            y_path,
            config
        )

        X = _ensure_samples_by_features(
            X_raw,
            y
        )

    # --------------------------------------------------------
    # scRNA-seq datasets
    # --------------------------------------------------------

    elif name in SCRNA_DATASETS:

        if scrna_dir is None:
            raise ValueError(
                "scrna_dir must be provided "
                "for scRNA-seq datasets."
            )

        base = Path(scrna_dir)
        config = SCRNA_DATASETS[name]

        x_path = base / config["x_file"]
        y_path = base / config["y_file"]

        # scRNA expression matrices are comma-separated
        X_raw = pd.read_csv(
            x_path,
            sep=",",
            header=None
        ).values

        # scRNA labels are whitespace-separated
        y = pd.read_csv(
            y_path,
            sep=r"\s+",
            header=None
        ).values.ravel()

        X = _ensure_samples_by_features(
            X_raw,
            y
        )

        # Paper preprocessing:
        # log(X + 1) for all scRNA-seq datasets except Patel
        if name != "patel":
            X = np.log1p(X)

    else:
        valid_names = (
            list(MICROARRAY_DATASETS.keys())
            + list(SCRNA_DATASETS.keys())
        )

        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Valid datasets are: {valid_names}"
        )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=float
    )

    y = np.asarray(
        y
    ).ravel()

    K = len(
        np.unique(y)
    )

    return X, y, K