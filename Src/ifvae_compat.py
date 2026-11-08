
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import norm
from sklearn.cluster import KMeans


def feature_selection_ifvae(Data, per=1, seed=42, chunk_size=512):
    """
    KS + Higher Criticism feature selection.

    Port of the feature_selection() procedure from
    Chen, Jin & Ke (2023).

    Data shape: (features, samples).
    """
    if not 0 < per <= 1:
        raise ValueError("per must be in (0, 1].")

    X = np.asarray(Data, dtype=float)
    p, n = X.shape

    if n < 3 or p < 4:
        raise ValueError("Insufficient data dimensions.")

    # Original feature-wise normalization
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, ddof=1, keepdims=True)

    if np.any(stds == 0):
        raise ValueError("Constant features detected.")

    W = (X - means) / stds

    # Simulate 50*p KS null statistics
    rep = 50 * p
    rng = np.random.RandomState(seed)
    KSvalue = np.empty(rep)

    kk_left = np.arange(n) / n
    kk_right = np.arange(1, n + 1) / n

    for start in range(0, rep, chunk_size):
        end = min(start + chunk_size, rep)
        z = rng.randn(end - start, n)

        z = (z - z.mean(axis=1, keepdims=True)) / (
            z.std(axis=1, keepdims=True)
        )
        z /= np.sqrt(1 - 1 / n)

        pi = np.sort(norm.cdf(z), axis=1)

        KSvalue[start:end] = np.maximum(
            np.max(np.abs(pi - kk_left), axis=1),
            np.max(np.abs(pi - kk_right), axis=1)
        )

    KSvalue *= np.sqrt(n)

    if per < 1:
        null_subset = np.sort(KSvalue)[:round(per * rep)]
    else:
        null_subset = KSvalue

    KSmean = null_subset.mean()
    KSstd = null_subset.std()

    # Observed KS statistics
    KS = np.empty(p)

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)

        pi = np.sort(
            norm.cdf(W[start:end] / np.sqrt(1 - 1 / n)),
            axis=1
        )

        KS[start:end] = np.maximum(
            np.max(np.abs(pi - kk_left), axis=1),
            np.max(np.abs(pi - kk_right), axis=1)
        )

    if per == 1:
        KS = (KS - KS.mean()) / KS.std()
    else:
        observed_subset = np.sort(KS)[:round(per * p)]
        KS = (KS - observed_subset.mean()) / observed_subset.std()

    KSadjust = KS * KSstd + KSmean

    # Empirical null p-values, as in original source
    null_sorted = np.sort(KSvalue)
    pval = (
        rep - np.searchsorted(
            null_sorted, KSadjust, side="right"
        )
    ) / rep

    psort = np.sort(pval)
    ranking = np.argsort(pval)

    # Higher Criticism
    kk = np.arange(1, p + 1) / (p + 1)

    HCsort = np.sqrt(p) * (kk - psort) / np.sqrt(kk)
    HCsort /= np.sqrt(
        np.maximum(np.sqrt(n) * (kk - psort) / kk, 0) + 1
    )

    pvalcut = np.log(p) / p
    candidates = np.flatnonzero(psort > pvalcut)

    if len(candidates) == 0:
        raise ValueError("No valid Higher Criticism cutoff.")

    first = candidates[0]
    HCsort[:first] = -np.inf
    HCsort[round(p / 2):] = -np.inf

    if not np.any(np.isfinite(HCsort)):
        raise ValueError("No valid Higher Criticism candidates.")

    # Preserve original zero-based cutoff convention
    numselect = np.flatnonzero(
        HCsort == np.max(HCsort)
    )[-1]

    return ranking, int(numselect)


class IFVAEModel(tf.keras.Model):
    """
    Original VAE architecture:
    Dense -> BatchNorm -> ReLU for mean and log variance,
    reparameterization, followed by a sigmoid decoder.
    """

    def __init__(self, original_dim, latent_dim=25):
        super().__init__()

        self.mean_dense = tf.keras.layers.Dense(
            latent_dim, kernel_initializer="glorot_uniform"
        )
        self.mean_bn = tf.keras.layers.BatchNormalization()
        self.mean_relu = tf.keras.layers.ReLU()

        self.logvar_dense = tf.keras.layers.Dense(
            latent_dim, kernel_initializer="glorot_uniform"
        )
        self.logvar_bn = tf.keras.layers.BatchNormalization()
        self.logvar_relu = tf.keras.layers.ReLU()

        self.decoder = tf.keras.layers.Dense(
            original_dim,
            activation="sigmoid",
            kernel_initializer="glorot_uniform"
        )

    def encode(self, x, training=False):
        mu = self.mean_relu(
            self.mean_bn(self.mean_dense(x), training=training)
        )

        logvar = self.logvar_relu(
            self.logvar_bn(
                self.logvar_dense(x), training=training
            )
        )

        return mu, logvar

    def call(self, x, training=False):
        mu, logvar = self.encode(x, training=training)

        epsilon = tf.random.normal(tf.shape(mu))
        z = mu + tf.exp(logvar / 2) * epsilon

        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


def train_ifvae_embedding(
    X,
    latent_dim=25,
    batch_size=50,
    epochs=100,
    learning_rate=0.0005,
    seed=42
):
    """Train the VAE and return encoder means."""

    tf.keras.utils.set_random_seed(seed)

    X = np.asarray(X, dtype=np.float32)
    n, p = X.shape

    # Original 90/10 random split
    rng = np.random.RandomState(seed)
    n_test = round(0.1 * n)
    test_idx = rng.choice(n, n_test, replace=False)

    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    X_train = X[train_mask]

    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(
        len(X_train), seed=seed, reshuffle_each_iteration=True
    ).batch(batch_size)

    model = IFVAEModel(p, latent_dim=latent_dim)

    # Build model before training
    model(tf.convert_to_tensor(X[:1]), training=False)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    beta = 0.0

    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                reconstructed, mu, logvar = model(
                    batch, training=True
                )

                # Original BCE reconstruction expression
                reconstruction_loss = p * tf.reduce_mean(
                    tf.keras.backend.binary_crossentropy(
                        batch, reconstructed
                    ),
                    axis=1
                )

                kl_loss = -0.5 * tf.reduce_sum(
                    1 + logvar - tf.square(mu) - tf.exp(logvar),
                    axis=1
                )

                loss = tf.reduce_mean(
                    reconstruction_loss + beta * kl_loss
                )

            grads = tape.gradient(
                loss, model.trainable_variables
            )

            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

        # Preserve original WarmUpCallback behavior:
        # beta: 0 -> 1 -> 2, then stays at 2.
        if beta <= 1:
            beta += 1

    # Use encoder mean for clustering, not sampled z
    mu, _ = model.encode(
        tf.convert_to_tensor(X), training=False
    )

    return mu.numpy()


def run_ifvae(X, K, seed=42, per=1):
    """
    IF-VAE baseline based on Chen, Jin & Ke (2023).

    X shape: (samples, features), matching data_loader.py.
    Returns predicted cluster labels and diagnostic details.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    # Feature-wise sample standardization
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)

    if np.any(stds == 0):
        raise ValueError("Constant features detected.")

    W = (X - means) / stds

    # Original IF-selection expects (features, samples)
    ranking, numselect = feature_selection_ifvae(
        pd.DataFrame(W.T), per=per, seed=seed
    )

    if numselect < 1:
        raise ValueError("Feature selection returned no features.")

    selected = ranking[:numselect]

    # VAE expects samples x selected features
    X_selected = W[:, selected]

    embedding = train_ifvae_embedding(
        X_selected,
        latent_dim=25,
        batch_size=50,
        epochs=100,
        learning_rate=0.0005,
        seed=seed
    )

    labels = KMeans(
        init="random",
        n_clusters=K,
        n_init=10,
        max_iter=300,
        random_state=42
    ).fit_predict(embedding)

    return {
        "labels": labels,
        "embedding": embedding,
        "selected_features": selected,
        "n_selected": len(selected)
    }
