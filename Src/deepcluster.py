
import random

import numpy as np
import optuna
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# 2. Autoencoder with classification head
# --------------------------------------------------

class DeepClusterAutoencoder(nn.Module):

    def __init__(self, input_dim, n_clusters,
                 hidden_dim=128, latent_dim=10):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.classifier = nn.Linear(
            latent_dim, n_clusters
        )

    def forward(self, x):

        z = self.encoder(x)
        reconstructed = self.decoder(z)
        logits = self.classifier(z)

        return z, reconstructed, logits


# --------------------------------------------------
# 3. Extract embeddings and generate pseudo-labels
# --------------------------------------------------

@torch.no_grad()
def get_embedding(model, X_tensor):

    model.eval()
    z = model.encoder(X_tensor).cpu().numpy()

    return normalize(z, norm="l2")


def generate_pseudo_labels(embedding, n_clusters, seed):

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=30,
        random_state=seed
    )

    labels = kmeans.fit_predict(embedding)

    return labels


# --------------------------------------------------
# 4. Train one DeepCluster configuration
# --------------------------------------------------

def train_deepcluster(
    X,
    n_clusters,
    hidden_dim=128,
    latent_dim=10,
    epochs=10,
    iterations=5,
    learning_rate=0.001,
    batch_size=64,
    seed=42
):

    set_seed(seed)

    device = torch.device("cpu")

    X_tensor = torch.tensor(
        X, dtype=torch.float32, device=device
    )

    model = DeepClusterAutoencoder(
        input_dim=X.shape[1],
        n_clusters=n_clusters,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    loss_history = []

    for iteration in range(iterations):

        # Step 1: Extract normalized embeddings
        embedding = get_embedding(model, X_tensor)

        # Step 2: Generate pseudo-labels with KMeans
        pseudo_labels = generate_pseudo_labels(
            embedding,
            n_clusters,
            seed + iteration
        )

        labels_tensor = torch.tensor(
            pseudo_labels,
            dtype=torch.long,
            device=device
        )

        dataset = TensorDataset(
            X_tensor, labels_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Step 3: Train using pseudo-labels
        for epoch in range(epochs):

            model.train()

            for batch_X, batch_labels in loader:

                optimizer.zero_grad()

                z, reconstructed, logits = model(batch_X)

                reconstruction_loss = reconstruction_criterion(
                    reconstructed, batch_X
                )

                classification_loss = classification_criterion(
                    logits, batch_labels
                )

                total_loss = (
                    reconstruction_loss + classification_loss
                )

                total_loss.backward()
                optimizer.step()

        # Evaluate the full-data objective after this iteration
        model.eval()

        with torch.no_grad():

            _, reconstructed, logits = model(X_tensor)

            reconstruction_loss = reconstruction_criterion(
                reconstructed, X_tensor
            )

            classification_loss = classification_criterion(
                logits, labels_tensor
            )

            final_loss = (
                reconstruction_loss + classification_loss
            ).item()

        loss_history.append(final_loss)

    # Final embedding after all training updates
    final_embedding = get_embedding(model, X_tensor)

    final_labels = generate_pseudo_labels(
        final_embedding,
        n_clusters,
        seed + iterations
    )

    return {
        "labels": final_labels,
        "embedding": final_embedding,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "model": model
    }


# --------------------------------------------------
# 5. Optuna hyperparameter optimization
# --------------------------------------------------

def run_deepcluster(
    X,
    K,
    seed=42,
    n_trials=10,
    latent_dim=10,
    batch_size=64
):

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("X must be a 2D data matrix.")

    if not 2 <= K < len(X):
        raise ValueError("K must be between 2 and n_samples - 1.")

    # Standardize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    def objective(trial):

        hidden_dim = trial.suggest_categorical(
            "hidden_dim", [64, 128, 256]
        )

        epochs = trial.suggest_int(
            "epochs", 5, 15
        )

        iterations = trial.suggest_int(
            "iterations", 3, 10
        )

        result = train_deepcluster(
            X=X_scaled,
            n_clusters=K,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            iterations=iterations,
            learning_rate=0.001,
            batch_size=batch_size,
            seed=seed + trial.number
        )

        labels = result["labels"]

        # Silhouette requires at least 2 distinct clusters,
        # with fewer clusters than samples.
        n_found = len(np.unique(labels))

        if not 2 <= n_found < len(labels):
            return -1e9

        silhouette = silhouette_score(
            result["embedding"],
            labels,
            metric="euclidean"
        )

        score = (
            silhouette - 0.5 * result["final_loss"]
        )

        return float(score)

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    best_params = study.best_params

    # Retrain using the best configuration
    result = train_deepcluster(
        X=X_scaled,
        n_clusters=K,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=latent_dim,
        epochs=best_params["epochs"],
        iterations=best_params["iterations"],
        learning_rate=0.001,
        batch_size=batch_size,
        seed=seed
    )

    result["best_params"] = best_params
    result["best_optuna_score"] = study.best_value

    return result
