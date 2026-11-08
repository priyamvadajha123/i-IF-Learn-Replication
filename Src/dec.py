
import numpy as np
import torch

from torch import nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class DECAutoencoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return z, reconstructed
def pretrain_autoencoder(model, X_tensor, epochs=10, batch_size=256, learning_rate=0.001):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    loss_history = []

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0

        for (batch_X,) in loader:

            optimizer.zero_grad()

            _, reconstructed = model(batch_X)

            loss = criterion(reconstructed, batch_X)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)

        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)

        print(f"Pretraining Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

    return loss_history


class DECClusteringLayer(nn.Module):

    def __init__(self, cluster_centers):
        super().__init__()

        self.cluster_centers = nn.Parameter(
            torch.tensor(
                cluster_centers,
                dtype=torch.float32
            )
        )

    def forward(self, z):

        # Squared Euclidean distance between samples and centers
        distances = torch.cdist(
            z, self.cluster_centers
        ).pow(2)

        # Student's t-distribution similarities
        q = 1.0 / (1.0 + distances)

        # Normalize to obtain soft cluster assignments
        q = q / q.sum(dim=1, keepdim=True)

        return q

def target_distribution(q):

    # Total soft membership for each cluster
    cluster_frequency = q.sum(dim=0)

    # Square assignments and adjust for cluster frequency
    weight = q.pow(2) / cluster_frequency.clamp_min(1e-12)

    # Normalize each sample's target assignments
    p = weight / weight.sum(dim=1, keepdim=True)

    return p

import torch.nn.functional as F


def train_dec(
    model,
    clustering_layer,
    X_tensor,
    epochs=100,
    batch_size=256,
    learning_rate=0.001
):
    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(clustering_layer.parameters()),
        lr=learning_rate
    )

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    history = {
        "reconstruction_loss": [],
        "kl_loss": [],
        "total_loss": []
    }

    for epoch in range(epochs):

        # Update target distribution using current assignments
        model.eval()
        clustering_layer.eval()

        with torch.no_grad():
            z = model.encoder(X_tensor)
            q = clustering_layer(z)
            p = target_distribution(q).detach()

        model.train()
        clustering_layer.train()

        total_reconstruction = 0.0
        total_kl = 0.0
        total_samples = 0

        # Keep each target distribution aligned with its sample
        indices = torch.randperm(len(X_tensor))

        for batch_indices in indices.split(batch_size):

            batch_X = X_tensor[batch_indices]
            batch_p = p[batch_indices]

            optimizer.zero_grad()

            z, reconstructed = model(batch_X)
            batch_q = clustering_layer(z)

            reconstruction_loss = F.mse_loss(
                reconstructed, batch_X
            )

            kl_loss = F.kl_div(
                torch.log(batch_q.clamp_min(1e-12)),
                batch_p,
                reduction="batchmean"
            )

            loss = reconstruction_loss + kl_loss

            loss.backward()
            optimizer.step()

            n_batch = len(batch_X)
            total_reconstruction += reconstruction_loss.item() * n_batch
            total_kl += kl_loss.item() * n_batch
            total_samples += n_batch

        reconstruction_avg = total_reconstruction / total_samples
        kl_avg = total_kl / total_samples

        history["reconstruction_loss"].append(reconstruction_avg)
        history["kl_loss"].append(kl_avg)
        history["total_loss"].append(
            reconstruction_avg + kl_avg
        )

        if (epoch + 1) % 10 == 0:
            print(
                f"DEC Epoch {epoch + 1}/{epochs} | "
                f"Reconstruction: {reconstruction_avg:.6f} | "
                f"KL: {kl_avg:.6f} | "
                f"Total: {reconstruction_avg + kl_avg:.6f}"
            )

    return history

def run_dec(X, K, seed=42):
    """
    Run the complete DEC pipeline:
    1. Standardize features
    2. Pretrain autoencoder
    3. Initialize cluster centers using KMeans
    4. Refine using KL divergence + reconstruction loss
    5. Return final cluster assignments
    """

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Step 1: Standardize features
    X = np.asarray(X, dtype=np.float32)
    X_scaled = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Step 2: Initialize and pretrain autoencoder
    model = DECAutoencoder(input_dim=X.shape[1])

    pretrain_losses = pretrain_autoencoder(
        model,
        X_tensor,
        epochs=10,
        batch_size=256,
        learning_rate=0.001
    )

    # Step 3: Initialize clusters with KMeans
    model.eval()

    with torch.no_grad():
        initial_embedding = model.encoder(X_tensor).numpy()

    kmeans = KMeans(
        n_clusters=K,
        n_init=20,
        random_state=seed
    )

    initial_labels = kmeans.fit_predict(initial_embedding)

    clustering_layer = DECClusteringLayer(
        kmeans.cluster_centers_
    )

    with torch.no_grad():
        initial_q = clustering_layer(
            model.encoder(X_tensor)
        )

        initial_confidence = (
            initial_q.max(dim=1).values.mean().item()
        )

    # Step 4: DEC refinement
    history = train_dec(
        model,
        clustering_layer,
        X_tensor,
        epochs=100,
        batch_size=256,
        learning_rate=0.001
    )

    # Step 5: Final cluster assignments
    model.eval()
    clustering_layer.eval()

    with torch.no_grad():
        embedding = model.encoder(X_tensor)
        final_q = clustering_layer(embedding)

        labels = final_q.argmax(dim=1).numpy()

        final_confidence = (
            final_q.max(dim=1).values.mean().item()
        )

    return {
        "labels": labels,
        "initial_labels": initial_labels,
        "embedding": embedding.numpy(),
        "initial_confidence": initial_confidence,
        "final_confidence": final_confidence,
        "pretrain_losses": pretrain_losses,
        "history": history
    }
