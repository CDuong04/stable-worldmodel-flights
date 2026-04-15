"""Train state decoder: latent embedding -> 17D physical state."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class StateDecoder(nn.Module):
    """MLP decoder: latent embedding -> 17D physical state."""

    def __init__(self, embed_dim, state_dim=17, architecture="mlp_2", hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.architecture = architecture

        if architecture == "linear":
            self.net = nn.Linear(embed_dim, state_dim)
        elif architecture == "mlp_2":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )
        elif architecture == "mlp_3":
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, z):
        return self.net(z)


class LyapunovFunction(nn.Module):
    """V(x) = ||target_rel||^2 + alpha*||v||^2 + beta*||omega||^2."""

    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        target_rel = x[:, 14:17]
        vel = x[:, 3:6]
        ang_vel = x[:, 10:13]
        V = (target_rel ** 2).sum(dim=-1) + \
            self.alpha * (vel ** 2).sum(dim=-1) + \
            self.beta * (ang_vel ** 2).sum(dim=-1)
        return V


def train_decoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = os.path.join("data/expert_trajectories", "expert_default_gnc.npz")
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    obs = data["obs"]
    actions = data["actions"]
    print(f"Loaded {obs.shape[0]} samples, obs={obs.shape}, actions={actions.shape}")

    embed_dim = obs.shape[1]

    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(obs, dtype=torch.float32)

    n_total = len(X)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    dataset = TensorDataset(X, Y)
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    decoder = StateDecoder(embed_dim, state_dim=17, architecture=args.architecture,
                           hidden_dim=args.hidden_dim).to(device)
    lyapunov = LyapunovFunction(alpha=args.alpha, beta=args.beta).to(device)

    print(f"Decoder: {args.architecture}, hidden={args.hidden_dim}, "
          f"params={sum(p.numel() for p in decoder.parameters())}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    dim_names = ["px", "py", "pz", "vx", "vy", "vz",
                 "qw", "qx", "qy", "qz", "wx", "wy", "wz",
                 "fuel", "tx", "ty", "tz"]

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        decoder.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = decoder(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        decoder.eval()
        val_loss = 0.0
        per_dim_mse = torch.zeros(17, device=device)
        lyap_mse = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = decoder(xb)
                val_loss += nn.functional.mse_loss(pred, yb).item() * len(xb)
                per_dim_mse += ((pred - yb) ** 2).sum(dim=0)
                v_pred = lyapunov(pred)
                v_true = lyapunov(yb)
                lyap_mse += ((v_pred - v_true) ** 2).sum().item()

        val_loss /= n_val
        per_dim_mse /= n_val
        lyap_mse /= n_val
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "decoder": decoder.state_dict(),
                "architecture": args.architecture,
                "embed_dim": embed_dim,
                "hidden_dim": args.hidden_dim,
                "epoch": epoch,
                "val_loss": val_loss,
            }, os.path.join(args.save_dir, "state_decoder_best.pt"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"lyap_mse={lyap_mse:.4f}")
            dim_strs = [f"{dim_names[i]}={per_dim_mse[i]:.4f}" for i in range(17)]
            print(f"  Per-dim MSE: {', '.join(dim_strs[:6])}")
            print(f"               {', '.join(dim_strs[6:13])}")
            print(f"               {', '.join(dim_strs[13:])}")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved to {args.save_dir}/state_decoder_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--architecture", type=str, default="mlp_2",
                        choices=["linear", "mlp_2", "mlp_3"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="checkpoints/decoder")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_decoder(args)


if __name__ == "__main__":
    main()
