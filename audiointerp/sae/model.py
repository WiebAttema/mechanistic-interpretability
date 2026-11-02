"""TopK sparse autoencoder (Gao et al. 2025), as used by Mariotte et al. 2025:

    z     = TopK(ReLU(W_e x + b_e))
    x_hat = W_d z
    loss  = MSE(x_hat, x)

Tied initialisation and unit-norm decoder rows are standard TopK stabilisers the paper
does not spell out. Inputs are z-scored with train-split statistics, which are stored
in the checkpoint so codes are always computed the same way.

Known limitation: dead units are never resampled, so longer training loses more units
rather than fewer.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from audiointerp.activations import layer_matrix
from audiointerp.data import split_indices
from audiointerp.layer_sweep import sweep_path
from audiointerp.utils import read_csv


class TopKSAE(nn.Module):
    def __init__(self, d_in, n_dict, k):
        super().__init__()
        self.d_in, self.n_dict, self.k = d_in, n_dict, k
        self.encoder = nn.Linear(d_in, n_dict)
        decoder = self.encoder.weight.detach().clone()  # (n_dict, d_in): tied init
        self.decoder = nn.Parameter(decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def encode(self, x):
        a = torch.relu(self.encoder(x))
        top_vals, top_idx = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, top_idx, top_vals)

    def decode(self, z):
        return z @ self.decoder

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renorm_decoder_(self):
        self.decoder.data /= self.decoder.data.norm(dim=1, keepdim=True).clamp_min(1e-8)


def sparsity_to_k(sparsity_pct, n_dict):
    """Active units per input, e.g. 90% sparsity with n_dict=4096 -> k=409."""
    return max(1, int((1 - sparsity_pct / 100.0) * n_dict))


def train_sae(X_train, X_val, n_dict, k, epochs=150, lr=1e-3, batch_size=32, device="cpu", seed=0):
    """Adam, lr 1e-3, batch 32 as in the paper, on standardized inputs. Keeps the epoch
    with the lowest validation MSE."""
    torch.manual_seed(seed)
    model = TopKSAE(X_train.shape[1], n_dict, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train.to(device)), batch_size=batch_size, shuffle=True)
    X_val = X_val.to(device)

    best_mse, best_state, best_epoch = math.inf, None, -1
    for epoch in range(epochs):
        model.train()
        for (xb,) in loader:
            optimizer.zero_grad()
            x_hat, _ = model(xb)
            F.mse_loss(x_hat, xb).backward()
            optimizer.step()
            model.renorm_decoder_()

        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(model(X_val)[0], X_val).item()
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dead_frac = ((model.encode(X_val) > 0).sum(dim=0) == 0).float().mean().item()
    return model, {"val_mse": best_mse, "best_epoch": best_epoch, "dead_frac": dead_frac}


@torch.no_grad()
def encode(model, X, mu, sd, device, batch_size=2048):
    """Standardize with the stored train statistics and return the codes on the CPU."""
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model.encode(((X[i:i + batch_size] - mu) / sd).to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def reconstruction_mse(model, X, mu, sd, device, batch_size=2048):
    total, count = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb = ((X[i:i + batch_size] - mu) / sd).to(device)
        total += F.mse_loss(model(xb)[0], xb, reduction="sum").item()
        count += xb.numel()
    return total / count


# Files: everything lives under <output_dir>/sae/<dataset>/.

def sae_dir(cfg, dataset):
    return cfg["paths"]["output_dir"] / "sae" / dataset


def checkpoint_path(cfg, dataset, model, layer, sparsity):
    return sae_dir(cfg, dataset) / "checkpoints" / f"{model}_{layer}_s{sparsity}.pt"


def lasso_coefs_path(cfg, dataset, model, layer, sparsity):
    return sae_dir(cfg, dataset) / "lasso_coefs" / f"{model}_{layer}_s{sparsity}.npz"


def save_checkpoint(path, model, mu, sd, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**meta, "state": model.state_dict(), "d_in": model.d_in, "n_dict": model.n_dict,
                "k": model.k, "mu": mu.cpu(), "sd": sd.cpu()}, path)


def load_checkpoint(path, device="cpu"):
    """The SAE and the checkpoint dict (meta, mu, sd)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TopKSAE(payload["d_in"], payload["n_dict"], payload["k"])
    model.load_state_dict(payload["state"])
    return model.to(device).eval(), payload


def resolve_layer(spec, cfg, dataset, model, layer_names):
    """'final' = the last layer; 'best' = the layer whose linear probe has the highest
    validation UAR in the layer sweep; anything else is a layer name."""
    if spec == "final":
        return layer_names[-1]
    if spec == "best":
        path = sweep_path(cfg, dataset, model)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run audiointerp.layer_sweep first or pass a layer name.")
        rows = [row for row in read_csv(path) if row["probe"] == "linear"]
        return max(rows, key=lambda row: row["val_uar"])["layer"]
    if spec not in layer_names:
        raise ValueError(f"No layer {spec!r}; available: {layer_names}")
    return spec


def sweep_reference_uar(cfg, dataset, model, layer):
    """Test UAR of the linear probe on the original layer, the reference for 'do the
    sparse codes keep the task information'. None if the sweep has no such row."""
    path = sweep_path(cfg, dataset, model)
    if path.exists():
        for row in read_csv(path):
            if row["probe"] == "linear" and row["layer"] == layer:
                return row["test_uar"]
    return None


def load_codes(data, cfg, dataset, model, layer, sparsity, device):
    """The SAE of one (model, layer, sparsity), plus its codes and the labels per split."""
    sae, meta = load_checkpoint(checkpoint_path(cfg, dataset, model, layer, sparsity), device)
    X = layer_matrix(data, layer)
    split = split_indices(data["speakers"], dataset)
    Z = {name: encode(sae, X[idx], meta["mu"], meta["sd"], device) for name, idx in split.items()}
    y = {name: data["labels"][idx] for name, idx in split.items()}
    return sae, Z, y
