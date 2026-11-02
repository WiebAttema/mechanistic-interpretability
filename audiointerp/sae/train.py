"""Train TopK SAEs per (model, layer, sparsity) and check the codes still carry the task:
test reconstruction MSE, and a linear probe on the codes compared with the probe on the
original layer (Mariotte et al. Fig. 1). Rows are appended as they finish, so pass
--resume after a time limit. The checkpoints feed the other sae scripts.

    python -m audiointerp.sae.train --dataset aibo --models wav2vec2-base
    python -m audiointerp.sae.train --dataset aibo --models wav2vec2-base --layers layer_8 --sparsities 90 99
"""
import argparse

import matplotlib.pyplot as plt

from audiointerp.activations import layer_matrix, load_pooled
from audiointerp.config import load_config
from audiointerp.data import DATASETS, split_indices
from audiointerp.plotting import model_color, save, style_axes
from audiointerp.probes import (
    LinearProbe, balanced_class_weights, predict, score_fields, test_scores, train_probe, zscore_stats,
)
from audiointerp.sae.model import (
    checkpoint_path, encode, reconstruction_mse, resolve_layer, sae_dir, save_checkpoint,
    sparsity_to_k, sweep_reference_uar, train_sae,
)
from audiointerp.utils import CSVLog, pick_device

FIELDS = ["model", "layer", "is_final", "sparsity", "k", "n_dict", "best_epoch",
          "val_mse", "test_mse", "dead_frac"]


def train_model(model, data, args, cfg, device, log):
    done = {(row["model"], row["layer"], row["sparsity"]) for row in log.rows}
    y, class_names = data["labels"], data["label_names"]
    split = split_indices(data["speakers"], args.dataset)
    train, val, test = split["train"], split["val"], split["test"]
    weights = balanced_class_weights(y[train], len(class_names))
    layers = dict.fromkeys(resolve_layer(spec, cfg, args.dataset, model, data["layer_names"])
                           for spec in args.layers)

    for layer in layers:
        X = layer_matrix(data, layer)
        mu, sd = zscore_stats(X[train])
        print(f"\n=== {model} {layer}, dim {X.shape[1]}, dict {args.dict_size}")
        for sparsity in args.sparsities:
            if (model, layer, sparsity) in done:
                continue
            k = sparsity_to_k(sparsity, args.dict_size)
            sae, info = train_sae((X[train] - mu) / sd, (X[val] - mu) / sd, args.dict_size, k,
                                  epochs=args.epochs, device=device, seed=args.seed)
            Z = {name: encode(sae, X[idx], mu, sd, device) for name, idx in split.items()}
            probe, _ = train_probe(LinearProbe(args.dict_size, len(class_names)),
                                   Z["train"], y[train], Z["val"], y[val], weights, device,
                                   epochs=args.probe_epochs)
            row = {"model": model, "layer": layer, "is_final": layer == data["layer_names"][-1],
                   "sparsity": sparsity, "k": k, "n_dict": args.dict_size,
                   "best_epoch": info["best_epoch"], "val_mse": info["val_mse"],
                   "test_mse": reconstruction_mse(sae, X[test], mu, sd, device),
                   "dead_frac": info["dead_frac"],
                   **test_scores(y[test].numpy(), predict(probe, Z["test"], device), class_names)}
            log.write(row)
            save_checkpoint(checkpoint_path(cfg, args.dataset, model, layer, sparsity), sae, mu, sd,
                            {"dataset": args.dataset, "model": model, "layer": layer,
                             "sparsity": sparsity, "val_mse": info["val_mse"], "seed": args.seed})
            print(f"  s={sparsity}% k={k:>5} | mse {row['test_mse']:.5f}  dead {info['dead_frac']:.2f}"
                  f" | probe uar {row['test_uar']:.4f}  acc {row['test_acc']:.4f}")


def plot(rows, cfg, dataset):
    """Reconstruction MSE and probe UAR vs sparsity; dashed = probe on the original layer,
    dotted lines = final layer."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for model, layer in dict.fromkeys((row["model"], row["layer"]) for row in rows):
        sub = sorted((r for r in rows if (r["model"], r["layer"]) == (model, layer)),
                     key=lambda r: r["sparsity"])
        style = ":" if sub[0]["is_final"] else "-"
        xs = [r["sparsity"] for r in sub]
        axes[0].plot(xs, [r["test_mse"] for r in sub], style, color=model_color(model), linewidth=2)
        axes[1].plot(xs, [100 * r["test_uar"] for r in sub], style, color=model_color(model),
                     linewidth=2, label=f"{model} {layer}")
        reference = sweep_reference_uar(cfg, dataset, model, layer)
        if reference is not None:
            axes[1].axhline(100 * reference, color=model_color(model), linewidth=1, linestyle="--", alpha=0.45)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Reconstruction MSE (test)")
    axes[1].set_ylabel("Probe UAR on sparse codes (%)")
    for ax, title in zip(axes, ["SAE reconstruction", "SAE probing"]):
        ax.set_xlabel("Sparsity (%)")
        ax.set_title(title, fontsize=11.5)
        style_axes(ax)
    axes[1].legend(frameon=False, fontsize=7.5)
    save(fig, sae_dir(cfg, dataset) / "sae_training.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate TopK SAEs.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--layers", nargs="+", default=["best", "final"],
                        help="'best' (from the layer sweep), 'final', or layer names.")
    parser.add_argument("--sparsities", nargs="+", type=int, default=[75, 80, 85, 90, 95, 99])
    parser.add_argument("--dict-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    print(f"Device: {device}")
    log = None
    for model in args.models:
        data = load_pooled(cfg, args.dataset, model)
        if log is None:  # the recall columns need the class names
            log = CSVLog(sae_dir(cfg, args.dataset) / "sae_training.csv",
                         FIELDS + score_fields(data["label_names"]), resume=args.resume)
        train_model(model, data, args, cfg, device, log)
    log.close()
    plot(log.rows, cfg, args.dataset)
