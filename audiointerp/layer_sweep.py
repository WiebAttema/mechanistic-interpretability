"""Per-layer probing: a linear and an MLP probe on every layer of one or more models.

Each layer is z-scored with train-split statistics. Probes keep their best epoch on
validation UAR, and the CSV stores that validation UAR next to the test metrics: pick
layers by val_uar, report the test columns. Rows are written as they finish, and
--resume skips rows already in the CSV.

    python -m audiointerp.layer_sweep --dataset emodb --models wav2vec2-base
    python -m audiointerp.layer_sweep --dataset aibo --models wav2vec2-base whisper-small --layers cnn layer_6
"""
import argparse

import matplotlib.pyplot as plt

from audiointerp.activations import layer_matrix, layer_sort_key, load_pooled
from audiointerp.config import load_config
from audiointerp.data import DATASETS, split_indices
from audiointerp.plotting import model_color, save, style_axes
from audiointerp.probes import (
    PROBES, balanced_class_weights, predict, score_fields, test_scores, train_probe, zscore_stats,
)
from audiointerp.utils import CSVLog, pick_device, read_csv

PROBE_STYLES = {"linear": "-", "mlp": "--"}


def sweep_path(cfg, dataset, model):
    return cfg["paths"]["output_dir"] / "layer_sweep" / f"{dataset}_{model}.csv"


def sweep(model, args, cfg, device):
    data = load_pooled(cfg, args.dataset, model)
    y, class_names = data["labels"], data["label_names"]
    split = split_indices(data["speakers"], args.dataset)
    train, val, test = split["train"], split["val"], split["test"]
    weights = balanced_class_weights(y[train], len(class_names))
    print(f"\n=== {model}: {len(y)} clips, train {len(train)} / val {len(val)} / test {len(test)}")

    log = CSVLog(sweep_path(cfg, args.dataset, model),
                 ["model", "layer", "probe", "val_uar"] + score_fields(class_names), resume=args.resume)
    done = {(row["layer"], row["probe"]) for row in log.rows}
    for layer in args.layers or data["layer_names"]:
        X = layer_matrix(data, layer)
        mu, sd = zscore_stats(X[train])
        X = (X - mu) / sd
        for probe in args.probes:
            if (layer, probe) in done:
                continue
            clf, val_uar = train_probe(PROBES[probe](X.shape[1], len(class_names)),
                                       X[train], y[train], X[val], y[val], weights, device,
                                       epochs=args.epochs, batch_size=args.batch_size)
            row = {"model": model, "layer": layer, "probe": probe, "val_uar": val_uar,
                   **test_scores(y[test].numpy(), predict(clf, X[test], device), class_names)}
            log.write(row)
            print(f"  {layer:>9} | {probe:>6} | val uar {val_uar:.4f} | "
                  f"test uar {row['test_uar']:.4f}  acc {row['test_acc']:.4f}")
    log.close()
    return log.rows


def plot(rows, dataset, path):
    """Test UAR per layer, one line per (model, probe)."""
    layers = sorted({row["layer"] for row in rows}, key=layer_sort_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in dict.fromkeys(row["model"] for row in rows):
        for probe, style in PROBE_STYLES.items():
            uar = {r["layer"]: r["test_uar"] for r in rows if r["model"] == model and r["probe"] == probe}
            xs = [i for i, layer in enumerate(layers) if layer in uar]
            if xs:
                ax.plot(xs, [100 * uar[layers[i]] for i in xs], style, color=model_color(model),
                        linewidth=2, label=f"{model} ({probe})")
    ax.set_xticks(range(len(layers)), layers, rotation=90, fontsize=8)
    ax.set_ylabel("Test UAR (%)")
    ax.set_title(f"Per-layer probing on {dataset.upper()}")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe every layer of one or more models.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--layers", nargs="+", help="Layer names (default: every layer).")
    parser.add_argument("--probes", nargs="+", default=["linear", "mlp"], choices=list(PROBES))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true", help="Skip (layer, probe) rows already in the CSV.")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figure from existing CSVs.")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    print(f"Device: {device}")
    rows = []
    for model in args.models:
        rows += read_csv(sweep_path(cfg, args.dataset, model)) if args.plot_only else sweep(model, args, cfg, device)
    plot(rows, args.dataset, cfg["paths"]["output_dir"] / "layer_sweep" / f"{args.dataset}.png")
