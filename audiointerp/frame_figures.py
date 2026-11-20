"""Figures for the frame analysis, drawn from saved results only:

    <dataset>_<model>_regression.png       best single channel vs all channels, per conv layer
    <dataset>_<model>_disentanglement.png  effective number of channels vs SAE units per descriptor
    <dataset>_<model>_sae_training.png     SAE training curves from train_frames logs (with --logs)

    python -m audiointerp.frame_figures --dataset aibo --model wav2vec2-base \
        --logs outputs/logs/sae_before_fix.out outputs/logs/sae_after_fix.out
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from audiointerp.config import load_config
from audiointerp.data import DATASETS
from audiointerp.frame_correlation import ACTIVE, ALPHA_RATIO, F0
from audiointerp.plotting import FALLBACK_COLOR, GRID_COLOR, model_color, save, style_axes
from audiointerp.sae.model import sae_dir

DESCRIPTORS = [("Loudness_sma3", "Loudness"), (ALPHA_RATIO, "Alpha ratio"), (F0, "Pitch (F0)")]


def short(name):
    return name.replace("_sma3nz", "").replace("_sma3", "")


def plot_regression(d, model, path):
    """Per conv layer: r² of the best active channel and held-out R² of all channels together."""
    layers, x = d["layers"], range(len(d["layers"]))
    fig, axes = plt.subplots(1, len(DESCRIPTORS), figsize=(12, 3.8), sharey=True)
    for ax, (name, title) in zip(axes, DESCRIPTORS):
        i = d["names"].index(name)
        single = [d["r"]["A"][l][d["norm_gain"][l].abs() > ACTIVE, i].pow(2).max().item() for l in layers]
        combined = [d["r2"][l][i].item() for l in layers]
        ax.plot(x, combined, "-o", color=model_color(model), linewidth=2, label="all channels (held-out R²)")
        ax.plot(x, single, "--o", color=FALLBACK_COLOR, linewidth=2, label="best single channel (r²)")
        ax.set_xticks(list(x), layers, rotation=45, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        style_axes(ax)
    axes[0].set_ylabel("Share of the descriptor explained")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{model}: one channel vs all channels, per conv layer")
    save(fig, path)


def plot_disentanglement(d, model, path, layer="cnn"):
    """Per descriptor, how many channels and SAE units the Lasso spreads it over; descriptors
    that neither predicts well (R² below 0.3) are faint, their concentration means little."""
    names, c, u = d["names"], d["scores"][layer]["channels"], d["scores"][layer]["units"]
    order = sorted(range(len(names)), key=lambda i: u[0, i].item())
    fig, ax = plt.subplots(figsize=(7, 7.5))
    for row, i in enumerate(order):
        alpha = 1.0 if max(c[0, i], u[0, i]) >= 0.3 else 0.3
        ax.plot([c[2, i], u[2, i]], [row, row], color=GRID_COLOR, linewidth=2, alpha=alpha, zorder=1)
        ax.scatter(c[2, i], row, color=FALLBACK_COLOR, alpha=alpha, zorder=2,
                   label="512 channels" if row == len(order) - 1 else None)
        ax.scatter(u[2, i], row, color=model_color(model), alpha=alpha, zorder=2,
                   label="4096 SAE units" if row == len(order) - 1 else None)
    ax.set_yticks(range(len(order)), [f"{short(names[i])}  (R² {u[0, i]:.2f})" for i in order], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Effective number of channels or units carrying the descriptor (log scale)")
    ax.set_title(f"{model} {layer}: Lasso per descriptor, sorted by the SAE's R²\n"
                 "faint = predicted poorly by both (R² below 0.3)", fontsize=10)
    style_axes(ax, grid="x")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, path)


def read_log(path):
    """(epoch, train mse, val mse) from the per-epoch lines train_frames prints."""
    rows = re.findall(r"epoch +(\d+): train mse ([\d.]+), val mse ([\d.]+)", Path(path).read_text())
    return [(int(e), float(t), float(v)) for e, t, v in rows]


def plot_training(logs, path):
    fig, axes = plt.subplots(1, len(logs), figsize=(5 * len(logs), 3.6), sharey=True, squeeze=False)
    for ax, log in zip(axes[0], logs):
        epochs, train, val = zip(*read_log(log))
        ax.plot(epochs, train, color=FALLBACK_COLOR, linewidth=2, label="training frames")
        ax.plot(epochs, val, color=model_color("wav2vec2-base"), linewidth=2, label="validation speakers")
        ax.set_yscale("log")
        ax.set_xlabel("Pass over the data (epoch)")
        ax.set_title(Path(log).stem, fontsize=10)
        style_axes(ax)
    axes[0][0].set_ylabel("Reconstruction error (MSE, log scale)")
    axes[0][0].legend(frameon=False, fontsize=8)
    save(fig, path)


def main(args):
    cfg = load_config(args.config)
    out = cfg["paths"]["output_dir"] / "figures" / f"{args.dataset}_{args.model}"
    d = torch.load(cfg["paths"]["output_dir"] / "frame_correlation" / f"{args.dataset}_{args.model}.pt",
                   weights_only=True)
    plot_regression(d, args.model, out.with_name(out.name + "_regression.png"))
    dis = sae_dir(cfg, args.dataset) / f"disentangle_frames_{args.model}_s{args.sparsity}.pt"
    if dis.exists():
        plot_disentanglement(torch.load(dis, weights_only=True), args.model,
                             out.with_name(out.name + "_disentanglement.png"))
    else:
        print(f"No {dis}; run audiointerp.sae.disentangle_frames (or copy it from the cluster).")
    if args.logs:
        plot_training(args.logs, out.with_name(out.name + "_sae_training.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figures of the frame analysis from saved results.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sparsity", type=int, default=99)
    parser.add_argument("--logs", nargs="+", help="train_frames job logs, one panel each (e.g. before and after a fix).")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
