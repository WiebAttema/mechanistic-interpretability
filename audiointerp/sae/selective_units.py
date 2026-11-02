"""Class-selective SAE units and a causal ablation test.

Per model: score every dictionary unit by how much more it fires on one class than on
the rest (train split only, so the test ablation stays honest), name the top units by
the eGeMAPS factors whose Lasso weights load on them (if disentanglement.py has run),
then zero a class's top units at test time and check that this class's recall drops
while the others hold.

Selective-unit counts depend on --threshold and --min-active, so both are written into
the summary CSV next to the counts.

    python -m audiointerp.sae.selective_units --dataset aibo --models wav2vec2-base
    python -m audiointerp.sae.selective_units --dataset aibo --models wav2vec2-base --targets emphatic
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score

from audiointerp.activations import load_pooled
from audiointerp.config import load_config
from audiointerp.data import DATASETS
from audiointerp.plotting import model_color, save, style_axes
from audiointerp.probes import LinearProbe, balanced_class_weights, predict, train_probe
from audiointerp.sae.model import lasso_coefs_path, load_codes, resolve_layer, sae_dir
from audiointerp.utils import pick_device, write_csv


def load_unit_factors(path):
    """(factors x units) Lasso weights and factor names from disentanglement.py, or None."""
    if not path.exists():
        return None
    data = np.load(path)
    return data["coefs"], list(data["factor_names"])


def top_factors_for_unit(unit, coefs, factor_names, n=3):
    """The n eGeMAPS factors whose Lasso weights load most on this unit."""
    weights = np.abs(coefs[:, unit])
    return "; ".join(factor_names[i] for i in np.argsort(weights)[::-1][:n] if weights[i] > 1e-12)


def class_selectivity(Z_train, y_train, num_classes):
    """selectivity[c, j] in [-1, 1]: how much more unit j fires on class c than on the
    rest; active[c, j]: the fraction of class-c clips on which it fires at all."""
    selectivity = np.zeros((num_classes, Z_train.shape[1]))
    active = np.zeros_like(selectivity)
    for c in range(num_classes):
        mask = (y_train == c).numpy()
        mu_c = Z_train[mask].mean(dim=0).numpy()
        mu_rest = Z_train[~mask].mean(dim=0).numpy()
        selectivity[c] = (mu_c - mu_rest) / (mu_c + mu_rest + 1e-9)
        active[c] = (Z_train[mask] > 0).float().mean(dim=0).numpy()
    return selectivity, active


def ablation_curve(probe, Z_test, y_test, ranked, steps, num_classes, device):
    """Per-class test recall after zeroing the top-m ranked units, for m in [0] + steps."""
    curve = {}
    for m in [0] + steps:
        codes = Z_test.clone()
        if m:
            codes[:, ranked[:m]] = 0.0
        curve[m] = recall_score(y_test.numpy(), predict(probe, codes, device), average=None,
                                labels=np.arange(num_classes), zero_division=0)
    return curve


def fit_code_probe(Z, y, n_dict, num_classes, device, epochs):
    """Linear probe on the SAE codes: the classifier every ablation is measured with."""
    weights = balanced_class_weights(y["train"], num_classes)
    probe, _ = train_probe(LinearProbe(n_dict, num_classes), Z["train"], y["train"], Z["val"], y["val"],
                           weights, device, epochs=epochs)
    return probe


def plot_ablation(panels, target, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), squeeze=False, sharey=True)
    for ax, (model, layer, class_names, curve) in zip(axes[0], panels):
        ms = sorted(curve)
        for c, name in enumerate(class_names):
            is_target = name == target
            ax.plot(ms, [100 * curve[m][c] for m in ms], "-o" if is_target else "-",
                    linewidth=2.4 if is_target else 1.2, markersize=4,
                    color=model_color(model) if is_target else "#b9bec6",
                    label=name if is_target else None)
        ax.set_title(f"{model} {layer}", fontsize=10)
        ax.set_xlabel(f"Top {target} units ablated")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("Recall (%)")
    fig.suptitle(f"Ablating {target}-selective SAE units\n(grey: the other classes)", fontsize=11.5)
    save(fig, out / f"selective_units_ablation_{target}.png")


def plot_counts(counts, class_names, args, out):
    """Selective units per class and model, at the chosen thresholds."""
    models = list(counts)
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 3.4))
    for i, model in enumerate(models):
        ax.bar(np.arange(len(class_names)) + (i - (len(models) - 1) / 2) * width,
               [counts[model][c] for c in class_names], width, color=model_color(model), label=model)
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_ylabel("Selective units")
    ax.set_title(f"Class-selective units (selectivity > {args.threshold}, active > {args.min_active})",
                 fontsize=10.5)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "selective_units_counts.png")


def main(args):
    cfg = load_config(args.config)
    device = pick_device()
    out = sae_dir(cfg, args.dataset)
    unit_rows, summary, ablation_rows, panels, counts_by_model = [], [], {}, {}, {}

    for model in args.models:
        data = load_pooled(cfg, args.dataset, model)
        layer = resolve_layer(args.layer, cfg, args.dataset, model, data["layer_names"])
        sae, Z, y = load_codes(data, cfg, args.dataset, model, layer, args.sparsity, device)
        class_names = data["label_names"]
        num_classes = len(class_names)
        targets = class_names if args.targets == ["all"] else args.targets
        if set(targets) - set(class_names):
            sys.exit(f"Unknown target class(es) {sorted(set(targets) - set(class_names))}; options: {class_names}")
        print(f"\n=== {model} {layer}, {args.sparsity}% sparsity, dict {sae.n_dict}")
        naming = load_unit_factors(lasso_coefs_path(cfg, args.dataset, model, layer, args.sparsity))

        # 1. Selectivity per class, with units ranked from most to least selective.
        selectivity, active = class_selectivity(Z["train"], y["train"], num_classes)
        ranked = {c: [j for j in np.argsort(selectivity[c])[::-1] if active[c][j] > args.min_active]
                  for c in range(num_classes)}
        counts = {}
        for c, name in enumerate(class_names):
            counts[name] = int(((selectivity[c] > args.threshold) & (active[c] > args.min_active)).sum())
            for rank, j in enumerate(ranked[c][: args.top_units]):
                unit_rows.append({"model": model, "layer": layer, "sparsity": args.sparsity,
                                  "class": name, "rank": rank, "unit": int(j),
                                  "selectivity": float(selectivity[c][j]),
                                  "active_rate_class": float(active[c][j]),
                                  "top_factors": top_factors_for_unit(j, *naming) if naming else ""})
        counts_by_model[model] = counts
        print("  selective units per class: " + "  ".join(f"{n} {counts[n]}" for n in class_names))

        # 2. One probe per model, reused for every target.
        probe = fit_code_probe(Z, y, sae.n_dict, num_classes, device, args.probe_epochs)

        # 3. Ablate each target class's top units.
        for target in targets:
            t = class_names.index(target)
            curve = ablation_curve(probe, Z["test"], y["test"], ranked[t], args.ablate, num_classes, device)
            for m in sorted(curve):
                ablation_rows.setdefault(target, []).append(
                    {"model": model, "layer": layer, "sparsity": args.sparsity, "target": target,
                     "ablated_units": m, "uar": float(np.mean(curve[m])),
                     **{f"recall_{n}": float(r) for n, r in zip(class_names, curve[m])}})
            baseline, ablated = curve[0][t], min(curve[m][t] for m in args.ablate)
            # The largest drop in any other class shows where the ablation stops being
            # specific and starts hurting the model in general.
            collateral = max(curve[0][c] - min(curve[m][c] for m in args.ablate)
                             for c in range(num_classes) if c != t)
            summary.append({"model": model, "layer": layer, "target": target,
                            "selective_units": counts[target],
                            "recall_baseline": float(baseline), "recall_ablated": float(ablated),
                            "max_other_class_drop": float(collateral),
                            "threshold": args.threshold, "min_active": args.min_active,
                            "sparsity": args.sparsity, "max_ablated": max(args.ablate)})
            panels.setdefault(target, []).append((model, layer, class_names, curve))
            print(f"  {target:>9} recall {baseline:.3f} → {ablated:.3f} after up to {max(args.ablate)} units"
                  f" (worst other-class drop {collateral:.3f})")

    write_csv(out / "selective_units.csv", unit_rows)
    write_csv(out / "selective_units_summary.csv", summary)
    for target, rows in ablation_rows.items():
        write_csv(out / f"selective_units_ablation_{target}.csv", rows)
        plot_ablation(panels[target], target, out)
    if counts_by_model:
        plot_counts(counts_by_model, class_names, args, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-selective SAE units and their ablation.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--layer", default="best", help="'best', 'final' or a layer name.")
    parser.add_argument("--sparsity", type=int, default=90)
    parser.add_argument("--targets", nargs="+", default=["all"], help="Class names, or 'all'.")
    parser.add_argument("--ablate", nargs="+", type=int, default=[1, 5, 10, 20, 40, 80, 160])
    parser.add_argument("--top-units", type=int, default=20, help="Units per class listed in the CSV.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Selectivity bar for the counts.")
    parser.add_argument("--min-active", type=float, default=0.02, help="Minimum firing rate on the class.")
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
