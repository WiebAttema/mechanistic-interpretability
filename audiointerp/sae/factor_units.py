"""Factor-driven SAE ablation. Where selective_units.py ranks dictionary units by the
emotion class they fire on, this ranks them by which eGeMAPS factors load on them (the
Lasso weights from disentanglement.py), zeroes those units at test time and reports
the effect on every emotion class.

Each ablation is compared with zeroing the same number of random active units, so a
drop only counts if it beats that control.

    python -m audiointerp.sae.factor_units --dataset aibo --models wav2vec2-base
    python -m audiointerp.sae.factor_units --dataset aibo --models wav2vec2-base --families loudness quality
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from audiointerp.activations import load_pooled
from audiointerp.config import load_config
from audiointerp.data import DATASETS
from audiointerp.egemaps import FAMILY_RULES, factor_family
from audiointerp.plotting import save, style_axes
from audiointerp.sae.model import lasso_coefs_path, load_codes, resolve_layer, sae_dir
from audiointerp.sae.selective_units import (
    ablation_curve, class_selectivity, fit_code_probe, load_unit_factors,
)
from audiointerp.utils import pick_device, write_csv


def resolve_groups(factor_names, families, factors):
    """[(group name, factor indices)]: whole families, then single factors."""
    groups = []
    for family in families:
        idx = [i for i, name in enumerate(factor_names) if factor_family(name) == family]
        if idx:
            groups.append((family, idx))
        else:
            print(f"[skip] no eGeMAPS factor in family {family!r}")
    for name in factors:
        if name in factor_names:
            groups.append((name, [factor_names.index(name)]))
        else:
            print(f"[skip] unknown factor {name!r}")
    return groups


def factor_scores(coefs, idx):
    """Per-unit score for a group of factors. Each factor's absolute Lasso weights are
    normalised to sum to 1 first, so a factor with large raw weights does not dominate."""
    w = np.abs(coefs[idx])
    total = w.sum(axis=1, keepdims=True)
    total[total < 1e-12] = 1.0
    return (w / total).mean(axis=0)


def random_control(probe, Z_test, y_test, pool, steps, num_classes, device, n_draws, seed):
    """Mean per-class recall after zeroing m units drawn at random from `pool`."""
    rng = np.random.RandomState(seed)
    curves = [ablation_curve(probe, Z_test, y_test, list(rng.permutation(pool)), steps, num_classes, device)
              for _ in range(n_draws)]
    return {m: np.mean([curve[m] for curve in curves], axis=0) for m in curves[0]}


def plot_group(panels, group, class_names, out):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), squeeze=False, sharey=True)
    for ax, (model, layer, curve, control) in zip(axes[0], panels):
        ms = sorted(curve)
        for c, name in enumerate(class_names):
            ax.plot(ms, [100 * curve[m][c] for m in ms], "-o", markersize=3, linewidth=1.8, label=name)
        ax.plot(ms, [100 * np.mean(control[m]) for m in ms], "--", linewidth=1.4, color="#898781",
                label="UAR, random units")
        ax.set_title(f"{model} {layer}", fontsize=10)
        ax.set_xlabel(f"Top {group} units ablated")
        style_axes(ax)
    axes[0][0].set_ylabel("Recall (%)")
    axes[0][-1].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Ablating units that carry {group}\n(dashed: random-unit control)", fontsize=11.5)
    save(fig, out / f"factor_units_ablation_{group}.png")


def main(args):
    cfg = load_config(args.config)
    device = pick_device()
    out = sae_dir(cfg, args.dataset)
    top_rows, ablation_rows, overlap_rows, panels = [], [], [], {}

    for model in args.models:
        data = load_pooled(cfg, args.dataset, model)
        layer = resolve_layer(args.layer, cfg, args.dataset, model, data["layer_names"])
        naming = load_unit_factors(lasso_coefs_path(cfg, args.dataset, model, layer, args.sparsity))
        if naming is None:
            print(f"[skip] {model} {layer}: no Lasso weights; run audiointerp.sae.disentanglement first.")
            continue
        coefs, factor_names = naming
        groups = resolve_groups(factor_names, args.families, args.factors)
        sae, Z, y = load_codes(data, cfg, args.dataset, model, layer, args.sparsity, device)
        class_names = data["label_names"]
        num_classes = len(class_names)
        print(f"\n=== {model} {layer}, {args.sparsity}% sparsity, dict {sae.n_dict}")

        selectivity, active = class_selectivity(Z["train"], y["train"], num_classes)
        alive = (Z["train"] > 0).float().mean(dim=0).numpy() > args.min_active
        probe = fit_code_probe(Z, y, sae.n_dict, num_classes, device, args.probe_epochs)
        control = random_control(probe, Z["test"], y["test"], np.flatnonzero(alive), args.ablate,
                                 num_classes, device, args.n_random, args.seed)

        for group, idx in groups:
            score = factor_scores(coefs, idx)
            ranked = [j for j in np.argsort(score)[::-1] if alive[j] and score[j] > 0]
            if not ranked:
                print(f"  [skip] {group}: no active unit carries this group")
                continue
            curve = ablation_curve(probe, Z["test"], y["test"], ranked, args.ablate, num_classes, device)
            for mode, recalls in (("factor", curve), ("random", control)):
                for m in sorted(recalls):
                    ablation_rows.append(
                        {"model": model, "layer": layer, "sparsity": args.sparsity, "group": group,
                         "ablated_units": m, "mode": mode, "uar": float(np.mean(recalls[m])),
                         **{f"recall_{n}": float(r) for n, r in zip(class_names, recalls[m])}})

            # Which class suffers most, and does that beat the random control?
            drops = curve[0] - curve[max(args.ablate)]
            control_drops = control[0] - control[max(args.ablate)]
            hit = int(np.argmax(drops - control_drops))

            for rank, j in enumerate(ranked[: args.top_units]):
                owners = [class_names[c] for c in range(num_classes)
                          if selectivity[c][j] > args.threshold and active[c][j] > args.min_active]
                top_rows.append({"model": model, "layer": layer, "sparsity": args.sparsity, "group": group,
                                 "rank": rank, "unit": int(j), "score": float(score[j]),
                                 "selective_for": ";".join(owners)})

            top_set = set(ranked[: args.top_units])
            overlap = {"model": model, "layer": layer, "group": group, "n_units_ranked": len(ranked),
                       "most_affected_class": class_names[hit], "drop": float(drops[hit]),
                       "drop_random_control": float(control_drops[hit]),
                       "threshold": args.threshold, "min_active": args.min_active,
                       "sparsity": args.sparsity, "top_units": args.top_units, "max_ablated": max(args.ablate)}
            for c, name in enumerate(class_names):
                selective = set(np.flatnonzero((selectivity[c] > args.threshold) & (active[c] > args.min_active)))
                overlap[f"overlap_{name}"] = len(top_set & selective)
            overlap_rows.append(overlap)
            panels.setdefault(group, []).append((model, layer, curve, control))
            print(f"  {group:>10}: {len(ranked)} units carry it | largest excess drop {class_names[hit]} "
                  f"{drops[hit]:.3f} vs {control_drops[hit]:.3f} random")

    write_csv(out / "factor_units.csv", top_rows)
    write_csv(out / "factor_units_ablation.csv", ablation_rows)
    write_csv(out / "factor_units_overlap.csv", overlap_rows)
    for group, group_panels in panels.items():
        plot_group(group_panels, group, class_names, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factor-driven SAE unit ablation.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--layer", default="best", help="'best', 'final' or a layer name.")
    parser.add_argument("--sparsity", type=int, default=90)
    parser.add_argument("--families", nargs="+", default=[family for family, _ in FAMILY_RULES],
                        help="eGeMAPS families to ablate as groups.")
    parser.add_argument("--factors", nargs="+", default=[], help="Single eGeMAPS factors, each its own group.")
    parser.add_argument("--ablate", nargs="+", type=int, default=[1, 5, 10, 20, 40, 80, 160])
    parser.add_argument("--top-units", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=5, help="Random-control draws to average over.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Class-selectivity bar for the overlap columns.")
    parser.add_argument("--min-active", type=float, default=0.02)
    parser.add_argument("--probe-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
