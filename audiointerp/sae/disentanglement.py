"""Disentanglement of SAE codes against eGeMAPS voice factors (Mariotte et al. Fig. 2-4).

One Lasso fit per factor on a subsample of the test split, reporting informativeness
(held-out R²) and completeness (DCI compactness of the Lasso weights), plus the same fit
on the original standardized layer as reference (sparsity "ref"). The Lasso weights are
saved so selective_units.py and factor_units.py can name units by the factors that load
on them.

Use --cv for reported numbers. With one fixed penalty for every factor, the 4096-dim
codes and the original layer are constrained about equally hard and the completeness
difference between them mostly disappears.

    python -m audiointerp.sae.disentanglement --dataset aibo --models wav2vec2-base --cv
    python -m audiointerp.sae.disentanglement --dataset aibo --models wav2vec2-base --sparsities 90
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neighbors import NearestNeighbors

from audiointerp.activations import layer_matrix, load_pooled
from audiointerp.config import load_config
from audiointerp.data import DATASETS, split_indices
from audiointerp.egemaps import FAMILIES, factor_family, load_egemaps
from audiointerp.plotting import model_color, save, style_axes
from audiointerp.sae.model import encode, lasso_coefs_path, load_checkpoint, sae_dir
from audiointerp.utils import CSVLog, pick_device

FIELDS = ["model", "layer", "sparsity", "factor", "family", "r2", "completeness", "entropy"]


def knn_entropy(x, k=3):
    """Kozachenko-Leonenko entropy estimate of a 1-D variable."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    r = np.clip(dist[:, k], 1e-12, None)
    return digamma(len(x)) - digamma(k) + np.log(2.0) + np.mean(np.log(r))


def fit_factor(f, X, holdout, alpha, use_cv, seed):
    """Lasso from X to one factor f. Returns held-out R², completeness and the weights."""
    order = np.random.RandomState(seed).permutation(len(f))
    n_fit = int(len(f) * (1 - holdout))
    fit_idx, eval_idx = order[:n_fit], order[n_fit:]

    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0)
    sd[sd < 1e-8] = 1.0
    X = (X - mu) / sd
    f = (f - f[fit_idx].mean()) / max(f[fit_idx].std(), 1e-8)

    if use_cv:
        # Each factor has its own natural scale, so the penalty grid is calibrated to
        # this fit instead of one fixed grid for every factor.
        alpha_max = max(np.max(np.abs(X[fit_idx].T @ f[fit_idx])) / n_fit, 1e-8)
        reg = LassoCV(alphas=np.logspace(np.log10(alpha_max * 1e-2), np.log10(alpha_max), 6),
                      cv=3, max_iter=100000, n_jobs=-1)
    else:
        reg = Lasso(alpha=alpha, max_iter=20000)
    reg.fit(X[fit_idx], f[fit_idx])
    r2 = reg.score(X[eval_idx], f[eval_idx])

    # DCI completeness: 1 = a single dimension carries the factor, 0 = spread evenly.
    w = np.abs(reg.coef_)
    if w.sum() < 1e-12:
        return r2, 0.0, reg.coef_
    rho = w / w.sum()
    nz = rho[rho > 0]
    return r2, float(1.0 + np.sum(nz * np.log(nz)) / np.log(len(w))), reg.coef_


def fit_all_factors(X, F, factor_names, entropies, tag, args):
    """One row per factor, and the (n_factors, dims) Lasso weight matrix."""
    rows, coefs = [], np.zeros((len(factor_names), X.shape[1]), dtype=np.float32)
    for i, name in enumerate(factor_names):
        r2, completeness, coefs[i] = fit_factor(F[:, i], X, args.holdout, args.alpha, args.cv, args.seed)
        rows.append({**tag, "factor": name, "family": factor_family(name), "r2": r2,
                     "completeness": completeness, "entropy": entropies[i]})
    return rows, coefs


def make_figures(rows, args, out):
    keys = list(dict.fromkeys((row["model"], row["layer"]) for row in rows))

    def cell(model, layer, sparsity=None):
        sub = [r for r in rows if (r["model"], r["layer"]) == (model, layer)]
        return [r for r in sub if r["sparsity"] == sparsity] if sparsity is not None else sub

    def top10(sub, metric):
        values = sorted((r[metric] for r in sub), reverse=True)[:10]
        return np.mean(values), np.std(values)

    # Fig. 2: top-10 informativeness and completeness vs sparsity.
    for metric, name in [("r2", "informativeness"), ("completeness", "completeness")]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for model, layer in keys:
            sparsities = sorted({r["sparsity"] for r in cell(model, layer) if r["sparsity"] != "ref"})
            stats = [top10(cell(model, layer, s), metric) for s in sparsities]
            ax.errorbar(sparsities, [m for m, _ in stats], yerr=[s for _, s in stats],
                        color=model_color(model), linewidth=2, capsize=2, label=f"{model} {layer}")
            if cell(model, layer, "ref"):
                ax.axhline(top10(cell(model, layer, "ref"), metric)[0], color=model_color(model),
                           linewidth=1, linestyle="--", alpha=0.45)
        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel(f"Top 10 {'R²' if metric == 'r2' else 'completeness'}")
        ax.set_title(f"{name.capitalize()} (dashed = original layer)", fontsize=11.5)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
        save(fig, out / f"disentangle_{name}.png")

    # Fig. 3: completeness vs factor entropy at one sparsity.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model, layer in keys:
        sub = cell(model, layer, args.scatter_sparsity)
        if sub:
            ax.scatter([r["entropy"] for r in sub], [r["completeness"] for r in sub], s=14, alpha=0.65,
                       color=model_color(model), label=f"{model} {layer}")
    ax.set_xlabel("Factor entropy estimate")
    ax.set_ylabel("Completeness")
    ax.set_title(f"Completeness vs factor entropy ({args.scatter_sparsity}% sparsity)", fontsize=11.5)
    style_axes(ax, grid="both")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "disentangle_entropy.png")

    # Fig. 4: families of the 10 best-predicted factors, averaged over sparsities.
    ncols = min(2, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, (model, layer) in zip(axes.flat, keys):
        sparsities = sorted({r["sparsity"] for r in cell(model, layer) if r["sparsity"] != "ref"})
        counts = np.zeros((len(sparsities), len(FAMILIES)))
        for si, s in enumerate(sparsities):
            for r in sorted(cell(model, layer, s), key=lambda r: r["r2"], reverse=True)[:10]:
                counts[si, FAMILIES.index(r["family"])] += 1
        ax.bar(range(len(FAMILIES)), counts.mean(axis=0), color=model_color(model))
        ax.set_xticks(range(len(FAMILIES)), FAMILIES, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{model} {layer}", fontsize=10)
        ax.set_ylabel("Avg count in top 10")
        style_axes(ax)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    save(fig, out / "disentangle_families.png")


def main(args):
    cfg = load_config(args.config)
    device = pick_device()
    out = sae_dir(cfg, args.dataset)
    clip_ids, factor_names, F_all = load_egemaps(cfg, args.dataset)
    row_of = {clip: i for i, clip in enumerate(clip_ids)}
    print(f"eGeMAPS: {len(clip_ids)} clips x {len(factor_names)} factors")

    # Checkpoints grouped per (model, layer), so each layer is loaded once.
    groups = {}
    for path in sorted((out / "checkpoints").glob("*.pt")):
        _, meta = load_checkpoint(path)
        if meta["model"] in args.models and (not args.sparsities or meta["sparsity"] in args.sparsities):
            groups.setdefault((meta["model"], meta["layer"]), {})[meta["sparsity"]] = path
    if not groups:
        sys.exit(f"No matching checkpoints in {out / 'checkpoints'}; run audiointerp.sae.train first.")

    log = CSVLog(out / "disentanglement.csv", FIELDS, resume=args.resume)
    done = {(row["model"], row["layer"], row["sparsity"]) for row in log.rows}
    for (model, layer), paths in groups.items():
        if all((model, layer, s) in done for s in ["ref", *paths]):
            print(f"[skip] {model} {layer}: already done")
            continue
        data = load_pooled(cfg, args.dataset, model)
        test = split_indices(data["speakers"], args.dataset)["test"]
        # The same seeded subsample for every (model, layer), so cells stay comparable.
        test = [test[i] for i in np.random.RandomState(args.seed).permutation(len(test))[: args.max_samples]]
        missing = [i for i in test if Path(data["file_paths"][i]).stem not in row_of]
        if missing:
            sys.exit(f"{len(missing)} clips have no eGeMAPS row; re-run audiointerp.egemaps.")
        F = F_all[[row_of[Path(data["file_paths"][i]).stem] for i in test]]
        entropies = [knn_entropy(F[:, j]) for j in range(F.shape[1])]
        X = layer_matrix(data, layer)[test]

        for path in paths.values():
            sae, meta = load_checkpoint(path, device)
            if (model, layer, "ref") not in done:
                print(f"\n=== {model} {layer}: reference (original layer)")
                rows, _ = fit_all_factors(((X - meta["mu"]) / meta["sd"]).numpy(), F, factor_names,
                                          entropies, {"model": model, "layer": layer, "sparsity": "ref"}, args)
                for row in rows:
                    log.write(row)
                done.add((model, layer, "ref"))
            if (model, layer, meta["sparsity"]) in done:
                continue
            print(f"=== {model} {layer}: SAE at {meta['sparsity']}% sparsity")
            Z = encode(sae, X, meta["mu"], meta["sd"], device).numpy()
            rows, coefs = fit_all_factors(Z, F, factor_names, entropies,
                                          {"model": model, "layer": layer, "sparsity": meta["sparsity"]}, args)
            for row in rows:
                log.write(row)
            coefs_path = lasso_coefs_path(cfg, args.dataset, model, layer, meta["sparsity"])
            coefs_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(coefs_path, coefs=coefs, factor_names=np.array(factor_names))
    log.close()
    make_figures(log.rows, args, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eGeMAPS disentanglement of SAE codes.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sparsities", nargs="+", type=int, help="Default: every trained sparsity.")
    parser.add_argument("--max-samples", type=int, default=4000, help="Test clips used for the Lasso fits.")
    parser.add_argument("--holdout", type=float, default=0.3, help="Fraction of those clips held out for R².")
    parser.add_argument("--alpha", type=float, default=0.05, help="Lasso penalty without --cv.")
    parser.add_argument("--cv", action="store_true", help="LassoCV per factor instead of a fixed alpha (slower).")
    parser.add_argument("--scatter-sparsity", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
