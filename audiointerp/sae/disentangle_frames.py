"""Disentanglement on single frames: for every eGeMAPS descriptor, a Lasso finds the
smallest set of frame SAE units (and, as reference, original channels) that predicts it,
in the eGeMAPS windows of the per-frame clips. Reports informativeness (held-out R²),
completeness (1 = one unit carries the descriptor, 0 = spread evenly; the measure of
Mariotte et al.) and the effective number of units the weight is spread over, which,
unlike completeness, is fair between 4096 units and 512 channels.

Needs frame SAEs (sae/train_frames.py) and the per-frame eGeMAPS. Output:
<output_dir>/sae/<dataset>/disentangle_frames_<model>_s<sparsity>.pt

    python -m audiointerp.sae.disentangle_frames --dataset aibo --model wav2vec2-base
"""
import argparse
import sys

import numpy as np
import torch
from sklearn.linear_model import LassoCV
from transformers import AutoConfig

from audiointerp.config import load_config
from audiointerp.data import DATASETS
from audiointerp.frame_correlation import conv_layout, load_aligned, per_descriptor, sae_transforms, zscore
from audiointerp.sae.model import sae_dir


def concentration(weights):
    """Completeness (1 = one column carries all the weight, 0 = spread evenly over all) and
    the effective number of columns: the weight is spread as if over that many equal ones."""
    w = np.abs(weights)
    if w.sum() < 1e-12:  # the Lasso switched everything off
        return 0.0, 0.0
    p = w[w > 0] / w.sum()
    entropy = -np.sum(p * np.log(p))
    return float(1 - entropy / np.log(len(w))), float(np.exp(entropy))


def lasso_scores(xs, ys, folds=5, n_penalties=6):
    """Per column of y, how a Lasso from all columns of x does, as a (3, y columns) tensor:
    held-out R², completeness, effective number. Standardised within clips like the rest of
    the frame analysis. Every folds-th clip is held out for R²; the penalty is chosen per
    column by cross-validation over the other clips, again holding out whole clips."""
    clip = np.concatenate([np.full(len(x), i) for i, x in enumerate(xs)])
    x = torch.cat([zscore(a) for a in xs]).double().numpy()
    y = torch.cat([zscore(b) for b in ys]).double().numpy()
    test = clip % folds == 0
    x_fit, y_fit, x_test, y_test = x[~test], y[~test], x[test], y[test]
    inner = [(np.flatnonzero(clip[~test] % folds != k), np.flatnonzero(clip[~test] % folds == k))
             for k in range(1, folds)]
    scores = []
    for j in range(y.shape[1]):
        top = np.abs(x_fit.T @ y_fit[:, j]).max() / len(x_fit)  # this penalty switches every weight off
        lasso = LassoCV(alphas=top * np.logspace(-2, 0, n_penalties), cv=inner, max_iter=10000)
        lasso.fit(x_fit, y_fit[:, j])
        scores.append([lasso.score(x_test, y_test[:, j]), *concentration(lasso.coef_)])
    return torch.tensor(scores).T.float()


def main(args):
    cfg = load_config(args.config)
    hf_cache = cfg["paths"]["hf_cache"]
    layout = conv_layout(AutoConfig.from_pretrained(cfg["models"][args.model]["hf_id"],
                                                    cache_dir=str(hf_cache) if hf_cache else None))
    transforms = sae_transforms(cfg, args.dataset, args.model, args.layers, args.sparsity)
    if not transforms:
        sys.exit(f"No frame SAEs for {args.model} at sparsity {args.sparsity}; run audiointerp.sae.train_frames first.")
    layout = {layer: layout[layer] for layer in transforms}

    names, channels, egemaps, voiced = load_aligned(cfg, args.dataset, args.model, layout)
    _, units, _, _ = load_aligned(cfg, args.dataset, args.model, layout, transforms)
    scores = {}  # {layer: {"channels" or "units": (3, descriptors): R², completeness, effective number}}
    for layer in layout:
        scores[layer] = {kind: per_descriptor(lasso_scores, acts[layer], egemaps, voiced, names)
                         for kind, acts in (("channels", channels), ("units", units))}
        print(f"\n{args.model} {layer}: R², completeness, effective number  (channels | SAE units)")
        for i, name in enumerate(names):
            c, u = scores[layer]["channels"][:, i], scores[layer]["units"][:, i]
            print(f"  {name[:26]:>26}  {c[0]:5.2f} {c[1]:5.2f} {c[2]:6.1f} | {u[0]:5.2f} {u[1]:5.2f} {u[2]:6.1f}")

    out = sae_dir(cfg, args.dataset) / f"disentangle_frames_{args.model}_s{args.sparsity}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "scores": scores, "rows": ["r2", "completeness", "effective"]}, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disentanglement of frame SAE units against per-frame eGeMAPS.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", nargs="+", default=["cnn"],
                        help="Layers with a frame SAE. conv_0 is very slow: its channels move almost in lockstep.")
    parser.add_argument("--sparsity", type=int, default=99)
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
