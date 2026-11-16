"""Train TopK SAEs on single frames of the CNN instead of pooled clips: the random frames
of every clip in frame_sample.pt (extract.py with frame_sample in the config). Frames are
split by the speaker of their clip, like train.py, and every SAE reports its test
reconstruction error and dead units.

    python -m audiointerp.sae.train_frames --dataset aibo --model wav2vec2-base --layers cnn conv_0
"""
import argparse
from pathlib import Path

import torch

from audiointerp.activations import load_frame_sample, load_pooled
from audiointerp.config import load_config
from audiointerp.data import DATASETS, split_indices
from audiointerp.probes import zscore_stats
from audiointerp.sae.model import checkpoint_path, reconstruction_mse, save_checkpoint, sparsity_to_k, train_sae
from audiointerp.utils import pick_device


def split_frames(clip, speakers, dataset):
    """{"train", "val", "test"} -> which sampled frames belong to it, by the speaker of their clip."""
    return {name: torch.isin(clip, torch.tensor(idx)) for name, idx in split_indices(speakers, dataset).items()}


def main(args):
    cfg = load_config(args.config)
    device = pick_device()
    sample = load_frame_sample(cfg, args.dataset, args.model)
    pooled = load_pooled(cfg, args.dataset, args.model)
    k = sparsity_to_k(args.sparsity, args.dict_size)
    for layer in args.layers:
        X = sample[layer]["frames"].float()
        split = split_frames(sample[layer]["clip"], pooled["speakers"], args.dataset)
        mu, sd = zscore_stats(X[split["train"]])
        if args.max_z is not None:  # drop frames far outside the normal range, then standardise without them
            normal = ((X - mu) / sd).abs().amax(dim=1) <= args.max_z
            clips = sample[layer]["clip"][~normal].unique().tolist()
            print(f"{args.model} {layer}: dropped {int((~normal).sum())} frames beyond {args.max_z} sd, from "
                  f"{len(clips)} clips: {', '.join(Path(pooled['file_paths'][c]).stem for c in clips[:20])}")
            split = {name: mask & normal for name, mask in split.items()}
            mu, sd = zscore_stats(X[split["train"]])
        sae, info = train_sae((X[split["train"]] - mu) / sd, (X[split["val"]] - mu) / sd, args.dict_size, k,
                              epochs=args.epochs, batch_size=args.batch_size, device=device, seed=args.seed,
                              verbose=True)
        test_mse = reconstruction_mse(sae, X[split["test"]], mu, sd, device)
        save_checkpoint(checkpoint_path(cfg, args.dataset, args.model, f"{layer}_frames", args.sparsity), sae, mu, sd,
                        {"dataset": args.dataset, "model": args.model, "layer": layer, "sparsity": args.sparsity,
                         "val_mse": info["val_mse"], "test_mse": test_mse, "seed": args.seed, "max_z": args.max_z})
        print(f"{args.model} {layer}: {int(split['train'].sum())} train frames, k={k} of {args.dict_size} | "
              f"test mse {test_mse:.3f}, dead {info['dead_frac']:.2f}, best epoch {info['best_epoch']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TopK SAEs on single CNN frames.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", nargs="+", default=["cnn"])
    parser.add_argument("--sparsity", type=int, default=99, help="Percent of units off per frame: 99 -> k=40 of 4096.")
    parser.add_argument("--dict-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256, help="train.py uses 32 for pooled clips; frames are far more.")
    parser.add_argument("--max-z", type=float,
                        help="Drop frames with a standardised value beyond this, e.g. 20 for cnn (recording glitches "
                             "such as frozen audio). Off by default: normal conv_0 frames can be that extreme.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
