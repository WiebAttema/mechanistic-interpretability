"""Extract the activations of every layer of one encoder on one dataset.

Every clip gets one mean-pooled vector per layer. A class-stratified subset of clips
(config: frames) also gets its per-frame activations, which are too large to keep
for everything. Output goes to <cache_dir>/activations/<dataset>/<model>/; see
activations.py for the file layout.

    python -m audiointerp.extract --dataset emodb --model wav2vec2-base
    python -m audiointerp.extract --dataset aibo --model whisper-small --limit 3   # smoke test
"""
import argparse
import sys

import numpy as np
import torch

from audiointerp.activations import activations_dir
from audiointerp.config import load_config
from audiointerp.data import DATASETS, SAMPLING_RATE, label_names, load_audio, load_index, split_indices
from audiointerp.encoders import load_encoder
from audiointerp.utils import pick_device


def frame_subset(clips, labels, dataset, frames_cfg):
    """Indices of up to frames_cfg["per_class"] random clips per class from one split."""
    if frames_cfg["split"] == "all":
        pool = list(range(len(clips)))
    else:
        pool = split_indices([clip.speaker for clip in clips], dataset)[frames_cfg["split"]]
    rng = np.random.RandomState(frames_cfg["seed"])
    chosen = set()
    for label in labels:
        candidates = [i for i in pool if clips[i].label == label]
        chosen.update(rng.permutation(candidates)[: frames_cfg["per_class"]].tolist())
    return chosen


def main(args):
    cfg = load_config(args.config)
    if args.model not in cfg["models"]:
        sys.exit(f"Unknown model {args.model!r}; options: {', '.join(cfg['models'])}")
    model_cfg = cfg["models"][args.model]
    out_dir = activations_dir(cfg, args.dataset, args.model)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    clips = load_index(args.dataset, cfg)
    labels = label_names(clips)  # from the full index, so --limit keeps every class
    clips = clips[: args.limit]
    frame_clips = frame_subset(clips, labels, args.dataset, cfg["frames"])
    frame_dtype = getattr(torch, cfg["frames"]["dtype"])
    max_duration = cfg["max_duration_sec"][args.dataset]

    device = pick_device()
    print(f"Model   : {args.model} ({model_cfg['hf_id']}) on {device}")
    print(f"Dataset : {args.dataset}, {len(clips)} clips, per-frame for {len(frame_clips)}")
    hf_cache = cfg["paths"]["hf_cache"]
    encoder = load_encoder(model_cfg, device, cfg["extraction"]["conv_layers"],
                           cache_dir=str(hf_cache) if hf_cache else None)

    pooled = {name: [] for name in encoder.layer_names}
    durations = []
    for i, clip in enumerate(clips):
        signal = load_audio(clip.path, max_duration)
        acts = encoder(signal)
        for name, frames in acts.items():
            pooled[name].append(frames.mean(dim=0))
        durations.append(len(signal) / SAMPLING_RATE)

        if i in frame_clips:
            torch.save({"file_path": clip.path, "label": labels.index(clip.label), "duration": durations[-1],
                        "activations": {name: frames.to(frame_dtype) for name, frames in acts.items()}},
                       out_dir / "frames" / f"{clip.id}.pt")
        if i == 0:
            print("Layers  : " + ", ".join(f"{name} {tuple(a.shape)}" for name, a in acts.items())
                  + f"  (first clip, {durations[0]:.2f} s)")
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(clips)}")

    path = out_dir / "pooled.pt"
    torch.save({
        "dataset": args.dataset,
        "model": args.model,
        "hf_id": model_cfg["hf_id"],
        "layer_names": encoder.layer_names,
        "pooled": {name: torch.stack(vectors) for name, vectors in pooled.items()},
        "labels": torch.tensor([labels.index(clip.label) for clip in clips]),
        "label_names": labels,
        "file_paths": [clip.path for clip in clips],
        "speakers": [clip.speaker for clip in clips],
        "durations": durations,
        "frame_clips": sorted(clips[i].id for i in frame_clips),
    }, path)
    print(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract layer-wise encoder activations.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, help="A key under `models` in the config.")
    parser.add_argument("--limit", type=int, help="Only the first N clips (smoke test).")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
