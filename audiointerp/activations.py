"""Reading the activation files written by extract.py.

<cache_dir>/activations/<dataset>/<model>/pooled.pt, for every clip:
    layer_names   layer names in depth order (see encoders.py)
    pooled        {layer name: (n_clips, dim) float32}, the mean over the clip's real frames
    labels        (n_clips,) class indices into label_names
    label_names, file_paths, speakers, durations (seconds, after truncation)
    frame_clips   ids of the clips that also have per-frame activations

<cache_dir>/activations/<dataset>/<model>/frames/<clip id>.pt, for those clips:
    {"file_path", "label", "duration", "activations": {layer name: (frames, dim)}}
"""
from pathlib import Path

import torch


def activations_dir(cfg, dataset, model):
    return Path(cfg["paths"]["cache_dir"]) / "activations" / dataset / model


def load_pooled(cfg, dataset, model):
    path = activations_dir(cfg, dataset, model) / "pooled.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run: python -m audiointerp.extract --dataset {dataset} --model {model}")
    return torch.load(path, weights_only=True)


def load_frames(cfg, dataset, model, clip_id):
    return torch.load(activations_dir(cfg, dataset, model) / "frames" / f"{clip_id}.pt", weights_only=True)


def layer_matrix(data, layer):
    """(n_clips, dim) pooled activations of one layer, by name."""
    if layer not in data["pooled"]:
        raise KeyError(f"No layer {layer!r}; available: {data['layer_names']}")
    return data["pooled"][layer].float()


def layer_sort_key(name):
    """Depth order of layer names: conv_0 < conv_1 < ... < cnn < layer_0 < layer_1 < ..."""
    kind, _, index = name.partition("_")
    return {"conv": 0, "cnn": 1, "layer": 2}[kind], int(index or 0)
