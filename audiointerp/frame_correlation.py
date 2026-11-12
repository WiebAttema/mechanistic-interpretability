"""Correlate CNN channels with eGeMAPS, frame by frame.

For every clip with per-frame activations, each eGeMAPS frame (a 20 ms window every
10 ms) gets the mean of the conv frames whose centre falls inside its window (version A),
and the same smoothed like openSMILE's sma3 (version B). Then every channel is correlated
with every descriptor, within clips. Descriptors ending in "nz" (pitch, voice quality,
formants) only exist on voiced frames, so they are correlated on those frames only.

Needs per-frame activations with conv layers (extraction.conv_layers: true) and
per-frame eGeMAPS (egemaps.py --frames). Output: <output_dir>/frame_correlation/.

    python -m audiointerp.frame_correlation --dataset aibo --model wav2vec2-base
"""
import argparse

import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from transformers import Wav2Vec2Model

from audiointerp.activations import load_frames
from audiointerp.config import load_config
from audiointerp.data import DATASETS, SAMPLING_RATE
from audiointerp.egemaps import egemaps_frames_dir
from audiointerp.plotting import model_color, save, style_axes

F0 = "F0semitoneFrom27.5Hz_sma3nz"
ALPHA_RATIO = "alphaRatio_sma3"


def frame_layout(kernels, strides):
    """Step and receptive field ("seen"), in samples, of every conv layer's frames."""
    step, seen, layout = 1, 1, []
    for kernel, stride in zip(kernels, strides):
        seen += (kernel - 1) * step  # needs the previous layer's step, so before updating it
        step *= stride
        layout.append((step, seen))
    return layout


def frame_centres(n_frames, step, seen):
    """Time in seconds of the middle of every frame."""
    return (step * torch.arange(n_frames, dtype=torch.float64) + seen / 2) / SAMPLING_RATE


def align(acts, centres, start, end):
    """Mean of the frames whose centre falls in each window, (windows, channels), and
    which windows contain any frame at all."""
    inside = (centres[None, :] >= start[:, None]) & (centres[None, :] < end[:, None])
    counts = inside.sum(dim=1, keepdim=True)
    return (inside.float() @ acts) / counts.clamp(min=1), counts[:, 0] > 0


def smooth3(x):
    """3-frame moving average over time, like openSMILE's sma3; the edge frames repeat."""
    padded = torch.cat([x[:1], x, x[-1:]])
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3


def zscore(x):
    """Every column to mean 0 and standard deviation 1."""
    return (x - x.mean(dim=0)) / x.std(dim=0, correction=0).clamp(min=1e-8)


def within_clip_correlation(xs, ys):
    """Correlation of every column of x with every column of y, (x columns, y columns).
    Each clip is standardised on its own before stacking, so only the ups and downs
    within clips count, not differences between clips."""
    x = torch.cat([zscore(x) for x in xs])
    y = torch.cat([zscore(y) for y in ys])
    return x.T @ y / len(x)


def filter_spectrum(filters, n_fft=512):
    """Frequencies in Hz, and how strongly each conv_0 filter, (channels, 1, kernel), passes
    them. The weights are zero-padded to n_fft first: 10 weights alone give only 6 frequencies."""
    return torch.fft.rfftfreq(n_fft, d=1 / SAMPLING_RATE), torch.fft.rfft(filters[:, 0], n=n_fft).abs()


def peak_frequencies(filters):
    """Frequency in Hz that each conv_0 filter passes most strongly."""
    hz, response = filter_spectrum(filters)
    return hz[response.argmax(dim=1)]


def band_ratio(filters):
    """Each conv_0 filter's own alpha ratio: the energy it passes in 1-5 kHz over the
    energy it passes in 50 Hz-1 kHz, in dB."""
    hz, response = filter_spectrum(filters)
    energy = response ** 2
    high = energy[:, (hz >= 1000) & (hz < 5000)].sum(dim=1)
    low = energy[:, (hz >= 50) & (hz < 1000)].sum(dim=1)
    return 10 * torch.log10(high / low)


def plot_alpha_ratio(r, names, filters, model, path):
    """One dot per conv_0 channel: how it follows alpha ratio (high band 1-5 kHz vs low band
    50 Hz-1 kHz) against what its filter listens to, as peak frequency and as band ratio."""
    y = r[:, names.index(ALPHA_RATIO)]
    xs = {"Peak frequency of the filter (kHz)": peak_frequencies(filters) / 1000,
          "Band ratio of the filter, 1-5 kHz vs 50 Hz-1 kHz (dB)": band_ratio(filters)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (label, x) in zip(axes, xs.items()):
        ax.scatter(x, y, s=10, color=model_color(model))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_title(f"rank correlation {spearmanr(x, y).statistic:.2f}")
        style_axes(ax, grid="both")
    axes[0].set_ylabel("r with alpha ratio")
    axes[0].set_ylim(-1, 1)
    fig.suptitle(f"{model}: conv_0 channels vs alpha ratio")
    save(fig, path)


def main(args):
    cfg = load_config(args.config)
    hf_cache = cfg["paths"]["hf_cache"]
    model = Wav2Vec2Model.from_pretrained(cfg["models"][args.model]["hf_id"],
                                          cache_dir=str(hf_cache) if hf_cache else None)
    conv = model.config
    layers = [f"conv_{i}" for i in range(len(conv.conv_kernel) - 1)] + ["cnn"]
    layout = dict(zip(layers, frame_layout(conv.conv_kernel, conv.conv_stride)))
    # what each conv_0 channel listens to is in its weights
    filters = model.feature_extractor.conv_layers[0].conv.weight.detach()

    aligned = {version: {layer: [] for layer in layers} for version in ("A", "B")}
    egemaps, voiced = [], []
    for path in sorted(egemaps_frames_dir(cfg, args.dataset).glob("*.pt")):
        eg = torch.load(path, weights_only=True)
        acts = load_frames(cfg, args.dataset, args.model, path.stem)["activations"]
        # openSMILE's last frame takes whatever audio is left, so it is longer than 20 ms
        start, end, values = eg["start"][:-1], eg["end"][:-1], eg["values"][:-1]
        egemaps.append(values)
        voiced.append(values[:, eg["names"].index(F0)] > 0)
        for layer in layers:
            a = acts[layer].float()
            means, has_frames = align(a, frame_centres(len(a), *layout[layer]), start, end)
            assert has_frames.all(), f"{path.stem}, {layer}: an eGeMAPS window without conv frames"
            aligned["A"][layer].append(means)
            aligned["B"][layer].append(smooth3(means))
    names = eg["names"]
    print(f"{args.model} on {args.dataset}: {len(egemaps)} clips, {sum(map(len, egemaps))} frames, "
          f"{int(sum(v.sum() for v in voiced))} voiced")

    nz = torch.tensor([name.endswith("nz") for name in names])
    keep = [v.sum() >= 3 for v in voiced]  # clips with almost no voiced frames say nothing about pitch
    r = {version: {} for version in aligned}
    for version, by_layer in aligned.items():
        for layer, acts in by_layer.items():
            matrix = torch.empty(acts[0].shape[1], len(names))
            matrix[:, ~nz] = within_clip_correlation(acts, [e[:, ~nz] for e in egemaps])
            matrix[:, nz] = within_clip_correlation(
                [a[v] for a, v, k in zip(acts, voiced, keep) if k],
                [e[v][:, nz] for e, v, k in zip(egemaps, voiced, keep) if k])
            r[version][layer] = matrix

    out = cfg["paths"]["output_dir"] / "frame_correlation" / f"{args.dataset}_{args.model}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "layers": layers, "n_clips": len(egemaps), "r": r,
                "conv0_filters": filters}, out)
    print(f"Saved → {out}")
    plot_alpha_ratio(r["A"]["conv_0"], names, filters, args.model, out.with_name(f"{out.stem}_alpha_ratio.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlate CNN channels with per-frame eGeMAPS.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
