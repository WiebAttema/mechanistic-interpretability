"""Correlate CNN channels with eGeMAPS, frame by frame.

For every clip with per-frame activations, each eGeMAPS frame (a 20 ms window every
10 ms) gets the mean of the conv frames whose centre falls inside its window (version A),
and the same smoothed like openSMILE's sma3 (version B). Then every channel is correlated
with every descriptor, within clips. Descriptors ending in "nz" (pitch, voice quality,
formants) only exist on voiced frames, so they are correlated on those frames only.

Needs per-frame activations with conv layers (extraction.conv_layers: true) and
per-frame eGeMAPS (egemaps.py --frames). Output: <output_dir>/frame_correlation/.

Correlation ignores how big a channel's output is, so the learned norm gain of every channel
is saved too: a channel with a gain near 0 is switched off (wav2vec2-large-emotion has
hundreds in conv_0) and should be left out when reading channel results.

With --sae, each frame is first described by the units of a frame SAE
(sae/train_frames.py) instead of its channels, and the units are analysed the same way.

    python -m audiointerp.frame_correlation --dataset aibo --model wav2vec2-base
    python -m audiointerp.frame_correlation --dataset aibo --model wav2vec2-base --sae 99
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
from audiointerp.plotting import FALLBACK_COLOR, model_color, save, style_axes
from audiointerp.sae.model import checkpoint_path, encode, load_checkpoint

F0 = "F0semitoneFrom27.5Hz_sma3nz"
ALPHA_RATIO = "alphaRatio_sma3"
ACTIVE = 0.1  # a channel whose norm gain is below this hardly reaches the next layer


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


def heldout_r2(xs, ys, folds=5, penalty=0.01):
    """How well a weighted sum of all columns of x predicts each column of y, (y columns,).
    Ridge regression, standardised within clips like within_clip_correlation; the clips
    are split into folds and every fold is predicted by weights fitted on the others."""
    fold = torch.cat([torch.full((len(x),), i % folds) for i, x in enumerate(xs)])
    x = torch.cat([zscore(x) for x in xs]).double()
    y = torch.cat([zscore(y) for y in ys]).double()
    predicted = torch.empty_like(y)
    for k in range(folds):
        train, test = fold != k, fold == k
        ridge = penalty * train.sum() * torch.eye(x.shape[1], dtype=x.dtype)
        weights = torch.linalg.solve(x[train].T @ x[train] + ridge, x[train].T @ y[train])
        predicted[test] = x[test] @ weights
    return (1 - ((y - predicted) ** 2).sum(dim=0) / (y ** 2).sum(dim=0)).float()


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


def plot_alpha_ratio(r, names, filters, active, model, path):
    """One dot per conv_0 channel: how it follows alpha ratio (high band 1-5 kHz vs low band
    50 Hz-1 kHz) against what its filter listens to, as peak frequency and as band ratio.
    Channels with a norm gain below ACTIVE are drawn faint and left out of the rank correlation."""
    y = r[:, names.index(ALPHA_RATIO)]
    xs = {"Peak frequency of the filter (kHz)": peak_frequencies(filters) / 1000,
          "Band ratio of the filter, 1-5 kHz vs 50 Hz-1 kHz (dB)": band_ratio(filters)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (label, x) in zip(axes, xs.items()):
        ax.scatter(x[~active], y[~active], s=10, color=FALLBACK_COLOR, alpha=0.3,
                   label=f"norm gain below {ACTIVE} ({int((~active).sum())})")
        ax.scatter(x[active], y[active], s=10, color=model_color(model), label=f"active ({int(active.sum())})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_title(f"rank correlation {spearmanr(x[active], y[active]).statistic:.2f} (active channels)")
        style_axes(ax, grid="both")
    if (~active).any():
        axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylabel("r with alpha ratio")
    axes[0].set_ylim(-1, 1)
    fig.suptitle(f"{model}: conv_0 channels vs alpha ratio")
    save(fig, path)


def conv_layout(config):
    """{conv layer name: (step, seen)} of a wav2vec2 model config."""
    layers = [f"conv_{i}" for i in range(len(config.conv_kernel) - 1)] + ["cnn"]
    return dict(zip(layers, frame_layout(config.conv_kernel, config.conv_stride)))


def sae_transforms(cfg, dataset, model, layers, sparsity):
    """{layer: frames -> SAE units} for the layers that have a frame SAE at this sparsity."""
    transforms = {}
    for layer in layers:
        path = checkpoint_path(cfg, dataset, model, f"{layer}_frames", sparsity)
        if path.exists():
            sae, ckpt = load_checkpoint(path)
            transforms[layer] = lambda a, sae=sae, ckpt=ckpt: encode(sae, a, ckpt["mu"], ckpt["sd"], "cpu")
    return transforms


def load_aligned(cfg, dataset, model, layout, transforms=None):
    """For every clip with per-frame eGeMAPS: per layer, the mean of the conv frames in each
    eGeMAPS window (version A); the descriptors; and which frames are voiced. transforms
    ({layer: function}) are applied to a clip's frames first, e.g. encoding with an SAE."""
    aligned, egemaps, voiced = {layer: [] for layer in layout}, [], []
    for path in sorted(egemaps_frames_dir(cfg, dataset).glob("*.pt")):
        eg = torch.load(path, weights_only=True)
        acts = load_frames(cfg, dataset, model, path.stem)["activations"]
        # openSMILE's last frame takes whatever audio is left, so it is longer than 20 ms
        start, end, values = eg["start"][:-1], eg["end"][:-1], eg["values"][:-1]
        egemaps.append(values)
        voiced.append(values[:, eg["names"].index(F0)] > 0)
        for layer, (step, seen) in layout.items():
            a = acts[layer].float()
            if transforms and layer in transforms:
                a = transforms[layer](a)
            means, has_frames = align(a, frame_centres(len(a), step, seen), start, end)
            assert has_frames.all(), f"{path.stem}, {layer}: an eGeMAPS window without conv frames"
            aligned[layer].append(means)
    return eg["names"], aligned, egemaps, voiced


def per_descriptor(fn, acts, egemaps, voiced, names):
    """fn(acts, descriptors) on all frames for the plain descriptors and on voiced frames for
    the nz ones (pitch, voice quality, formants), put back together in the order of names (last axis)."""
    nz = torch.tensor([name.endswith("nz") for name in names])
    keep = [v.sum() >= 3 for v in voiced]  # clips with almost no voiced frames say nothing about pitch
    plain = fn(acts, [e[:, ~nz] for e in egemaps])
    voiced_only = fn([a[v] for a, v, k in zip(acts, voiced, keep) if k],
                     [e[v][:, nz] for e, v, k in zip(egemaps, voiced, keep) if k])
    out = torch.empty(plain.shape[:-1] + (len(names),))
    out[..., ~nz], out[..., nz] = plain, voiced_only
    return out


def main(args):
    cfg = load_config(args.config)
    hf_cache = cfg["paths"]["hf_cache"]
    model = Wav2Vec2Model.from_pretrained(cfg["models"][args.model]["hf_id"],
                                          cache_dir=str(hf_cache) if hf_cache else None)
    conv = model.config
    layout = conv_layout(conv)
    layers = list(layout)
    # what each conv_0 channel listens to is in its weights
    filters = model.feature_extractor.conv_layers[0].conv.weight.detach()
    # learned gain per channel of each conv block's norm (all 1 without one); about 0 = switched off
    norm_gain = {layer: block.layer_norm.weight.detach() if hasattr(block, "layer_norm") else torch.ones(dim)
                 for layer, block, dim in zip(layers, model.feature_extractor.conv_layers, conv.conv_dim)}
    print("Active channels (norm gain above %.1f): " % ACTIVE
          + ", ".join(f"{layer} {int((gain.abs() > ACTIVE).sum())}" for layer, gain in norm_gain.items()))
    # with --sae: frames described by SAE units instead of channels, where a frame SAE exists
    transforms = sae_transforms(cfg, args.dataset, args.model, layers, args.sae) if args.sae is not None else {}
    if args.sae is not None:
        layers = list(transforms)
        layout = {layer: layout[layer] for layer in layers}

    names, aligned_a, egemaps, voiced = load_aligned(cfg, args.dataset, args.model, layout, transforms)
    aligned = {"A": aligned_a, "B": {layer: [smooth3(m) for m in means] for layer, means in aligned_a.items()}}
    print(f"{args.model} on {args.dataset}: {len(egemaps)} clips, {sum(map(len, egemaps))} frames, "
          f"{int(sum(v.sum() for v in voiced))} voiced")

    r = {version: {layer: per_descriptor(within_clip_correlation, acts, egemaps, voiced, names)
                   for layer, acts in by_layer.items()} for version, by_layer in aligned.items()}
    r2 = {layer: per_descriptor(heldout_r2, acts, egemaps, voiced, names) for layer, acts in aligned["A"].items()}
    unit = "SAE unit" if transforms else "channel"
    print(f"Held-out R² of all {unit}s, (best single {unit} r²), version A:")
    for name in ("Loudness_sma3", ALPHA_RATIO, F0, "jitterLocal_sma3nz"):
        i = names.index(name)
        print(f"  {name[:20]:>20} " + "  ".join(
            f"{layer} {r2[layer][i]:.2f} ({r['A'][layer][:, i].pow(2).max():.2f})" for layer in layers))

    suffix = f"_sae{args.sae}" if transforms else ""
    out = cfg["paths"]["output_dir"] / "frame_correlation" / f"{args.dataset}_{args.model}{suffix}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": names, "layers": layers, "n_clips": len(egemaps), "r": r, "r2": r2,
                "conv0_filters": filters, "norm_gain": norm_gain, "sae": args.sae}, out)
    print(f"Saved → {out}")
    if not transforms:  # units have no filter weights
        plot_alpha_ratio(r["A"]["conv_0"], names, filters, norm_gain["conv_0"].abs() > ACTIVE, args.model, out.with_name(f"{out.stem}_alpha_ratio.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlate CNN channels with per-frame eGeMAPS.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sae", type=int, metavar="SPARSITY",
                        help="Use the units of the frame SAEs (sae/train_frames.py) with this sparsity "
                             "instead of channels, for the layers that have one.")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
