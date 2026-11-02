# Audio encoder interpretability

What do the layers of a frozen speech encoder compute, from the CNN feature encoder
through every transformer block, and how do pretrained and fine-tuned checkpoints
differ? This repo studies wav2vec 2.0 models and the Whisper encoder on speech
emotion recognition (FAU AIBO, EMoDB).

Methods so far: per-layer linear and MLP probes, TopK sparse autoencoders (SAEs),
eGeMAPS disentanglement (openSMILE + Lasso), and class-selective SAE unit ablation.

## Setup

```bash
# 1. PyTorch for your GPU, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. This package and its dependencies (add [dev] for pytest)
pip install -e ".[dev]"
```

Then set the paths in `config.yaml`, or point `--config` (or `AUDIOINTERP_CONFIG`) to a
copy per machine:

| Key | What |
|---|---|
| `paths.aibo` | FAU AIBO root, containing `wav/` and `labels/IS2009EmotionChallenge/` |
| `paths.emodb` | audb cache for EMoDB (`null` = `~/audb`; audb downloads it on first use) |
| `paths.hf_cache` | Hugging Face model cache (`null` = HF default) |
| `paths.cache_dir` | extracted activations and eGeMAPS features |
| `paths.output_dir` | probe CSVs, SAE checkpoints, figures |

The same file holds the model registry (`models`) and the extraction settings. To add a
model, add an entry with its Hugging Face id and `family: wav2vec2` or `family: whisper`.

## Layout

```
config.yaml             paths, models, extraction settings
audiointerp/
  data.py               AIBO and EMoDB index, audio loading, speaker-independent splits
  encoders.py           wav2vec2 and Whisper adapters: every layer's activations for one clip
  extract.py            CLI: pooled activations for all clips, per-frame for a subset
  activations.py        reading what extract.py wrote
  probes.py             linear and MLP probes, metrics
  layer_sweep.py        CLI: a probe on every layer
  egemaps.py            CLI: openSMILE eGeMAPS features per clip, factor families
  plotting.py           colors and axis style
  config.py, utils.py   config loading, device choice, CSV helpers
  sae/
    model.py            TopK SAE, training, checkpoints
    train.py            CLI: SAEs per (model, layer, sparsity), with a probe on the codes
    disentanglement.py  CLI: Lasso from codes to eGeMAPS factors
    selective_units.py  CLI: class-selective units and their ablation
    factor_units.py     CLI: ablation of units ranked by eGeMAPS factor
tests/test_smoke.py
```

Every CLI runs as `python -m audiointerp.<module>` (from anywhere after `pip install -e .`,
otherwise from the repo root), takes `--dataset` (`aibo` or `emodb`) and `--config`, and
prints its options with `--help`.

## Quick start

```bash
# 1. Activations of every layer (once per dataset and model)
python -m audiointerp.extract --dataset emodb --model wav2vec2-base

# 2. A linear and an MLP probe on every layer
python -m audiointerp.layer_sweep --dataset emodb --models wav2vec2-base
# ... or just some layers, several models, one probe type
python -m audiointerp.layer_sweep --dataset emodb --models wav2vec2-base whisper-small \
    --layers cnn layer_6 layer_12 --probes linear
```

Results go to `<output_dir>/layer_sweep/<dataset>_<model>.csv` with a figure next to it.
Probes keep their best epoch on validation UAR, and the CSV stores that `val_uar`
alongside the test metrics, so you can choose layers on validation and report test.

## Activation files

`extract.py` writes to `<cache_dir>/activations/<dataset>/<model>/`:

- `pooled.pt`: every clip, every layer, mean-pooled over the clip's real frames
  (`pooled[layer]` is `(n_clips, dim)`), plus labels, speakers, file paths and durations.
- `frames/<clip id>.pt`: per-frame activations (`(frames, dim)` per layer, float16) for
  `frames.per_class` random clips per class from `frames.split`.

Layer names are the same for both families:

| Name | wav2vec2 | Whisper |
|---|---|---|
| `conv_i` | conv block i, 512 channels (only with `extraction.conv_layers: true`) | first conv, 100 frames/s (same flag) |
| `cnn` | output of the last conv block (the CNN feature encoder), 512-dim, 50 frames/s | second conv, model width, 50 frames/s |
| `layer_0` | CNN features projected to model width + positional conv embedding | conv output + positional embedding |
| `layer_k` | output of transformer block k | output of transformer block k |

Two things to keep in mind when comparing layers:

- **Padding.** wav2vec2 gets the un-padded waveform, so every frame is real speech.
  The Whisper encoder only accepts a 30 s window, so every clip is padded to it and
  each layer's output is trimmed back to the clip's duration before pooling. Output
  frames map linearly to input time, so the first `ceil(frames * clip / window)` frames
  of every layer are the ones that cover the clip.
- **LayerNorm placement.** wav2vec2-base is post-LN. wav2vec2-large-robust (and the
  audeering model built from it) and Whisper are pre-LN with a final LayerNorm, so only
  their last `layer_k` is normalized. The probes and SAEs z-score every layer, but raw
  activation scales are not comparable across depth.

Clips are truncated to `max_duration_sec` (AIBO 5 s, which covers ~99% of clips). The
eGeMAPS features use the same truncated signal.

## Sparse autoencoders

```bash
python -m audiointerp.sae.train            --dataset aibo --models wav2vec2-base   # needs the layer sweep for --layers best
python -m audiointerp.egemaps              --dataset aibo
python -m audiointerp.sae.disentanglement  --dataset aibo --models wav2vec2-base --cv
python -m audiointerp.sae.selective_units  --dataset aibo --models wav2vec2-base
python -m audiointerp.sae.factor_units     --dataset aibo --models wav2vec2-base
```

`sae.train` trains one SAE per layer (`best` = highest linear-probe `val_uar` in the
sweep, and `final`) and sparsity (75 to 99%), and checks reconstruction MSE plus a
linear probe on the codes against the probe on the original layer. Long runs write
rows as they go; pass `--resume` after a time limit. Outputs go to
`<output_dir>/sae/<dataset>/`.

Choices that differ from Mariotte et al. 2025 (arXiv 2509.24793):

1. Inputs are z-scored with train statistics (stored in each checkpoint). Mid-network
   layers are not as well scaled as the final layer the paper used.
2. Tied initialisation and unit-norm decoder rows, the standard TopK stabilisers.
3. R² is held-out (`--holdout`, 70/30 inside the test subsample), not in-sample.
4. Dictionary size 4096 (`--dict-size`).
5. The task is SER, so probes are selected on validation UAR.

Things to watch for:

- Use `disentanglement --cv` for reported numbers. A single fixed Lasso penalty
  constrains the codes and the original layer about equally hard and hides the
  completeness difference between them.
- Dead units are never resampled, so longer training loses more units rather than
  fewer. A high `dead_frac` at 99% sparsity is expected; at 75 to 90% it is a sign of
  this limitation.
- Selective-unit counts depend strongly on `--threshold` and `--min-active`. Both are
  written into the summary CSVs; never quote a count without them.
- With many units ablated the effect can stop being class-specific and start hurting
  other classes; `max_other_class_drop` in the summary shows where that happens.

## Datasets

**FAU AIBO**: 18,216 German segments of children talking to a robot, 5 classes,
INTERSPEECH 2009 split: 24 Ohm speakers train (9,068), Ohm_31 and Ohm_32 validation
(891), all 25 Mont speakers test (8,257). The test set is 65% neutral, so UAR is the
metric that matters.

**EMoDB**: 816 acted German utterances, 10 speakers, 7 emotions, loaded through audb.
Train on speakers 11 to 16 (493), validate on 09 and 10 (161), test on 03 and 08 (162).

## Tests

```bash
python -m pytest tests/
```
