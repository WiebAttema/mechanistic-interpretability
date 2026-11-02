# mechanistic-interpretability

Guided research (TUM). Looking at what the layers of frozen speech encoders compute: CNN part vs transformer part, pretrained vs fine-tuned. Tasks/data: speech emotion recognition on FAU AIBO and EMoDB.

Models so far (see config.yaml):
- wav2vec2-base (pretrained only)
- wav2vec2-base-960h (fine-tuned for ASR)
- wav2vec2-large-emotion (audeering, fine-tuned for emotion)
- whisper-small (encoder only)

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # cu126 because the cluster has a TITAN X
pip install -e ".[dev]"
```

Paths are in config.yaml, or use a copy with `--config` / `AUDIOINTERP_CONFIG`.

## Run

```bash
# activations of every layer, mean pooled per clip
python -m audiointerp.extract --dataset emodb --model wav2vec2-base

# linear + MLP probe on every layer
python -m audiointerp.layer_sweep --dataset emodb --models wav2vec2-base

# on the cluster
sbatch jobs/gpujob.sh audiointerp.extract --dataset aibo --model wav2vec2-base
```

Layer names: `cnn` = output of the conv feature encoder, `layer_0` = input to the first transformer block, `layer_k` = output of block k.

SAE code (adapted from LAM4SER) is in audiointerp/sae/.

Tests: `python -m pytest tests/`

## Next

- extract all models on AIBO
- compare CNN vs transformer layers, pretrained vs fine-tuned
- gradient maps
- SAEs on the interesting layers
