"""Frozen audio encoders that return the activations of every layer for one clip.

Both families share one interface:

    encoder = load_encoder(model_cfg, device)
    encoder.layer_names     # ["cnn", "layer_0", ..., "layer_12"]
    acts = encoder(signal)  # {layer name: (frames, dim) float32 tensor on the CPU}

Layer names mean the same thing in both families:

    conv_i    output of conv block i of the CNN front end (only with conv_layers=True;
              the last block is always kept, as `cnn`)
    cnn       output of the last conv block, i.e. the CNN feature encoder
    layer_0   input to the first transformer block: CNN features projected to the model
              width plus positional information (hidden_states[0] in transformers)
    layer_k   output of transformer block k (hidden_states[k])

Only frames that cover the real clip are returned, so pooling never averages padding.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, Wav2Vec2Model, WhisperModel

from audiointerp.data import SAMPLING_RATE


def real_frames(n_frames, clip_samples, window_samples):
    """How many leading output frames of a padded window cover the real clip.

    Output frames map linearly to input time (the convs are local, and attention
    mixes content between frames without moving them), so a clip filling a fraction
    of the window is covered by the same fraction of every layer's frames. This is
    computed per layer because layers can run at different frame rates.
    """
    return min(n_frames, max(1, math.ceil(n_frames * clip_samples / window_samples)))


class _Encoder:
    """Shared plumbing: capture conv outputs with forward hooks and name every layer."""

    def __init__(self):
        self._captured = {}

    def _capture(self, module, name, activation=None):
        def hook(_module, _inputs, output):
            if activation is not None:
                output = activation(output)
            self._captured[name] = output[0].transpose(0, 1)  # (channels, frames) -> (frames, channels)
        module.register_forward_hook(hook)

    def _activations(self, hidden_states):
        acts = dict(self._captured)
        acts.update({f"layer_{k}": h[0] for k, h in enumerate(hidden_states)})
        return acts


class Wav2Vec2Encoder(_Encoder):
    """wav2vec 2.0: seven conv blocks on the raw waveform (50 frames/s at the output),
    then a transformer. The waveform is fed un-padded, so every frame is real speech."""

    # The conv stack needs at least 400 samples to produce one frame.
    MIN_SAMPLES = 640

    def __init__(self, hf_id, device, conv_layers=False, cache_dir=None):
        super().__init__()
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id, cache_dir=cache_dir)
        self.model = Wav2Vec2Model.from_pretrained(hf_id, cache_dir=cache_dir).to(device).eval()

        blocks = self.model.feature_extractor.conv_layers
        conv_names = [f"conv_{i}" for i in range(len(blocks) - 1)] if conv_layers else []
        for name, block in zip(conv_names, blocks):
            self._capture(block, name)
        self._capture(blocks[-1], "cnn")
        n_layers = self.model.config.num_hidden_layers
        self.layer_names = conv_names + ["cnn"] + [f"layer_{k}" for k in range(n_layers + 1)]

    @torch.no_grad()
    def __call__(self, signal):
        if len(signal) < self.MIN_SAMPLES:
            signal = np.pad(signal, (0, self.MIN_SAMPLES - len(signal)))
        inputs = self.feature_extractor(signal, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        out = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        acts = self._activations(out.hidden_states)
        return {name: acts[name].float().cpu() for name in self.layer_names}


class WhisperEncoder(_Encoder):
    """Whisper encoder: two convs on the log-mel spectrogram (100, then 50 frames/s),
    then a transformer. It only accepts a fixed 30 s window, so every clip is padded to
    it and each layer's output is trimmed back to the clip's own duration."""

    # 10 ms mel hop at 16 kHz: one mel frame covers 160 input samples.
    MEL_HOP_LENGTH = 160

    def __init__(self, hf_id, device, conv_layers=False, cache_dir=None):
        super().__init__()
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_id, cache_dir=cache_dir)
        whisper = WhisperModel.from_pretrained(hf_id, cache_dir=cache_dir)
        self.model = whisper.get_encoder().to(device).eval()

        # The encoder applies GELU outside the conv modules, so the hooks apply it too.
        conv_names = ["conv_0"] if conv_layers else []
        if conv_layers:
            self._capture(self.model.conv1, "conv_0", F.gelu)
        self._capture(self.model.conv2, "cnn", F.gelu)
        n_layers = self.model.config.encoder_layers
        self.layer_names = conv_names + ["cnn"] + [f"layer_{k}" for k in range(n_layers + 1)]

    @torch.no_grad()
    def __call__(self, signal):
        features = self.feature_extractor(signal, sampling_rate=SAMPLING_RATE,
                                          return_tensors="pt").input_features
        out = self.model(features.to(self.device), output_hidden_states=True)
        acts = self._activations(out.hidden_states)
        window_samples = features.shape[-1] * self.MEL_HOP_LENGTH
        return {name: acts[name][: real_frames(len(acts[name]), len(signal), window_samples)].float().cpu()
                for name in self.layer_names}


FAMILIES = {"wav2vec2": Wav2Vec2Encoder, "whisper": WhisperEncoder}


def load_encoder(model_cfg, device, conv_layers=False, cache_dir=None):
    """Build the encoder for one entry of the config's `models`."""
    return FAMILIES[model_cfg["family"]](model_cfg["hf_id"], device, conv_layers, cache_dir)
