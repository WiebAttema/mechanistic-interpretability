"""Fast checks that need no data or downloads: every module imports, Whisper trimming
keeps the right frames, and the SAE and the probes train on synthetic data."""
import importlib

import pytest
import torch

MODULES = [
    "audiointerp.config", "audiointerp.utils", "audiointerp.data", "audiointerp.plotting",
    "audiointerp.encoders", "audiointerp.activations", "audiointerp.extract", "audiointerp.probes",
    "audiointerp.layer_sweep", "audiointerp.egemaps", "audiointerp.sae.model", "audiointerp.sae.train",
    "audiointerp.sae.disentanglement", "audiointerp.sae.selective_units", "audiointerp.sae.factor_units",
]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)


def test_real_frames():
    from audiointerp.encoders import real_frames

    window = 30 * 16000
    assert real_frames(1500, 3 * 16000, window) == 150   # 3 s of a 30 s window
    assert real_frames(3000, 3 * 16000, window) == 300   # the first conv runs at twice the rate
    assert real_frames(1500, 1, window) == 1             # never empty
    assert real_frames(1500, 2 * window, window) == 1500  # never more than the window


def test_layer_sort_key():
    from audiointerp.activations import layer_sort_key

    names = ["layer_10", "cnn", "layer_2", "conv_1", "layer_0", "conv_0"]
    assert sorted(names, key=layer_sort_key) == ["conv_0", "conv_1", "cnn", "layer_0", "layer_2", "layer_10"]


def test_sae_codes_are_sparse():
    from audiointerp.sae.model import train_sae

    X = torch.randn(256, 16)
    sae, info = train_sae(X, X[:64], n_dict=64, k=4, epochs=2)
    z = sae.encode(X)
    assert ((z > 0).sum(dim=1) <= 4).all()
    assert torch.allclose(sae.decoder.norm(dim=1), torch.ones(64), atol=1e-5)
    assert 0.0 <= info["dead_frac"] <= 1.0


def test_probe_learns_separable_data():
    from audiointerp.probes import LinearProbe, balanced_class_weights, train_probe

    torch.manual_seed(0)
    X = torch.randn(400, 8)
    y = (X[:, 0] > 0).long()
    _, val_uar = train_probe(LinearProbe(8, 2), X[:300], y[:300], X[300:], y[300:],
                             balanced_class_weights(y[:300], 2), "cpu", epochs=100, batch_size=32)
    assert val_uar > 0.9
