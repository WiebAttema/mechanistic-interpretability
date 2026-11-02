"""Loads config.yaml: data locations, cache and output folders, and the model registry."""
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path=None):
    """Read the config from `path`, else $AUDIOINTERP_CONFIG, else the repo's config.yaml.

    Entries under `paths` become absolute Paths, resolved against the config file's
    folder, so every script behaves the same from any working directory.
    """
    path = Path(path or os.environ.get("AUDIOINTERP_CONFIG") or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    for key, value in cfg["paths"].items():
        if value is not None:
            cfg["paths"][key] = (path.parent / Path(value).expanduser()).resolve()
    return cfg
