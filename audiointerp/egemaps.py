"""eGeMAPS voice descriptors: the 88 openSMILE eGeMAPSv02 functionals per clip (pitch,
loudness, jitter, shimmer, formants, MFCC, spectral shape, rhythm), used to name what
layers and SAE units encode.

Features are computed on the same truncated signal the encoders see, and written to
<cache_dir>/egemaps/<dataset>.csv with one row per clip.

    python -m audiointerp.egemaps --dataset aibo
    python -m audiointerp.egemaps --dataset emodb --limit 5   # smoke test
"""
import argparse
import csv

import numpy as np

from audiointerp.config import load_config
from audiointerp.data import DATASETS, SAMPLING_RATE, load_audio, load_index

# The seven factor families of Mariotte et al. 2025; the first keyword hit wins.
FAMILY_RULES = [
    ("pitch", ["F0semitone", "pitch"]),
    ("formants", ["F1", "F2", "F3", "formant"]),
    ("mfcc", ["mfcc"]),
    ("quality", ["jitter", "shimmer", "HNR"]),
    # Before "spectral": loudness_sma3_meanRisingSlope and friends contain "slope".
    ("loudness", ["loudness", "equivalentSoundLevel"]),
    ("spectral", ["alphaRatio", "slope", "Flux", "Hammarberg"]),
    ("rhythm", ["Segments", "Pause", "rate", "Length"]),
]
FAMILIES = [family for family, _ in FAMILY_RULES] + ["other"]


def factor_family(name):
    for family, keywords in FAMILY_RULES:
        if any(keyword.lower() in name.lower() for keyword in keywords):
            return family
    return "other"


def egemaps_path(cfg, dataset):
    return cfg["paths"]["cache_dir"] / "egemaps" / f"{dataset}.csv"


def load_egemaps(cfg, dataset):
    """Clip ids, factor names and the (n_clips, n_factors) matrix, NaN and inf set to 0."""
    path = egemaps_path(cfg, dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python -m audiointerp.egemaps --dataset {dataset}")
    with open(path, newline="") as f:
        reader = csv.reader(f)
        names = next(reader)[2:]
        clip_ids, rows = [], []
        for row in reader:
            clip_ids.append(row[0])
            rows.append([float(value) for value in row[2:]])
    F = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return clip_ids, names, F


def main(args):
    import opensmile  # deferred: the analysis scripts only read the CSV

    cfg = load_config(args.config)
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    clips = load_index(args.dataset, cfg)[: args.limit]
    max_duration = cfg["max_duration_sec"][args.dataset]
    path = egemaps_path(cfg, args.dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"eGeMAPS for {len(clips)} {args.dataset} clips")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for i, clip in enumerate(clips):
            features = smile.process_signal(load_audio(clip.path, max_duration), SAMPLING_RATE)
            if i == 0:
                writer.writerow(["clip", "label"] + list(features.columns))
            writer.writerow([clip.id, clip.label] + [f"{value:.6g}" for value in features.iloc[0]])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(clips)}")
                f.flush()
    print(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract eGeMAPS functionals per clip.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, help="Only the first N clips (smoke test).")
    parser.add_argument("--config", help="Config file (default: config.yaml).")
    main(parser.parse_args())
