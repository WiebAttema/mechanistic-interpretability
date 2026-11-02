"""FAU AIBO and EMoDB: clip index, audio loading and speaker-independent splits."""
from dataclasses import dataclass
from pathlib import Path

import audiofile
import numpy as np

SAMPLING_RATE = 16000
DATASETS = ("aibo", "emodb")

# IS2009 Emotion Challenge 5-class codes.
AIBO_LABELS = {"A": "anger", "E": "emphatic", "N": "neutral", "P": "positive", "R": "rest"}

# AIBO follows the INTERSPEECH 2009 protocol: Ohm school for training with two speakers
# held out for validation, Mont school for test. EMoDB speakers are the first two
# characters of the file name.
SPEAKER_SPLITS = {
    "aibo": {"val": {"Ohm_31", "Ohm_32"}, "test": {f"Mont_{i:02d}" for i in range(1, 26)}},
    "emodb": {"val": {"09", "10"}, "test": {"03", "08"}},
}


@dataclass(frozen=True)
class Clip:
    path: str
    label: str
    speaker: str

    @property
    def id(self):
        return Path(self.path).stem


def load_index(dataset, cfg):
    """Every labelled clip of a dataset, in a fixed order."""
    if dataset == "aibo":
        return _aibo_index(cfg["paths"]["aibo"])
    if dataset == "emodb":
        return _emodb_index(cfg["paths"]["emodb"])
    raise ValueError(f"Unknown dataset {dataset!r}; options: {DATASETS}")


def _aibo_index(root):
    """Each line of the label file is "<chunk> <label code> <confidence>". The confidence
    is ignored: every chunk is used with its majority label."""
    labels_file = Path(root) / "labels" / "IS2009EmotionChallenge" / "chunk_labels_5cl_corpus.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"AIBO label file not found: {labels_file} (set paths.aibo in the config)")
    clips = []
    for line in labels_file.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        chunk, code = parts[0], parts[1]
        speaker = "_".join(chunk.split("_")[:2])  # "Mont_01_000_00" -> "Mont_01"
        clips.append(Clip(str(Path(root) / "wav" / f"{chunk}.wav"), AIBO_LABELS[code], speaker))
    return clips


def _emodb_index(cache_root):
    import audb  # deferred: slow to import and only needed for EMoDB

    db = audb.load("emodb", version="2.0.0", sampling_rate=SAMPLING_RATE, mixdown=True,
                   format="wav", cache_root=str(cache_root) if cache_root else None, verbose=False)
    emotions = db["emotion"].get()["emotion"]
    return [Clip(path, emotion, Path(path).name[:2]) for path, emotion in emotions.items()]


def label_names(clips):
    """Sorted class names; a clip's label index is its position in this list."""
    return sorted({clip.label for clip in clips})


def load_audio(path, max_duration_sec=None):
    """Mono 16 kHz float32 signal, truncated to max_duration_sec."""
    signal, sr = audiofile.read(path, always_2d=False)
    if sr != SAMPLING_RATE or signal.ndim != 1:
        raise ValueError(f"{path}: expected mono {SAMPLING_RATE} Hz audio, got shape {signal.shape} at {sr} Hz")
    if max_duration_sec is not None:
        signal = signal[: int(max_duration_sec * SAMPLING_RATE)]
    return signal.astype(np.float32)


def split_indices(speakers, dataset):
    """Speaker-independent {"train", "val", "test"} -> clip indices."""
    split = SPEAKER_SPLITS[dataset]
    indices = {"train": [], "val": [], "test": []}
    for i, speaker in enumerate(speakers):
        if speaker in split["test"]:
            indices["test"].append(i)
        elif speaker in split["val"]:
            indices["val"].append(i)
        else:
            indices["train"].append(i)
    return indices
