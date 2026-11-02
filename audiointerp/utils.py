"""Small helpers shared by the scripts: device choice and CSV files."""
import csv
from pathlib import Path

import torch


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse(value):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return {"True": True, "False": False}.get(value, value)


def read_csv(path):
    """Rows as dicts, with numbers and booleans converted back from text."""
    with open(path, newline="") as f:
        return [{key: _parse(value) for key, value in row.items()} for row in csv.DictReader(f)]


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


class CSVLog:
    """Append-only CSV for long runs. Every row is flushed as soon as it is written, so a
    job killed by a time limit keeps its results; with resume=True the rows of an
    earlier run are kept and available as `rows`, so finished work can be skipped."""

    def __init__(self, path, fields, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if resume and self.path.exists():
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), None)
            if header != fields:
                raise ValueError(f"{self.path} has different columns; delete it or run without --resume.")
            self.rows = read_csv(self.path)
            print(f"Resuming: {len(self.rows)} rows already in {self.path}")
        self._file = open(self.path, "a" if self.rows else "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        if not self.rows:
            self._writer.writeheader()

    def write(self, row):
        self._writer.writerow(row)
        self._file.flush()
        self.rows.append(row)

    def close(self):
        self._file.close()
        print(f"Saved → {self.path}")
