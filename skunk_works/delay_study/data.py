"""Validated experiment inputs and an explicit audit of duplicate records."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Experiment:
    name: str
    time: np.ndarray
    temperature: np.ndarray
    input: np.ndarray
    ambient: float
    power: float = 200.0
    source: str = "synthetic"

    def __post_init__(self):
        for name in ("time", "temperature", "input"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if len(self.time) < 2 or not (len(self.time) == len(self.temperature) == len(self.input)):
            raise ValueError("Experiment arrays must have equal lengths >= 2")
        if not all(np.all(np.isfinite(getattr(self, key))) for key in ("time", "temperature", "input")):
            raise ValueError("Nonfinite experiment data")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("Time must be strictly increasing")
        if not np.isfinite(self.ambient) or not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("Invalid ambient temperature or heater power")


def load(name: str) -> Experiment:
    aliases = {"sine": "tclab_sine_test_5min_period.csv", "step": "tclab_step_test.csv"}
    path = ROOT / "data" / aliases.get(name, name)
    frame = pd.read_csv(path)
    return Experiment(name, frame.Time.values, frame.T1.values, frame.Q1.values,
                      float(frame.T1.iloc[0]), source=str(path.relative_to(ROOT)))


def audit() -> dict:
    rows = []
    for path in sorted((ROOT / "data").glob("*.csv")):
        frame = pd.read_csv(path)
        array = np.asarray(frame[["Time", "T1", "T2", "Q1", "Q2"]], dtype="<f8")
        signature = hashlib.sha256(array.tobytes()).hexdigest()
        d = load(path.name)
        rows.append({"file": str(path.relative_to(ROOT)), "n": len(d.time),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "numeric_sha256": signature, "duration_s": float(d.time[-1]-d.time[0]),
                     "ambient_assumed_C": d.ambient,
                     "second_heater_active": bool(np.any(frame.Q2.values != 0)),
                     "dt_min_s": float(np.diff(d.time).min()),
                     "dt_max_s": float(np.diff(d.time).max()),
                     "unique_dt_rounded_9dp": len(np.unique(np.round(np.diff(d.time), 9)))})
    groups = {}
    for row in rows:
        groups.setdefault(row["numeric_sha256"], []).append(row["file"])
    return {"files": rows, "exact_numeric_duplicates": [v for v in groups.values() if len(v) > 1],
            "ambient_status": "First recorded sensor temperature assumed ambient; not independently measured",
            "power_status": "P=200 and alpha=0.00016 inherited from workshop; calibration uncertainty unknown",
            "validation_policy": "Exclude all numeric duplicates of calibration data from held-out evaluation; filenames alone do not establish independent replication."}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
