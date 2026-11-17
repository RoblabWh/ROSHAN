"""Scenarios: a map plus everything placed on it, stored as one .npz file."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Scenario:
    """Arrays are [ny, nx] with row 0 in the north; placements are (x, y) cells.

    fuel       uint8, 0 = non-burnable, 1..40 = position in roshan._core.fuel_models() + 1
    elevation  float32 metres, or None for a flat map
    landcover  uint8 source land-cover class, kept for reference only (may be None)
    moisture   fractions for 1-h, 10-h, 100-h dead fuel, live herbs, live woody
    wind       list of (start time s, speed at 10 m in m/s, to_x, to_y); (to_x, to_y) is the unit
               vector the wind blows towards, x to the east and y to the south
    meta       free-form: name, georeference, data sources
    """

    fuel: np.ndarray
    cell_m: float = 10.0
    elevation: np.ndarray | None = None
    landcover: np.ndarray | None = None
    ignitions: list = field(default_factory=list)
    drones: list = field(default_factory=list)
    stations: list = field(default_factory=list)
    moisture: tuple = (0.06, 0.07, 0.08, 0.60, 0.90)
    wind: list = field(default_factory=lambda: [(0.0, 5.0, 1.0, 0.0)])
    meta: dict = field(default_factory=dict)

    def _plain(self):
        cells = lambda xs: [[int(x), int(y)] for x, y in xs]  # noqa: E731
        return {
            "schema_version": SCHEMA_VERSION,
            "cell_m": float(self.cell_m),
            "ignitions": cells(self.ignitions),
            "drones": cells(self.drones),
            "stations": cells(self.stations),
            "moisture": [round(float(m), 6) for m in self.moisture],
            "wind": [[round(float(v), 6) for v in w] for w in self.wind],
            "meta": self.meta,
        }

    def to_dict(self):
        """The form the simulator and the viewer take."""
        ny, nx = self.fuel.shape
        plain = self._plain()
        return {
            "nx": nx,
            "ny": ny,
            "cell_m": plain["cell_m"],
            "fuel": np.ascontiguousarray(self.fuel, np.uint8),
            "elevation": None if self.elevation is None else np.ascontiguousarray(self.elevation, np.float32),
            "landcover": None if self.landcover is None else np.ascontiguousarray(self.landcover, np.uint8),
            **{k: plain[k] for k in ("ignitions", "drones", "stations", "moisture", "wind")},
            "meta": json.dumps(self.meta, sort_keys=True),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            fuel=np.array(d["fuel"], np.uint8),
            cell_m=d["cell_m"],
            elevation=None if d["elevation"] is None else np.array(d["elevation"], np.float32),
            landcover=None if d.get("landcover") is None else np.array(d["landcover"], np.uint8),
            ignitions=[tuple(c) for c in d["ignitions"]],
            drones=[tuple(c) for c in d["drones"]],
            stations=[tuple(c) for c in d["stations"]],
            moisture=tuple(d["moisture"]),
            wind=[tuple(w) for w in d["wind"]],
            meta=json.loads(d["meta"]) if d["meta"] else {},
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"fuel": np.asarray(self.fuel, np.uint8)}
        if self.elevation is not None:
            arrays["elevation"] = np.asarray(self.elevation, np.float32)
        if self.landcover is not None:
            arrays["landcover"] = np.asarray(self.landcover, np.uint8)
        np.savez_compressed(path, plain=json.dumps(self._plain(), sort_keys=True), **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            plain = json.loads(str(f["plain"]))
            if plain["schema_version"] != SCHEMA_VERSION:
                raise ValueError(f"{path}: schema version {plain['schema_version']}, expected {SCHEMA_VERSION}")
            arrays = {k: f[k] for k in ("fuel", "elevation", "landcover") if k in f}
        plain.pop("schema_version")
        placed = {k: [tuple(c) for c in plain.pop(k)] for k in ("ignitions", "drones", "stations")}
        wind = [tuple(w) for w in plain.pop("wind")]
        return cls(**arrays, **placed, wind=wind, moisture=tuple(plain.pop("moisture")), **plain)

    def fingerprint(self):
        """Digest of everything that defines the scenario; equal means the same scenario."""
        h = hashlib.sha256(json.dumps(self._plain(), sort_keys=True).encode())
        for a, dtype in ((self.fuel, np.uint8), (self.elevation, np.float32), (self.landcover, np.uint8)):
            h.update(b"-" if a is None else np.ascontiguousarray(a, dtype).tobytes())
            h.update(b"-" if a is None else str(a.shape).encode())
        return h.hexdigest()[:16]
