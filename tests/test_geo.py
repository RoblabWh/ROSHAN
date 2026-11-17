"""Scenario files and the terrain pipeline. No network: the two downloads are replaced."""

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from roshan import _core, geo
from roshan.scenario import Scenario


def test_crosswalk_maps_classes_to_fuel_ids():
    names = _core.fuel_models()
    assert len(names) == 40 and names[0] == "GR1" and names[-1] == "SB4"
    fuel = geo.to_fuel(np.array([[1, 2, 6], [10, 255, 0]], np.uint8))
    assert fuel.tolist() == [[0, names.index("TU2") + 1, names.index("GR6") + 1], [0, 0, 0]]
    assert set(geo.CROSSWALK) == set(range(1, 12)), "every CLC+ class has a decision"


def test_tiff_bytes_decode_to_the_class_array():
    classes = np.arange(12, dtype=np.uint8).reshape(3, 4)
    profile = dict(driver="GTiff", width=4, height=3, count=1, dtype="uint8", crs="EPSG:3035",
                   transform=from_origin(4_000_000, 3_000_000, 10, 10))
    with MemoryFile() as memory:
        with memory.open(**profile) as ds:
            ds.write(classes, 1)
        assert np.array_equal(geo._read_tiff(memory.read()), classes)


def test_fetch_builds_an_aligned_scenario(monkeypatch):
    asked = {}

    def landcover(bounds, nx, ny):
        asked["landcover"] = (bounds, nx, ny)
        return np.full((ny, nx), 6, np.uint8)

    def elevation(bounds, nx, ny, cell_m):
        asked["elevation"] = (bounds, nx, ny)
        return np.tile(np.arange(nx, dtype=np.float32), (ny, 1))

    monkeypatch.setattr(geo, "_landcover", landcover)
    monkeypatch.setattr(geo, "_elevation", elevation)
    s = geo.fetch(51.72, 7.19, 1.5, 0.8)

    assert s.fuel.shape == s.elevation.shape == s.landcover.shape == (80, 150)
    bounds, nx, ny = asked["landcover"]
    assert asked["elevation"] == asked["landcover"], "both layers share one grid"
    assert all(b % 10 == 0 for b in bounds) and (bounds[2] - bounds[0], bounds[3] - bounds[1]) == (1500, 800)
    assert s.meta["origin"] == [bounds[0], bounds[3]] and s.meta["crs"] == "EPSG:3035"
    # Haltern am See in EPSG:3035 is about (4.13e6, 3.18e6).
    assert 4.12e6 < bounds[0] < 4.14e6 and 3.17e6 < bounds[1] < 3.19e6
    assert (s.fuel == _core.fuel_models().index("GR6") + 1).all()
    _core.check_scenario(s.to_dict())
    with pytest.raises(ValueError, match="between"):
        geo.fetch(51.72, 7.19, 0.01, 0.01)


def test_save_and_load_give_the_same_scenario(tmp_path):
    s = Scenario.from_dict(_core.demo_scenario())
    s.landcover = np.full(s.fuel.shape, 6, np.uint8)
    s.meta["origin"] = [4126370, 3182980]
    loaded = Scenario.load(s.save(tmp_path / "demo.npz"))
    assert loaded.fingerprint() == s.fingerprint()
    assert loaded.ignitions == [(20, 60)] and loaded.stations == [(3, 3)] and len(loaded.drones) == 4
    assert np.array_equal(loaded.elevation, s.elevation) and loaded.meta == s.meta

    loaded.drones.append((7, 7))
    assert loaded.fingerprint() != s.fingerprint()
    flat = Scenario(fuel=np.ones((4, 6), np.uint8))
    assert Scenario.load(flat.save(tmp_path / "flat.npz")).fingerprint() == flat.fingerprint()


def test_the_simulator_round_trips_a_scenario_and_rejects_bad_ones():
    s = Scenario.from_dict(_core.demo_scenario())
    back = Scenario.from_dict(_core.check_scenario(s.to_dict()))
    assert back.fingerprint() == s.fingerprint()

    d = s.to_dict()
    with pytest.raises(ValueError, match="outside"):
        _core.check_scenario({**d, "drones": [[500, 0]]})
    with pytest.raises(ValueError, match="fuel id"):
        _core.check_scenario({**d, "fuel": np.full_like(d["fuel"], 99)})
    with pytest.raises(ValueError, match="shape"):
        _core.check_scenario({**d, "nx": d["nx"] + 1})
    with pytest.raises(KeyError):
        _core.check_scenario({k: v for k, v in d.items() if k != "wind"})


def test_old_maps_import_as_scenarios(tmp_path):
    path = tmp_path / "old.tif"
    classes = np.array([[2, 2, 6], [10, 6, 6]], np.float32)  # the old maps store classes as floats
    with rasterio.open(path, "w", driver="GTiff", width=3, height=2, count=1, dtype="float32",
                       crs="EPSG:3035", transform=from_origin(4_126_000, 3_183_000, 10, 10)) as ds:
        ds.write(classes, 1)
    s = geo.import_tif(path)
    assert s.fuel.shape == (2, 3) and s.cell_m == 10.0 and s.fuel[1, 0] == 0 and s.fuel[0, 0] > 0
    assert s.meta["name"] == "old" and s.meta["origin"] == [4_126_000, 3_183_000]
    _core.check_scenario(s.to_dict())
