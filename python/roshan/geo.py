"""Real terrain for scenarios: land cover and elevation for an area in Europe.

    uv run python -m roshan.geo fetch --lat 51.72 --lon 7.19 --km 1.5 1.0 --out scenarios/haard.npz
    uv run python -m roshan.geo import-tif old_map.tif --out scenarios/old_map.npz
    uv run python -m roshan.geo refuel scenarios/haard.npz       # after the fuel table changed

Land cover: Copernicus CLC+ Backbone 2021, 10 m, from the EEA image server (no account).
Elevation: Copernicus DEM GLO-30 from its public AWS bucket, resampled bilinearly.
Both land on one grid in EPSG:3035 (metres), row 0 in the north.
"""

import argparse
import math
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform

from roshan import _core
from roshan.scenario import Scenario

LANDCOVER_URL = (
    "https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/CLC_plus/"
    "CLMS_CLCplus_RASTER_2021_010m_eu/ImageServer/exportImage"
)
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
MAX_CELLS = (15000, 4100)  # largest image the land-cover server returns in one request

# CLC+ Backbone class -> Scott & Burgan fuel model (None = does not burn). The one place where
# land cover turns into fire behaviour. From the European fuel classification of Aragoneses
# et al. 2023 (Earth Syst. Sci. Data 15, Table 5, humid regime), with forest taken as closed
# because CLC+ carries no canopy cover. Heath is SH8 because it is closest to the spread rates
# observed in Calluna (docs/systems.md, "Calibration"). After changing this table, rebuild the
# fuel of saved scenarios with `python -m roshan.geo refuel`.
CROSSWALK = {
    1: None,  # sealed
    2: "TU2",  # woody, needle-leaved trees   -> closed needleleaf forest
    3: "TU3",  # woody, broadleaved deciduous -> closed broadleaf deciduous forest
    4: "TU2",  # woody, broadleaved evergreen -> closed broadleaf evergreen forest
    5: "SH8",  # low-growing woody plants     -> medium shrub
    6: "GR6",  # permanent herbaceous         -> low grass
    7: "GR6",  # periodically herbaceous      -> herbaceous cropland
    8: "GR1",  # lichens and mosses           (not in the classification: short, sparse grass)
    9: None,  # non- and sparsely vegetated
    10: None,  # water
    11: None,  # snow and ice
}


def to_fuel(landcover):
    """Fuel ids for an array of CLC+ classes. Classes not in the table do not burn."""
    names = _core.fuel_models()
    table = np.zeros(256, np.uint8)
    for cls, model in CROSSWALK.items():
        table[cls] = 0 if model is None else names.index(model) + 1
    return table[np.asarray(landcover, np.uint8)]


def _read_tiff(data):
    with MemoryFile(data) as memory, memory.open() as ds:
        return ds.read(1)


def _landcover(bounds, nx, ny):
    query = urllib.parse.urlencode(
        {
            "bbox": ",".join(str(b) for b in bounds),
            "bboxSR": 3035,
            "imageSR": 3035,
            "size": f"{nx},{ny}",
            "format": "tiff",
            "pixelType": "U8",
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        }
    )
    with urllib.request.urlopen(f"{LANDCOVER_URL}?{query}", timeout=120) as response:
        classes = _read_tiff(response.read())
    if classes.shape != (ny, nx):
        raise RuntimeError(f"land-cover server returned {classes.shape}, asked for {(ny, nx)}")
    return classes.astype(np.uint8)


def _elevation(bounds, nx, ny, cell_m):
    x0, y0, x1, y1 = bounds
    lon, lat = transform("EPSG:3035", "EPSG:4326", [x0, x1, x0, x1], [y0, y0, y1, y1])
    grid = from_origin(x0, y1, cell_m, cell_m)
    out = np.full((ny, nx), np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for la in range(math.floor(min(lat)), math.floor(max(lat)) + 1):
            for lo in range(math.floor(min(lon)), math.floor(max(lon)) + 1):
                tile = "Copernicus_DSM_COG_10_{}{:02d}_00_{}{:03d}_00_DEM".format(
                    "N" if la >= 0 else "S", abs(la), "E" if lo >= 0 else "W", abs(lo)
                )
                try:
                    source = rasterio.open(DEM_URL.format(name=tile))
                except rasterio.errors.RasterioIOError:
                    continue  # no tile: open sea
                with source:
                    part = np.full((ny, nx), np.nan, np.float32)
                    reproject(
                        rasterio.band(source, 1), part, dst_transform=grid, dst_crs="EPSG:3035",
                        dst_nodata=np.nan, resampling=Resampling.bilinear,
                    )
                out = np.where(np.isnan(out), part, out)
    return np.nan_to_num(out, nan=0.0)


def fetch(lat, lon, width_km, height_km, cell_m=10.0):
    """A scenario with real land cover and elevation around (lat, lon). Nothing is placed on it."""
    nx, ny = round(width_km * 1000 / cell_m), round(height_km * 1000 / cell_m)
    if not (3 <= nx <= MAX_CELLS[0] and 3 <= ny <= MAX_CELLS[1]):
        raise ValueError(f"{nx} x {ny} cells; the map must be between 3 x 3 and {MAX_CELLS[0]} x {MAX_CELLS[1]}")
    (cx,), (cy,) = transform("EPSG:4326", "EPSG:3035", [lon], [lat])
    # Snap to the 10 m grid of the source raster so cells line up with its pixels.
    x0 = round((cx - nx * cell_m / 2) / 10) * 10
    y1 = round((cy + ny * cell_m / 2) / 10) * 10
    bounds = (x0, y1 - ny * cell_m, x0 + nx * cell_m, y1)
    landcover = _landcover(bounds, nx, ny)
    return Scenario(
        fuel=to_fuel(landcover),
        cell_m=cell_m,
        elevation=_elevation(bounds, nx, ny, cell_m),
        landcover=landcover,
        meta={
            "name": f"{lat:.4f}N_{lon:.4f}E",
            "crs": "EPSG:3035",
            "origin": [x0, y1],
            "centre": [lat, lon],
            "sources": ["CLC+ Backbone 2021 (EEA)", "Copernicus DEM GLO-30"],
        },
    )


def import_tif(path):
    """A scenario from a GeoTIFF of CLC+ classes, such as the maps of the old ROSHAN."""
    with rasterio.open(path) as ds:
        landcover = ds.read(1).astype(np.uint8)
        cell_m = abs(ds.transform.a) or 10.0
        origin = [ds.transform.c, ds.transform.f]
        crs = ds.crs.to_string() if ds.crs else None
    name = str(path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return Scenario(
        fuel=to_fuel(landcover), cell_m=cell_m, landcover=landcover,
        meta={"name": name, "crs": crs, "origin": origin, "sources": [str(path)]},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("fetch")
    f.add_argument("--lat", type=float, required=True)
    f.add_argument("--lon", type=float, required=True)
    f.add_argument("--km", type=float, nargs=2, default=(1.5, 1.0), metavar=("WIDTH", "HEIGHT"))
    f.add_argument("--out", required=True)
    t = commands.add_parser("import-tif")
    t.add_argument("path")
    t.add_argument("--out", required=True)
    r = commands.add_parser("refuel", help="rebuild the fuel of a saved scenario from its land cover")
    r.add_argument("out")
    args = parser.parse_args()
    if args.command == "refuel":
        scenario = Scenario.load(args.out)
        if scenario.landcover is None:
            parser.error(f"{args.out} has no land cover to rebuild the fuel from")
        scenario.fuel = to_fuel(scenario.landcover)
    else:
        scenario = fetch(args.lat, args.lon, *args.km) if args.command == "fetch" else import_tif(args.path)
    ny, nx = scenario.fuel.shape
    burnable = (scenario.fuel > 0).mean() * 100
    relief = 0.0 if scenario.elevation is None else float(np.ptp(scenario.elevation))
    print(f"{scenario.save(args.out)}: {nx} x {ny} cells, {burnable:.0f}% burnable, {relief:.0f} m relief")


if __name__ == "__main__":
    main()
