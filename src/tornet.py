"""TorNet: download, catalog, reading single samples and preprocessing.

Facts about the dataset (checked against the v1.1 files):
  * One sample = one NetCDF (.nc) file.
  * Each radar variable has shape (time=4, azimuth=120, range=240, sweep=2):
    4 frames 5 minutes apart, 60 degrees x 60 km at 0.5 deg / 250 m resolution,
    sweeps at 0.5 and 0.9 degrees elevation.
  * Missing measurements are NaN.
  * `frame_labels` gives a label per frame; the sample label is the LAST frame,
    which is 1 exactly when category == "TOR".
  * catalog.csv lists all 203,133 samples of all years.

Source: Veillette et al. (2025), "A Benchmark Dataset for Tornado Detection and
Prediction using Full-Resolution Polarimetric Weather Radar Data",
AIES 4(1). Data: Zenodo, CC BY 4.0.
"""
import io
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Zenodo record of each year (dataset version 1.1). The 2013 record also has catalog.csv.
ZENODO_RECORDS = {
    2013: 12636522, 2014: 12637032, 2015: 12655151, 2016: 12655179, 2017: 12655183,
    2018: 12655187, 2019: 12655716, 2020: 12655717, 2021: 12655718, 2022: 12655719,
}

# Fixed physical ranges used for scaling (from the TorNet reference code).
# Fixed ranges, not statistics of our data, so no information leaks from test to train.
CHANNEL_RANGES = {
    "DBZ": (-20.0, 60.0),    # reflectivity, dBZ
    "VEL": (-60.0, 60.0),    # radial velocity, m/s
    "KDP": (-2.0, 5.0),
    "RHOHV": (0.2, 1.04),
    "ZDR": (-1.0, 8.0),
    "WIDTH": (0.0, 9.0),
}

CATEGORY_NAMES = {
    "TOR": "confirmed tornado",
    "WRN": "tornado warning, no tornado",
    "NUL": "random storm, no tornado",
}


def file_url(year: int, filename: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORDS[year]}/files/{filename}?download=1"


def tar_url(year: int) -> str:
    return file_url(year, f"tornet_{year}.tar.gz")


def download(url: str, dest: Path) -> Path:
    """Download with resume support (curl -C -). Skips if the file is already complete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "-C", "-", "--fail", "--retry", "5", "-o", str(dest), url], check=True)
    else:  # fallback without resume
        urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------- catalog

def load_catalog(path) -> pd.DataFrame:
    """Read catalog.csv and add `year` and the binary `label` (TOR=1, WRN/NUL=0)."""
    cat = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    cat["year"] = cat["start_time"].dt.year
    cat["label"] = (cat["category"] == "TOR").astype(np.uint8)
    return cat


def get_catalog(data_dir) -> pd.DataFrame:
    path = Path(data_dir) / "catalog.csv"
    if not path.exists():
        download(file_url(2013, "catalog.csv"), path)
    return load_catalog(path)


# ---------------------------------------------------------------- one sample

def open_sample(source):
    """Open one .nc sample from a path or from raw bytes."""
    import xarray as xr
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return xr.load_dataset(source, engine="h5netcdf")


def to_uint8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale physical values to 1..255 and keep 0 for 'missing' (NaN).

    1 byte per pixel instead of 4 makes the processed data 4x smaller.
    Resolution: 80 dBZ / 254 steps = 0.31 dBZ, 120 m/s / 254 = 0.47 m/s.
    """
    out = np.zeros(x.shape, dtype=np.uint8)
    ok = np.isfinite(x)
    scaled = (np.clip(x[ok], lo, hi) - lo) / (hi - lo)
    out[ok] = np.round(scaled * 254).astype(np.uint8) + 1
    return out


def from_uint8(q: np.ndarray, variables) -> np.ndarray:
    """Turn stored bytes back into network input in [-1, 1].

    Missing pixels (0) are filled with a physically sensible value instead of
    blindly using 0 after scaling:
      * DBZ: -1 (= the lowest reflectivity, i.e. "no echo")
      * VEL:  0 (= zero radial velocity)
    """
    x = (q.astype(np.float32) - 1.0) / 254.0 * 2.0 - 1.0   # 1..255 -> -1..1
    missing = q == 0
    fill_values = {"DBZ": -1.0, "VEL": 0.0}
    ch_axis = -3  # (..., C, H, W)
    for i, v in enumerate(variables):
        sl = [slice(None)] * x.ndim
        sl[ch_axis] = i
        sl = tuple(sl)
        x[sl][missing[sl]] = fill_values.get(v, -1.0)
    return x


def preprocess_sample(ds, variables=("DBZ", "VEL"), sweep=0, frame=-1) -> np.ndarray:
    """Pick channels, sweep and frame -> uint8 array of shape (C, 120, 240)."""
    chans = []
    for v in variables:
        lo, hi = CHANNEL_RANGES[v]
        arr = ds[v].values[frame, :, :, sweep]  # (azimuth, range)
        chans.append(to_uint8(arr, lo, hi))
    return np.stack(chans)


def preprocess_bytes(raw: bytes, variables=("DBZ", "VEL"), sweep=0, frame=-1):
    """Same result as preprocess_sample(open_sample(raw)), ~10x faster.

    Reads only the needed slices with h5py instead of loading the whole file
    with xarray. Used for processing whole years. Returns (uint8 array, label).
    """
    import h5py
    with h5py.File(io.BytesIO(raw), "r") as h:
        chans = []
        for v in variables:
            d = h[v]
            arr = d[frame, :, :, sweep].astype(np.float32)
            fill = d.attrs.get("_FillValue")
            if fill is not None:
                arr[arr == fill[0]] = np.nan
            chans.append(to_uint8(arr, *CHANNEL_RANGES[v]))
        label = int(h["frame_labels"][frame])
    return np.stack(chans), label


# ---------------------------------------------------------------- whole year

def iter_tar_samples(tar_path_or_url):
    """Yield (relative_name, bytes) for every .nc in a TorNet tar.gz, without extracting to disk."""
    src = str(tar_path_or_url)
    fileobj = urllib.request.urlopen(src) if src.startswith("http") else open(src, "rb")
    with fileobj, tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".nc"):
                yield member.name, tar.extractfile(member).read()


def sample_from_tar(tar_path_or_url, per_category=None, max_files=5000):
    """Grab a few samples of each category by reading the start of an archive.

    Reading stops as soon as enough samples are found, so only a small part of
    a 15 GB archive is downloaded. Returns {filename: xarray.Dataset}.
    """
    import os
    per_category = per_category or {"TOR": 2, "WRN": 1, "NUL": 1}
    found, counts = {}, {k: 0 for k in per_category}
    for i, (name, raw) in enumerate(iter_tar_samples(tar_path_or_url)):
        cat = os.path.basename(name)[:3]
        if counts.get(cat, 0) < per_category.get(cat, 0):
            found[name] = open_sample(raw)
            counts[cat] += 1
        if all(counts[k] >= v for k, v in per_category.items()) or i + 1 >= max_files:
            break
    return found


def process_year(year, tar_path_or_url, out_dir, variables=("DBZ", "VEL"), sweep=0, frame=-1,
                 catalog=None, progress=True) -> Path:
    """Turn one year's archive into one compressed .npz with X, y and metadata.

    Output: <out_dir>/tornet_<year>.npz with
      X        uint8 (N, C, 120, 240)   - see to_uint8 / from_uint8
      y        uint8 (N,)               - 1 = tornado
      filename str   (N,)               - joins with catalog.csv for split/episode/EF info
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tornet_{year}.npz"
    if out_path.exists():
        print(f"{out_path.name} already exists, skipping.")
        return out_path

    expected = None
    if catalog is not None:
        expected = int((catalog["year"] == year).sum())

    X, y, names = [], [], []
    it = iter_tar_samples(tar_path_or_url)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, total=expected, desc=f"TorNet {year}", unit="file")
    for name, raw in it:
        x, label = preprocess_bytes(raw, variables, sweep, frame)
        X.append(x)
        y.append(label)
        names.append(name)

    X = np.stack(X)
    y = np.asarray(y, dtype=np.uint8)
    names = np.asarray(names)
    tmp = out_path.with_name(out_path.stem + ".partial.npz")
    np.savez_compressed(tmp, X=X, y=y, filename=names, variables=np.asarray(variables))
    tmp.rename(out_path)  # only a finished file gets the final name
    print(f"Saved {out_path} | {len(y)} samples, {int(y.sum())} tornado | "
          f"{out_path.stat().st_size / 1e9:.2f} GB")
    return out_path
