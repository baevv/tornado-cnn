"""Plotting radar samples.

TorNet images are in polar coordinates (azimuth x range). Drawing them as a
plain 120x240 image distorts shapes, so we convert each pixel to x/y kilometres
from the radar. The network itself still sees the polar 120x240 grid.
"""
import matplotlib.pyplot as plt
import numpy as np

CMAPS = {"DBZ": "turbo", "VEL": "seismic"}
LIMITS = {"DBZ": (-20, 60), "VEL": (-40, 40)}
UNITS = {"DBZ": "dBZ", "VEL": "m/s"}


def _xy(ds):
    az = np.deg2rad(ds["azimuth"].values)        # degrees clockwise from north
    rng = ds["range"].values / 1000.0            # km
    A, R = np.meshgrid(az, rng, indexing="ij")
    return R * np.sin(A), R * np.cos(A)          # x = east, y = north


def plot_sample(ds, variables=("DBZ", "VEL"), frame=-1, sweep=0, axes=None, title=None):
    """Draw selected variables of one sample side by side, in km from the radar."""
    x, y = _xy(ds)
    if axes is None:
        _, axes = plt.subplots(1, len(variables), figsize=(5.5 * len(variables), 4.5), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, variables):
        vmin, vmax = LIMITS.get(v, (None, None))
        data = ds[v].values[frame, :, :, sweep]
        m = ax.pcolormesh(x, y, data, cmap=CMAPS.get(v, "viridis"), vmin=vmin, vmax=vmax, shading="gouraud")
        plt.colorbar(m, ax=ax, label=UNITS.get(v, v))
        ax.set_aspect("equal")
        ax.set_xlabel("km east of radar")
        ax.set_ylabel("km north of radar")
        ax.set_title(v)
    if title is None:
        a = ds.attrs
        ef = a.get("ef_number", -1)
        ef_txt = f", EF{int(ef)}" if ef >= 0 else ""
        title = f"{a.get('category')}{ef_txt} | {a.get('site_name')} | {str(ds['time'].values[frame])[:16]}"
    axes[0].figure.suptitle(title)
    return axes


def plot_frames(ds, variable="VEL", sweep=0):
    """All 4 time frames of one variable, with their per-frame labels."""
    x, y = _xy(ds)
    n = ds.sizes["time"]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True, layout="constrained")
    vmin, vmax = LIMITS.get(variable, (None, None))
    for t, ax in enumerate(axes):
        data = ds[variable].values[t, :, :, sweep]
        ax.pcolormesh(x, y, data, cmap=CMAPS.get(variable, "viridis"), vmin=vmin, vmax=vmax, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"{str(ds['time'].values[t])[11:16]}  label={int(ds['frame_labels'].values[t])}")
    fig.suptitle(f"{variable} over time | {ds.attrs.get('category')} | {ds.attrs.get('site_name')}")
    return axes
