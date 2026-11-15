"""Radar örneklerini çizme.

TorNet görüntüleri kutupsal koordinattadır (açı x mesafe). Bunları düz bir
120x240 resim gibi çizmek şekilleri bozar, bu yüzden her pikseli radara göre
x/y kilometreye çeviriyoruz. Ağın kendisi yine kutupsal 120x240 ızgarayı görür.
"""
import matplotlib.pyplot as plt
import numpy as np

CMAPS = {"DBZ": "turbo", "VEL": "seismic"}
LIMITS = {"DBZ": (-20, 60), "VEL": (-40, 40)}
UNITS = {"DBZ": "dBZ", "VEL": "m/s"}


def _xy(ds):
    """Her pikselin radara göre doğu (x) ve kuzey (y) konumu, km cinsinden."""
    az = np.deg2rad(ds["azimuth"].values)        # kuzeyden saat yönünde derece
    rng = ds["range"].values / 1000.0            # km
    A, R = np.meshgrid(az, rng, indexing="ij")
    return R * np.sin(A), R * np.cos(A)          # x = doğu, y = kuzey


def plot_sample(ds, variables=("DBZ", "VEL"), frame=-1, sweep=0, axes=None, title=None):
    """Bir örneğin seçilen değişkenlerini yan yana, radara göre km cinsinden çizer."""
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
        ax.set_xlabel("radarın doğusu (km)")
        ax.set_ylabel("radarın kuzeyi (km)")
        ax.set_title(v)
    if title is None:
        a = ds.attrs
        ef = a.get("ef_number", -1)
        ef_txt = f", EF{int(ef)}" if ef >= 0 else ""
        title = f"{a.get('category')}{ef_txt} | {a.get('site_name')} | {str(ds['time'].values[frame])[:16]}"
    axes[0].figure.suptitle(title)
    return axes


def plot_frames(ds, variable="VEL", sweep=0):
    """Bir değişkenin 4 zaman adımının hepsini, her karenin etiketiyle birlikte çizer."""
    x, y = _xy(ds)
    n = ds.sizes["time"]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True, layout="constrained")
    vmin, vmax = LIMITS.get(variable, (None, None))
    for t, ax in enumerate(axes):
        data = ds[variable].values[t, :, :, sweep]
        ax.pcolormesh(x, y, data, cmap=CMAPS.get(variable, "viridis"), vmin=vmin, vmax=vmax, shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"{str(ds['time'].values[t])[11:16]}  etiket={int(ds['frame_labels'].values[t])}")
    fig.suptitle(f"{variable}, zamana göre | {ds.attrs.get('category')} | {ds.attrs.get('site_name')}")
    return axes
