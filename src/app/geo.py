"""Harita sekmesi için coğrafi yardımcılar (kararlar V1–V3).

Radar konumu dışarıdan bir istasyon listesi gerektirmeden katalogdan hesaplanır: catalog.csv'deki lat / lon pencerenin
merkezidir; pencere merkezinin radara göre açısı ve uzaklığı bilindiği için radar geriye doğru bulunur (düz Dünya
yaklaşımı, pencere ölçeğinde hata < 1 km):
    lat₀ = lat − r·cos θ / 111,32        lon₀ = lon − r·sin θ / (111,32 · cos lat)

Kutupsal kare (120 açı × 240 mesafe hücresi) coğrafi bir resme yeniden örneklenir; resim haritaya köşe koordinatlarıyla
yarı saydam katman olarak konur. Model kutupsal veriyle çalışır; haritadaki görüntü yalnız gösterim içindir.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd

KM_PER_DEG = 111.32
DAZ, DR = 0.5, 0.25                  # açı adımı (derece), mesafe adımı (km)
N_AZ, N_RNG = 120, 240
WIN_AZ, WIN_R = N_AZ * DAZ, N_RNG * DR   # 60°, 60 km


def window_center(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pencere merkezinin radara göre açısı (derece) ve uzaklığı (km). samples.rng_min km cinsindendir."""
    theta = samples["az_min"].to_numpy(dtype=float) + WIN_AZ / 2
    r = samples["rng_min"].to_numpy(dtype=float) + WIN_R / 2
    return theta, r


def radar_positions(samples: pd.DataFrame) -> pd.DataFrame:
    """Her istasyonun tahmini konumu (ortanca) ve tutarlılığı (km): site, lat, lon, spread_km, n, n_tor."""
    theta, r = window_center(samples)
    lat, lon = samples["lat"].to_numpy(dtype=float), samples["lon"].to_numpy(dtype=float)
    t = np.radians(theta)
    lat0 = lat - r * np.cos(t) / KM_PER_DEG
    lon0 = lon - r * np.sin(t) / (KM_PER_DEG * np.cos(np.radians(lat)))
    df = pd.DataFrame({"site": samples["site"].to_numpy(), "lat0": lat0, "lon0": lon0,
                       "tor": samples["label"].to_numpy() == 1})
    g = df.groupby("site")
    out = pd.DataFrame({"lat": g["lat0"].median(), "lon": g["lon0"].median(), "n": g.size(), "n_tor": g["tor"].sum()})
    dlat = (df["lat0"] - df["site"].map(out["lat"])) * KM_PER_DEG
    dlon = (df["lon0"] - df["site"].map(out["lon"])) * KM_PER_DEG * np.cos(np.radians(df["site"].map(out["lat"])))
    out["spread_km"] = np.hypot(dlat, dlon).groupby(df["site"]).median()
    return out.reset_index()


def _to_xy(lat, lon, lat0, lon0):
    """Coğrafi → radara göre km (x doğu, y kuzey)."""
    return (np.asarray(lon) - lon0) * KM_PER_DEG * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * KM_PER_DEG


def _to_ll(x, y, lat0, lon0):
    return lat0 + np.asarray(y) / KM_PER_DEG, lon0 + np.asarray(x) / (KM_PER_DEG * np.cos(np.radians(lat0)))


def frame_image(rgb: np.ndarray, az0: float, r0: float, lat0: float, lon0: float, size=360, alpha=0.8,
                valid: np.ndarray | None = None) -> dict:
    """Kutupsal RGB kareyi (120, 240, 3) coğrafi PNG'ye yeniden örnekler.

    valid (120, 240) bool verilirse yalnız o hücreler boyanır (boş pikseller saydam).
    Döner: {"source": data URI, "coordinates": [[lon, lat] × 4] (sol üst, sağ üst, sağ alt, sol alt)}.
    """
    # Pencerenin sınır noktalarından coğrafi kutu
    az = np.radians(az0 + np.linspace(0, WIN_AZ, 31))
    rr = r0 + np.linspace(0, WIN_R, 31)
    A, R = np.meshgrid(az, [r0, r0 + WIN_R])
    bx = np.r_[(R * np.sin(A)).ravel(), (rr * np.sin(az[0])), (rr * np.sin(az[-1]))]
    by = np.r_[(R * np.cos(A)).ravel(), (rr * np.cos(az[0])), (rr * np.cos(az[-1]))]
    x0, x1, y0, y1 = bx.min(), bx.max(), by.min(), by.max()
    # Kare şeklinde km ızgarası (en uzun kenar `size` piksel)
    span = max(x1 - x0, y1 - y0)
    nx, ny = max(2, int(size * (x1 - x0) / span)), max(2, int(size * (y1 - y0) / span))
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y1, y0, ny)                           # resmin üstü kuzey
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X, Y)
    a = (np.degrees(np.arctan2(X, Y)) - az0) % 360
    i = np.floor(a / DAZ).astype(int)
    j = np.floor((r - r0) / DR).astype(int)
    ok = (i >= 0) & (i < N_AZ) & (j >= 0) & (j < N_RNG)
    ic, jc = np.clip(i, 0, N_AZ - 1), np.clip(j, 0, N_RNG - 1)
    if valid is not None:
        ok &= valid[ic, jc]
    out = np.zeros((ny, nx, 4), dtype=np.uint8)
    out[..., :3] = rgb[ic, jc]
    out[..., 3] = np.where(ok, int(255 * alpha), 0)
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(out, "RGBA").save(buf, format="PNG")
    (la_t, la_b), (lo_l, lo_r) = _to_ll(np.array([x0, x1]), np.array([y1, y0]), lat0, lon0)
    corners = [[float(lo_l), float(la_t)], [float(lo_r), float(la_t)], [float(lo_r), float(la_b)], [float(lo_l), float(la_b)]]
    return {"source": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "coordinates": corners}


def site_metrics(samples: pd.DataFrame, p: np.ndarray, thr: float, min_tor=30) -> pd.DataFrame:
    """İstasyon başına başarım (V3): yalnız en az `min_tor` TOR karesi olan istasyonlar."""
    from src.helper.evaluate import summarize
    rows = []
    df = samples.assign(prob=p)
    for site, g in df.groupby("site"):
        y = g["label"].to_numpy()
        if y.sum() < min_tor or y.min() == y.max():
            continue
        s = summarize(y, g["prob"].to_numpy(), thr)
        tor_eps = int(g.loc[g["label"] == 1, "episode_id"].nunique())
        rows.append({"site": site, "n": len(g), "n_tor": int(y.sum()), "hortumlu fırtına": tor_eps,
                     "hortum oranı": float(y.mean()),
                     "PR-AUC": s["PR-AUC"], "ROC-AUC": s["ROC-AUC"], "CSI": s["CSI"], "POD": s["POD"], "SR": s["SR"]})
    return pd.DataFrame(rows)
