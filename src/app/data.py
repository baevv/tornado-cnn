"""Arayüzün veri katmanı: işlenmiş radar dizileri (npy, mmap) ve örnek/fırtına tabloları.

Diskte (scripts/export_npy.py bir kez çalıştırılınca):
  <data_dir>/processed/npy/tornet_<yıl>_X.npy   uint8 (N, C, 120, 240), np.load(mmap_mode="r") ile açılır
  <data_dir>/processed/npy/samples.parquet      her örnek için: yıl, satır, dosya adı, katalog bilgisi, bölüm
  <data_dir>/processed/npy/meta.json            kanal adları

mmap sayesinde bütün yıllar belleğe alınmaz; yalnız gösterilen fırtınanın kareleri diskten okunur.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.helper import paths
from src.helper.tornet import CHANNEL_RANGES

SAMPLES_FILE = "samples.parquet"
META_FILE = "meta.json"

# Fırtına filtresi: kimlik → (simge, ipucu). Tanımlar episode_viewer.py ile aynı.
FILTERS = {
    "all": ("layers", "Tümü"),
    "tor": ("tornado", "Hortumlu"),
    "ef2": ("flame", "EF2+"),
    "fa": ("bell-off", "Yanlış alarm"),
    "nul": ("circle-dashed", "NUL"),
}
SPLITS = {"all": "Tümü", "train": "Eğitim", "val": "Doğrulama", "test": "Test"}


def npy_dir() -> Path:
    return paths.processed_dir() / "npy"


def x_path(year: int) -> Path:
    return npy_dir() / f"tornet_{year}_X.npy"


@dataclass
class Store:
    """Açılmış veri: örnek tablosu, fırtına tablosu ve yıl başına mmap dizileri."""
    samples: pd.DataFrame
    episodes: pd.DataFrame
    arrays: dict
    variables: list

    def images(self, rows: pd.DataFrame) -> np.ndarray:
        """Seçilen örneklerin uint8 görüntüleri, tablo sırasıyla: (F, C, 120, 240)."""
        return np.stack([np.asarray(self.arrays[int(y)][int(r)]) for y, r in zip(rows["year"], rows["row"])])

    def frames(self, episode_id) -> pd.DataFrame:
        """Bir fırtınanın kareleri, zaman sırasıyla."""
        idx = self.episodes.loc[episode_id, "rows"]
        return self.samples.loc[idx]


def open_store(root: Path | None = None) -> Store:
    """npy klasörünü açar. Klasör yoksa FileNotFoundError (arayüz kullanıcıya ne yapacağını söyler)."""
    root = Path(root) if root else npy_dir()
    samples_path = root / SAMPLES_FILE
    if not samples_path.exists():
        raise FileNotFoundError(str(samples_path))
    samples = pd.read_parquet(samples_path)
    meta = json.loads((root / META_FILE).read_text(encoding="utf-8"))
    arrays = {int(y): np.load(root / f"tornet_{int(y)}_X.npy", mmap_mode="r") for y in sorted(samples["year"].unique())}
    return Store(samples=samples, episodes=build_episodes(samples), arrays=arrays, variables=list(meta["variables"]))


def build_episodes(samples: pd.DataFrame) -> pd.DataFrame:
    """Örnekleri fırtına olayına (episode_id) göre toplar; filtreler bu tabloyu kullanır."""
    s = samples.sort_values(["start_time", "end_time", "filename"])
    g = s.groupby("episode_id", sort=False)
    ep = pd.DataFrame({
        "start_time": g["start_time"].min(),
        "n_frames": g.size(),
        "n_tor": g["label"].sum(),
        "n_wrn": g["category"].apply(lambda c: int((c == "WRN").sum())),
        "n_nul": g["category"].apply(lambda c: int((c == "NUL").sum())),
        "max_ef": g["ef"].max(),
        "splits": g["split"].apply(lambda x: frozenset(x)),
        "rows": pd.Series({k: list(v) for k, v in g.groups.items()}),
    })
    ep["has_tor"] = ep["n_tor"] > 0
    ep["has_wrn"] = ep["n_wrn"] > 0
    return ep.sort_values("start_time")


def filter_mask(episodes: pd.DataFrame, key: str) -> pd.Series:
    if key == "tor":
        return episodes["has_tor"]
    if key == "ef2":
        return episodes["max_ef"] >= 2
    if key == "fa":
        return ~episodes["has_tor"] & episodes["has_wrn"]
    if key == "nul":
        return ~episodes["has_tor"] & ~episodes["has_wrn"]
    return pd.Series(True, index=episodes.index)


def split_mask(episodes: pd.DataFrame, split: str) -> pd.Series:
    if split == "all":
        return pd.Series(True, index=episodes.index)
    return episodes["splits"].apply(lambda s: split in s)


def pool(episodes: pd.DataFrame, filt: str, split: str) -> pd.Index:
    """Filtre ve bölüme uyan fırtınaların kimlikleri, zaman sırasıyla."""
    return episodes.index[filter_mask(episodes, filt) & split_mask(episodes, split)]


def filter_counts(episodes: pd.DataFrame, split: str) -> dict:
    sm = split_mask(episodes, split)
    return {k: int((filter_mask(episodes, k) & sm).sum()) for k in FILTERS}


def physical(q: np.ndarray, variables) -> np.ndarray:
    """uint8 → fiziksel birim (float32, boş piksel NaN). q: (..., C, H, W)."""
    out = np.full(q.shape, np.nan, dtype=np.float32)
    for i, v in enumerate(variables):
        lo, hi = CHANNEL_RANGES[v]
        qi = q[..., i, :, :]
        ok = qi > 0
        out[..., i, :, :][ok] = (qi[ok].astype(np.float32) - 1.0) / 254.0 * (hi - lo) + lo
    return out


def rotation_cells(q: np.ndarray, variables) -> list:
    """Her karede en güçlü dönmenin (açı, mesafe) hücresi (eda.rotation_location); yoksa None."""
    if "VEL" not in variables:
        return [None] * len(q)
    try:
        from src.helper.eda import rotation_location
    except ImportError:                                   # scipy yoksa işaret konmaz
        return [None] * len(q)
    vi = list(variables).index("VEL")
    vel = physical(q, variables)[:, vi]
    out = []
    for v in vel:
        loc = rotation_location(v, center=True)
        out.append(list(loc) if loc is not None else None)
    return out
