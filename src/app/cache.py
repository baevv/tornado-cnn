"""Streamlit önbellekleri: veri, modeller, skorlar, canlı tahmin ve Grad-CAM.

Önbellek anahtarlarına dosya değişiklik zamanları girer; model yeniden eğitilince ya da
preds.parquet yenilenince arayüz yeniden başlatmadan güncel veriyi kullanır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.app import data as D
from src.app import models as M


def _mtime(p) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Veri açılıyor…")
def _store(root: str, stamp: float):
    return D.open_store()


def store():
    """Açılmış veri ya da (veri yoksa) None ve beklenen klasör."""
    root = D.npy_dir()
    try:
        return _store(str(root), _mtime(root / D.SAMPLES_FILE)), root
    except FileNotFoundError:
        return None, root


@st.cache_resource
def _discover(root: str, stamp: tuple):
    return M.discover()


def models() -> list:
    root = M.models_root()
    stamp = tuple(sorted((p.name, _mtime(p / "results.json"), _mtime(p / "best.pt"), _mtime(p / M.PREDS_FILE))
                         for p in root.iterdir() if p.is_dir())) if root.exists() else ()
    return _discover(str(root), stamp)


@st.cache_data(show_spinner=False, max_entries=16)
def _aligned(path: str, stamp: float, filenames: tuple | None, n: int, _samples: pd.DataFrame):
    s = pd.read_parquet(path, columns=["filename", "prob"]).set_index("filename")["prob"]
    return _samples["filename"].map(s).to_numpy(dtype=np.float32)


def probs(info, samples: pd.DataFrame) -> np.ndarray | None:
    """samples sırasıyla hortum olasılıkları (skoru olmayan örnekte NaN); preds.parquet yoksa None."""
    if not info.has_preds:
        return None
    return _aligned(str(info.preds_path), _mtime(info.preds_path), None, len(samples), samples)


@st.cache_data(show_spinner=False, max_entries=32)
def strip_codes(model_key: str, thr: float, stamp: float, n: int, _samples: pd.DataFrame, _p: np.ndarray) -> pd.Series:
    """Her fırtına için seçili modelin sonucu: p yakaladı, n kaçırdı, f yanlış alarm, t doğru ret, x skor yok."""
    df = pd.DataFrame({"ep": _samples["episode_id"].to_numpy(), "y": _samples["label"].to_numpy(), "p": _p})
    df["fire"] = df["p"] >= thr
    df["hit"] = df["fire"] & (df["y"] == 1)
    df["has"] = df["p"].notna()
    g = df.groupby("ep").agg(tor=("y", "max"), hit=("hit", "any"), fire=("fire", "any"), has=("has", "any"))
    code = np.where(g["tor"] > 0, np.where(g["hit"], "p", "n"), np.where(g["fire"], "f", "t"))
    return pd.Series(np.where(g["has"], code, "x"), index=g.index)


@st.cache_resource(show_spinner=False, max_entries=4)
def net(model_key: str, stamp: float):
    info = next(m for m in models() if m.key == model_key)
    return M.load_model(info)


@st.cache_data(show_spinner=False, max_entries=64)
def live_scores(model_key: str, stamp: float, ep_id: int, _q: np.ndarray, variables: tuple, _r0=None) -> list:
    """preds.parquet yoksa fırtınanın karelerini canlı skorlar. _r0: kare başına rng_min, km (CoordConv)."""
    return M.predict_images(net(model_key, stamp), _q, list(variables), r0_km=_r0).tolist()


@st.cache_data(show_spinner=False, max_entries=48)
def cam(model_key: str, stamp: float, ep_id, _q: np.ndarray, variables: tuple, _r0=None) -> np.ndarray:
    """Grad-CAM, uint8 (F, H, W). _r0: kare başına rng_min, km (CoordConv)."""
    out = []
    n = net(model_key, stamp)
    for i in range(0, len(_q), 16):
        r = None if _r0 is None else _r0[i:i + 16]
        out.append(M.gradcam(n, _q[i:i + 16], list(variables), r0_km=r))
    return np.round(np.concatenate(out) * 255).astype(np.uint8)


@st.cache_data(show_spinner=False, max_entries=64)
def rotation(ep_id, stamp: float, _q: np.ndarray, variables: tuple) -> list:
    return D.rotation_cells(_q, list(variables))


@st.cache_data(show_spinner="Açıklama hesaplanıyor…", max_entries=48)
def explain(model_key: str, stamp: float, filename: str, _q: np.ndarray, variables: tuple, rot, az0: float, r0: float) -> dict:
    """Tek kare için bütün açıklamalar (src/helper/xai.explain_frame); kare dosya adıyla önbelleğe alınır."""
    from src.helper.xai import explain_frame
    return explain_frame(net(model_key, stamp), _q, list(variables), rot_cell=None if rot is None else tuple(rot),
                         az0=az0, r0=r0)
