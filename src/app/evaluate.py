"""Bölüm bazlı skor ve ölçütler (Özet, Karşılaştırma, Hata Analizi ortak kullanır)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.app import cache

SPLIT_KEYS = {"val": "validation", "test": "test"}


def split_scores(info, store, split: str):
    """(örnek tablosu, olasılıklar) seçili bölüm için; skor yoksa (None, None)."""
    if store is None:
        return None, None
    p = cache.probs(info, store.samples)
    if p is None:
        return None, None
    m = (store.samples["split"] == split).to_numpy() & ~np.isnan(p)
    if not m.any():
        return None, None
    return store.samples[m], p[m]


def metrics_for(info, store, split: str) -> dict | None:
    """Ölçütler: önce preds.parquet'ten hesaplanır (seçili eşikte), yoksa results.json'daki kayıt."""
    s, p = split_scores(info, store, split)
    if s is not None and s["label"].nunique() > 1:
        from src.helper.train import metrics
        return metrics(s["label"].to_numpy(), p, info.threshold)
    return info.results.get(SPLIT_KEYS[split])


def kpis(m: dict | None) -> list:
    if not m:
        return []
    keys = [("ROC-AUC", "ROC-AUC"), ("PR-AUC", "PR-AUC"), ("F1", "F1"), ("Recall", "recall (yakalama)"), ("Precision", "precision (isabet)")]
    return [[k, f"{m[src]:.3f}"] for k, src in keys if src in m and m[src] == m[src]]


def no_scores_note(split: str) -> str:
    return (f"<div class='note'>Bu bölüm ({split}) için skor yok. Ya veride bu bölüm yok "
            "(<code>val</code> için veri klasöründe <code>splits.csv</code> gerekir) ya da skorlar hesaplanmamış: "
            "<code>python scripts/make_preds.py</code>.</div>")


def history_df(info) -> pd.DataFrame:
    return pd.DataFrame(info.history) if info.history else pd.DataFrame()
