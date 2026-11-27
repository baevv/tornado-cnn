"""Bölüm bazlı skor ve ölçütler (Özet, Karşılaştırma, Hata Analizi ortak kullanır).

Karşılaştırma sekmesi için ayrıca: seed ensemble'ları (O3), test tablosu (bootstrap %95 GA, temel modelin seed
ortalamasına göre Δ; M3), NC / RC / WC (S2). Hesaplar src/helper/evaluate.py'dedir; burada yalnız arayüz verisine
uyarlanır ve önbelleğe alınır.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from src.app import cache
from src.helper import evaluate as ev

SPLIT_KEYS = {"val": "validation", "test": "test"}
BASE = "03_baseline"          # Δ referansı: bu deneyin seed'lerinin ortalaması
N_BOOT = 500                  # arayüzde 500 örneklem (06 defteri 1000 kullanır)


# ---------------------------------------------------------------- ensemble'lar (O3)

def ensembles(models: list) -> list:
    """Birden fazla seed klasörü olan deneyler için sanal model: skor = seed skorlarının ortalaması.

    Eşik burada üyelerin eşik ortalamasıdır; with_ensembles doğrulama skorları varsa eşiği orada F1 ile yeniden seçer.
    """
    from src.helper.train import seed_group
    groups: dict = {}
    for m in models:
        if m.key.endswith("_hizli"):
            continue
        groups.setdefault(seed_group(m.key)[0], []).append(m)
    out = []
    for base, members in groups.items():
        if len(members) < 2 or not all(m.has_preds for m in members):
            continue
        head = next((m for m in members if m.key == base), members[0])
        out.append(replace(head, key=f"{base}_ens", label=f"{head.label.split(' · seed')[0]} · ensemble",
                           long_label=f"{head.long_label.split(' · seed')[0]} · ensemble ({len(members)} seed)",
                           threshold=float(np.mean([m.threshold for m in members])), params=head.params,
                           history=[], members=members))
    return out


def with_ensembles(models: list, store) -> list:
    """Modeller + ensemble'lar; ensemble eşiği doğrulama varsa orada seçilir."""
    ens = ensembles(models)
    if store is not None:
        for e in ens:
            s, p = split_scores(e, store, "val")
            if s is not None and s["label"].nunique() > 1:
                from src.helper.train import choose_threshold
                e.threshold = choose_threshold(s["label"].to_numpy(), p)
    return [*models, *ens]


# ---------------------------------------------------------------- skorlar ve ölçütler

def _probs(info, store) -> np.ndarray | None:
    members = getattr(info, "members", None)
    if members:
        ps = [cache.probs(m, store.samples) for m in members]
        if any(p is None for p in ps):
            return None
        return np.mean(np.stack(ps), axis=0)
    return cache.probs(info, store.samples)


def split_scores(info, store, split: str):
    """(örnek tablosu, olasılıklar) seçili bölüm için; skor yoksa (None, None)."""
    if store is None:
        return None, None
    p = _probs(info, store)
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


def meta(samples: pd.DataFrame) -> pd.DataFrame:
    """Arayüz örnek tablosu → src/helper/evaluate'in beklediği sütunlar (ef_number, rng_min metre)."""
    return pd.DataFrame({"label": samples["label"].to_numpy(), "category": samples["category"].to_numpy(),
                         "ef_number": samples["ef"].to_numpy(), "episode_id": samples["episode_id"].to_numpy(),
                         "rng_min": samples["rng_min"].to_numpy(dtype=float) * 1000.0})


def _stamp(info) -> tuple:
    members = getattr(info, "members", None) or [info]
    return tuple((m.key, m.preds_path.stat().st_mtime if m.has_preds else 0.0) for m in members)


@st.cache_data(show_spinner="Bootstrap güven aralıkları hesaplanıyor…", max_entries=8)
def _boot_table(keys: tuple, stamps: tuple, split: str, n_boot: int, _infos: list, _store) -> pd.DataFrame:
    by_key = {m.key: m for m in _infos}
    s0 = None
    data = {}
    for k in keys:
        s, p = split_scores(by_key[k], _store, split)
        if s is None:
            continue
        s0 = s if s0 is None else s0
        data[k] = (s, p)
    if not data:
        return pd.DataFrame()
    # Bütün modeller aynı kareler üzerinde (skor eksikliği farklıysa kesişim)
    common = set.intersection(*(set(s["filename"]) for s, _ in data.values()))
    aligned = {}
    for k, (s, p) in data.items():
        m = s["filename"].isin(common).to_numpy()
        order = np.argsort(s["filename"].to_numpy()[m], kind="stable")
        aligned[k] = (s[m].iloc[order], p[m][order])
    ref_s = next(iter(aligned.values()))[0]
    y = ref_s["label"].to_numpy()
    idx = ev.bootstrap_indices(ref_s["episode_id"].to_numpy(), n_boot=n_boot, seed=0)
    boots = {k: ev.bootstrap(y, p, by_key[k].threshold, idx) for k, (_, p) in aligned.items()}
    summ = {k: ev.summarize(y, p, by_key[k].threshold) for k, (_, p) in aligned.items()}

    from src.helper.train import seed_group
    ref = [k for k in aligned if seed_group(k)[0] == BASE and not k.endswith("_ens")]
    ref_pr = float(np.mean([summ[k]["PR-AUC"] for k in ref])) if ref else np.nan
    sd = float(np.std([summ[k]["PR-AUC"] for k in ref], ddof=1)) if len(ref) > 1 else np.nan
    ref_boot = np.mean([boots[k]["PR-AUC"].to_numpy() for k in ref], axis=0) if ref else None

    rows = []
    for k in aligned:
        s, b, info = summ[k], boots[k], by_key[k]
        row = {"model": info.label, "eşik": round(info.threshold, 3),
               "PR-AUC": round(s["PR-AUC"], 3), "PR-AUC %95 GA": _ci(b["PR-AUC"]),
               "ROC-AUC": round(s["ROC-AUC"], 3), "ROC-AUC %95 GA": _ci(b["ROC-AUC"]),
               "POD": round(s["POD"], 3), "SR": round(s["SR"], 3),
               "CSI": round(s["CSI"], 3), "CSI %95 GA": _ci(b["CSI"]), "Bias": round(s["Bias"], 2)}
        if ref_boot is not None:
            d = s["PR-AUC"] - ref_pr
            lo, hi = ev.ci(b["PR-AUC"].to_numpy() - ref_boot)
            row["Δ PR-AUC"] = round(d, 3)
            row["Δ %95 GA"] = f"[{lo:+.3f}, {hi:+.3f}]"
            row["Δ/σ"] = round(d / sd, 1) if sd == sd and sd > 0 else None
            if seed_group(k)[0] == BASE or k == f"{BASE}_ens":
                row["yorum"] = "temel model"
            elif sd == sd and not (lo <= 0 <= hi) and abs(d) > 2 * sd:
                row["yorum"] = "fark"
            elif sd == sd and abs(d) > sd:
                row["yorum"] = "eğilim"
            else:
                row["yorum"] = "gürültü"
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    out.attrs.update(n=len(y), storms=int(ref_s["episode_id"].nunique()), base_rate=float(y.mean()),
                     ref=ref_pr, sd=sd, n_ref=len(ref))
    return out


def _ci(samples) -> str:
    lo, hi = ev.ci(samples)
    return f"[{lo:.3f}, {hi:.3f}]"


def boot_table(models: list, store, split: str, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Seçili modeller + temel modelin bütün seed'leri (Δ referansı için) ile test tablosu."""
    from src.helper.train import seed_group
    keys = [m.key for m in models]
    pool = {m.key: m for m in models}
    for m in cache.models():
        if seed_group(m.key)[0] == BASE and m.key not in pool and m.has_preds:
            pool[m.key] = m
            keys.append(m.key)
    infos = list(pool.values())
    return _boot_table(tuple(keys), tuple(_stamp(m) for m in infos), split, n_boot, infos, store)


def views_table(models: list, store, split: str) -> pd.DataFrame:
    """NC / RC / WC: PR-AUC, ROC-AUC ve CSI (karar S2)."""
    rows = []
    for m in models:
        s, p = split_scores(m, store, split)
        if s is None or s["label"].nunique() < 2:
            continue
        t = ev.view_table(meta(s), p, m.threshold)
        for v, r in t.iterrows():
            rows.append({"model": m.label, "görünüm": v, "PR-AUC": r["PR-AUC"], "ROC-AUC": r["ROC-AUC"], "CSI": r["CSI"],
                         "hortum oranı": r["hortum oranı"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index="model", columns="görünüm", values=["PR-AUC", "ROC-AUC", "CSI"], sort=False)
    wide.columns = [f"{a} · {b}" for a, b in wide.columns]
    wide = wide[[f"{a} · {b}" for a in ("PR-AUC", "ROC-AUC", "CSI") for b in ev.VIEWS]].round(3)
    wide.attrs["rates"] = df.drop_duplicates("görünüm").set_index("görünüm")["hortum oranı"].to_dict()
    return wide.reset_index()


# ---------------------------------------------------------------- sekmeler arası küçük yardımcılar

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
