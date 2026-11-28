"""Ana sayfa: proje sunumu. Slaytlar ve animasyonlar bileşendedir (src/app/frontend, "deck" bölümü);
burada yalnız slaytlardaki sayılar hesaplanır. Hiçbir sayı elle yazılmaz: veri (samples.parquet) ve model
klasörleri (results.json, preds.parquet) değişince slaytlar da değişir. Anlatım kronolojiktir: model sonuçları
ancak eğitimin anlatıldığı slaytlardan sonra görünür (kapakta yalnız veri). Konuşma metni: sunum_notlari.md.
"""
import base64
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.app import cache, geo
from src.app import data as D
from src.app import evaluate as E
from src.app import models as M
from src.helper import paths
from src.helper.tornet import CHANNEL_RANGES

GRID_DEG = 0.3            # veri haritası: örnekler bu boyda enlem/boylam hücrelerine toplanır (tarayıcıya ~10 bin hücre)
STORM_MAX_FRAMES = 12     # bölüm 3'te oynatılan fırtına dizisinin en fazla kare sayısı
STORM_GAP_MIN = 30        # dizideki iki örnek arasında en fazla bu kadar dakika (aynı olay)
STORM_SHIFT_KM = 20       # dizideki pencereler en fazla bu kadar kaydıysa aynı fırtına sayılır


def defaults(ss, models):
    pass


def handle(ev, ss):
    pass


def _kapak(store) -> dict:
    out = {"n": None, "n_tor": None, "pct": None, "years": []}
    if store is not None:
        s = store.samples
        out.update(n=int(len(s)), n_tor=int(s["label"].sum()), pct=float(s["label"].mean()),
                   years=sorted(int(y) for y in s["year"].unique()))
    return out


@st.cache_data(show_spinner=False, max_entries=2)
def _catalog(path: str, stamp: float) -> dict:
    """Bütün TorNet kataloğu (2013–2022): sayılar, yıllar ve haritası için hücre toplamları."""
    c = pd.read_csv(path, usecols=["start_time", "lat", "lon", "category", "site_name"])
    year = pd.to_datetime(c["start_time"]).dt.year
    tor = (c["category"] == "TOR").to_numpy()
    cats = c["category"].value_counts()
    by_year = pd.DataFrame({"y": year, "t": tor}).groupby("y")["t"].agg(["size", "sum"])
    cell = pd.DataFrame({"i": np.floor(c["lat"] / GRID_DEG).astype(int), "j": np.floor(c["lon"] / GRID_DEG).astype(int),
                         "t": tor}).groupby(["i", "j"])["t"].agg(["size", "sum"]).reset_index()
    return {
        "n": int(len(c)), "tor": int(cats.get("TOR", 0)), "wrn": int(cats.get("WRN", 0)), "nul": int(cats.get("NUL", 0)),
        "years": [int(year.min()), int(year.max())], "n_sites": int(c["site_name"].nunique()),
        "by_year": [[int(y), int(r["size"]), int(r["sum"])] for y, r in by_year.iterrows()],
        "cells": {"lat": ((cell["i"] + .5) * GRID_DEG).round(2).tolist(), "lon": ((cell["j"] + .5) * GRID_DEG).round(2).tolist(),
                  "n": cell["size"].astype(int).tolist(), "t": cell["sum"].astype(int).tolist()},
    }


def _veri(store) -> dict:
    """Bölüm 2: bütün veri seti (katalog) ve bizim kullandığımız yıllar (işlenmiş örnek tablosu)."""
    cat_path = Path(paths.data_dir()) / "catalog.csv"
    out = {"all": _catalog(str(cat_path), cat_path.stat().st_mtime) if cat_path.exists() else None, "used": None, "sites": []}
    if store is not None:
        s = store.samples
        cats = s["category"].value_counts()
        ef = s.loc[s["category"] == "TOR", "ef"].value_counts()
        out["used"] = {"n": int(len(s)), "tor": int(cats.get("TOR", 0)), "wrn": int(cats.get("WRN", 0)), "nul": int(cats.get("NUL", 0)),
                       "years": sorted(int(y) for y in s["year"].unique()),
                       "ef": [int(ef.get(k, 0)) for k in range(6)]}
        r = geo.radar_positions(s)
        out["sites"] = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(r["lat"], r["lon"])]
    return out


def _window_center(rows: pd.DataFrame) -> np.ndarray:
    """Pencere merkezleri, radara göre km (x doğu, y kuzey)."""
    th, r = geo.window_center(rows)
    return np.c_[r * np.sin(np.radians(th)), r * np.cos(np.radians(th))]


def _sequence(store, best: int) -> list:
    """En iyi karenin çevresinde aynı olayın kareleri: aynı istasyon, zamanda ardışık, pencere fazla kaymamış."""
    s = store.samples
    b = s.loc[best]
    rows = s.loc[store.episodes.loc[b["episode_id"], "rows"]]
    rows = rows[rows["site"] == b["site"]].sort_values("end_time")
    # Aynı anda iki pencere (iki hortum) olabilir: her zamandan en iyi kareye en yakın pencere kalır
    d = np.hypot(*(_window_center(rows) - _window_center(s.loc[[best]])[0]).T)
    rows = (rows.assign(_d=d, _o=rows.index != best).sort_values(["end_time", "_o", "_d"])
            .drop_duplicates("end_time").drop(columns=["_d", "_o"]))
    c = _window_center(rows)
    rows = rows[np.hypot(*(c - _window_center(s.loc[[best]])[0]).T) <= STORM_SHIFT_KM]
    t = rows["end_time"].to_numpy()
    k = int(np.flatnonzero(rows.index.to_numpy() == best)[0])
    lo = hi = k                                        # zaman boşluğu olmayan en uzun parça
    gap = np.timedelta64(STORM_GAP_MIN, "m")
    while lo > 0 and t[lo] - t[lo - 1] <= gap:
        lo -= 1
    while hi < len(t) - 1 and t[hi + 1] - t[hi] <= gap:
        hi += 1
    idx = list(rows.index[lo:hi + 1])
    if len(idx) > STORM_MAX_FRAMES:                    # ilk hortum karesinden biraz önce başlayan bir pencere
        cats = s.loc[idx, "category"].tolist()
        first = cats.index("TOR") if "TOR" in cats else idx.index(best)
        start = max(0, min(first - 3, idx.index(best), len(idx) - STORM_MAX_FRAMES))
        idx = idx[start:start + STORM_MAX_FRAMES]
    return idx


@st.cache_data(show_spinner="Sunum için örnek fırtına seçiliyor…", max_entries=2)
def _pick_storm(stamp: float, _store) -> dict:
    """Bölüm 3'ün örnek fırtınası (otomatik): EF2+ hortum kareleri arasında merkezde dönmesi en güçlü olan
    (01 defteri C2 ile aynı ölçü, eda.rotation_strength). En az 5 karelik dizisi olan ve radara yakın
    (5–120 km) olanlar önce gelir. TorNet'te hortumlu olayların kareleri baştan sona TOR'dur; dizi, hortumun
    yerdeki seyridir."""
    from src.helper.eda import rotation_strength
    s = _store.samples
    cand = s[(s["category"] == "TOR") & (s["ef"] >= 2)]
    if cand.empty:
        cand = s[s["category"] == "TOR"]
    if cand.empty:
        return {}
    vi = _store.variables.index("VEL")
    strength = []
    for k in range(0, len(cand), 256):
        ph = D.physical(_store.images(cand.iloc[k:k + 256]), _store.variables)
        strength += [rotation_strength(v[vi]) for v in ph]
    cand = cand.assign(rot=np.nan_to_num(strength, nan=-1.0)).sort_values("rot", ascending=False)
    best, best_key = None, None
    for i, r in cand.head(60).iterrows():
        seq = _sequence(_store, i)
        key = (len(seq) >= 5, 5 <= r["rng_min"] <= 120, r["rot"])
        if best_key is None or key > best_key:
            best, best_key = (i, seq), key
    i, seq = best
    return {"best": int(i), "seq": [int(x) for x in seq], "rot": float(cand.loc[i, "rot"])}


def _storm(store) -> dict | None:
    """Bölüm 3: seçilen fırtınanın kareleri (uint8, gezgin ile aynı biçim) ve her karede en güçlü dönme."""
    if store is None or "VEL" not in store.variables:
        return None
    pick = _pick_storm(cache._mtime(D.npy_dir() / D.SAMPLES_FILE), store)
    if not pick:
        return None
    rows = store.samples.loc[pick["seq"]]
    q = store.images(rows)
    variables = list(store.variables)
    frames = [{"time": f"{t:%H:%M}", "date": f"{t:%Y-%m-%d}", "cat": c, "ef": int(e), "az0": float(a), "r0": float(r)}
              for t, c, e, a, r in zip(rows["end_time"], rows["category"], rows["ef"], rows["az_min"], rows["rng_min"])]
    return {
        "key": f"{pick['best']}:{len(rows)}",
        "shape": list(q.shape), "variables": variables,
        "x": base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii"),
        "ranges": {v: list(CHANNEL_RANGES[v]) for v in variables},
        "daz": geo.DAZ, "dr": geo.DR,
        "frames": frames, "rot": D.rotation_cells(q, variables),
        "best": pick["seq"].index(pick["best"]), "rot_ms": round(pick["rot"], 1),
        "site": str(rows["site"].iloc[0]), "episode_id": str(rows["episode_id"].iloc[0]),
    }


class _Table(HTMLParser):
    """Defter çıktısındaki ilk pandas tablosu (text/html) → satırlar; her satır hücre metinleri (th + td)."""

    def __init__(self):
        super().__init__()
        self.rows, self.head, self._row, self._cell, self._in, self._part, self._done = [], [], None, None, False, None, False

    def handle_starttag(self, tag, attrs):
        if self._done:
            return
        if tag == "table" and "dataframe" in (dict(attrs).get("class") or ""):
            self._in = True
        elif self._in and tag in ("thead", "tbody"):
            self._part = tag
        elif self._in and tag == "tr":
            self._row = []
        elif self._in and tag in ("th", "td") and self._row is not None:
            self._cell = ""

    def handle_endtag(self, tag):
        if not self._in:
            return
        if tag in ("th", "td") and self._cell is not None:
            self._row.append(self._cell.strip()); self._cell = None
        elif tag == "tr" and self._row is not None:
            (self.head if self._part == "thead" else self.rows).append(self._row); self._row = None
        elif tag == "table":
            self._in, self._done = False, True

    def handle_data(self, data):
        if self._cell is not None:
            self._cell += data


def _num(t: str):
    m = re.search(r"-?\d+(?:[.,]\d+)?", t or "")
    return float(m.group(0).replace(",", ".")) if m else None


def _nb_cell(nb: dict, needle: str):
    """Kodunda `needle` geçen ilk hücrenin (tablo başlığı, satırlar, akış metni)."""
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code" and needle in "".join(c.get("source", [])):
            text, table = "", None
            for o in c.get("outputs", []):
                if o.get("output_type") == "stream":
                    text += "".join(o.get("text", []))
                html = o.get("data", {}).get("text/html")
                if html and table is None:
                    t = _Table(); t.feed("".join(html)); table = t
            return table, text
    return None, ""


@st.cache_data(show_spinner=False, max_entries=2)
def _eda_nb(path: str, stamp: float) -> dict:
    """Bölüm 4: 01 defterinin D bölümündeki tablolar, defterin kayıtlı çıktılarından okunur (ham veri gerekmez).
    Defter yeniden çalıştırılıp kaydedilince slaytlar da değişir. Hücreler kodlarından tanınır."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    t, _ = _nb_cell(nb, "separability(feat, cols0)")             # D3: tek değişkenli AUC (ayırt etme gücü)
    if t and t.rows:
        out["d3"] = [{"v": r[0], "tor": _num(r[2]), "fa": _num(r[4])} for r in t.rows if len(r) >= 5]
    t, _ = _nb_cell(nb, "cross_val_score")                       # D3b: ölçüm grupları birlikte (lojistik regresyon)
    if t and t.rows:
        out["d3b"] = [{"name": r[0], "k": _num(r[1]), "lr": _num(r[2]), "rf": _num(r[3])} for r in t.rows if len(r) >= 4]
    t, _ = _nb_cell(nb, "sweeps = pd.DataFrame")                 # D4: iki tarama açısı
    if t and t.rows:
        out["d4"] = {"cols": [h for h in (t.head[0] if t.head else [])[1:3]],
                     "rows": [{"v": r[0], "a": _num(r[1]), "b": _num(r[2])} for r in t.rows if len(r) >= 3]}
    t, text = _nb_cell(nb, "first_tor_frame")                    # D5: hortumun başladığı kare
    if t and t.rows:
        out["d5"] = [{"k": r[0], "n": int(_num(r[1]) or 0)} for r in t.rows if len(r) >= 2]
        out["d5_last_label"] = "True" in text.splitlines()[0] if text else None
    return out


@st.cache_data(show_spinner=False, max_entries=2)
def _d6(path: str, stamp: float) -> dict:
    """01 defteri D6: kanal başına en büyük yuvarlama hatası (uint8)."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    t, _ = _nb_cell(nb, "uint8 adımı")
    return {r[0]: _num(r[3]) for r in (t.rows if t else []) if len(r) >= 4}


@st.cache_data(show_spinner=False, max_entries=2)
def _checks(split_path: str, cat_path: str, stamp: tuple, years: tuple) -> dict:
    """02 defterindeki bölme kontrolleri, splits.csv ve katalog üzerinde canlı (src/helper/data.check_splits)."""
    from src.helper import data as HD
    from src.helper import tornet
    cat = tornet.load_catalog(cat_path)
    sp = pd.read_csv(split_path)
    r = HD.check_splits(sp, cat, list(years))
    shared = int(r["sonuç"].iloc[:4].astype(int).sum())            # ilk dört kontrol: kümeler arası ortak fırtına
    return {"n": len(r), "ok": int((r["tamam"] == "✅").sum()), "shared": shared}


def _bolme(store) -> dict | None:
    """Bölüm 5b: kümeler (örnek, fırtına, hortum oranı), örnek fırtınalar ve kontroller."""
    if store is None:
        return None
    s = store.samples
    order = ["train", "val", "test", "dropped"]
    g = s.groupby("split").agg(n=("label", "size"), tor=("label", "sum"), eps=("episode_id", "nunique"))
    sets = [{"k": k, "n": int(g.loc[k, "n"]), "tor": int(g.loc[k, "tor"]), "eps": int(g.loc[k, "eps"])} for k in order if k in g.index]
    # Örnek fırtınalar: her fırtınanın karelerinin kümesi (gösterim için en fazla 14 kare)
    rng = np.random.default_rng(0)
    eps = store.episodes
    eps = eps[(eps["n_frames"] >= 3)]
    pick = []                                       # her kümeden birkaç fırtına (çıkarılan kareleri olanlar dahil)
    for k, quota in (("train", 20), ("val", 10), ("test", 10), ("dropped", 6)):
        ids = eps.index[eps["splits"].apply(lambda x, k=k: k in x and (k == "dropped" or len(x) == 1))].to_numpy()
        ids = np.setdiff1d(ids, pick)
        pick += list(rng.choice(ids, size=min(quota, len(ids)), replace=False))
    code = {"train": "t", "val": "v", "test": "s", "dropped": "d"}
    strips = []
    for e in pick:
        rows = s.loc[eps.loc[e, "rows"]].sort_values("end_time").head(14)
        strips.append("".join(code.get(x, "d") for x in rows["split"]))
    strips.sort(key=lambda x: ("tvsd".index(max(set(x), key=x.count)), "d" in x, -len(x)))
    out = {"sets": sets, "strips": strips, "checks": None}
    sp, cp = Path(paths.data_dir()) / "splits.csv", Path(paths.data_dir()) / "catalog.csv"
    if sp.exists() and cp.exists():
        try:
            out["checks"] = _checks(str(sp), str(cp), (sp.stat().st_mtime, cp.stat().st_mtime), tuple(paths.load_config("data.json")["years"]))
        except Exception:  # noqa: BLE001  kontrol hesaplanamazsa slayt yalnız sayıları gösterir
            pass
    return out


def _onisleme(store) -> dict:
    """Bölüm 5a: doldurma değerleri (tornet.from_uint8'den), aralıklar, boyutlar, yuvarlama hatası (01 · D6)."""
    from src.helper import tornet
    cfg = paths.load_config("data.json")
    v = list(cfg["variables"])
    fill = tornet.from_uint8(np.zeros((len(v), 1, 1), dtype=np.uint8), v)[:, 0, 0]
    nb = paths.REPO_ROOT / "notebooks" / "01_explore_data.ipynb"
    n = int(len(store.samples)) if store is not None else None
    px = len(v) * geo.N_AZ * geo.N_RNG
    return {"fill": {k: float(x) for k, x in zip(v, fill)}, "ranges": {k: list(tornet.CHANNEL_RANGES[k]) for k in v},
            "err": _d6(str(nb), nb.stat().st_mtime) if nb.exists() else {},
            "bytes32": n * px * 4 if n else None, "bytes8": n * px if n else None}


def _girdi() -> dict:
    """Bölüm 4: defterdeki kanıt + configs/data.json'daki seçim (kanallar, tarama, kare)."""
    nb = paths.REPO_ROOT / "notebooks" / "01_explore_data.ipynb"
    cfg = paths.load_config("data.json")
    from src.helper.eda import VARIABLES
    return {"eda": _eda_nb(str(nb), nb.stat().st_mtime) if nb.exists() else {},
            "all_vars": list(VARIABLES), "vars": list(cfg["variables"]), "sweep": int(cfg.get("sweep", 0)),
            "frame": int(cfg.get("frame", -1)), "n_sweeps": 2, "n_frames": 4, "na": geo.N_AZ, "nr": geo.N_RNG}


# ---------------------------------------------------------------- bölüm 6: TornadoCNN
# Anlatılan modeller sabit değil: doğrulamada en iyi TornadoCNN varyantı ve en iyi ResNet-18 (evaluate.core_models).
# Varyant farkları ise başlangıç modeline (03_baseline, seed ortalaması) göre verilir.
BASE_KEY = E.BASE
TOP_MAPS = 8              # her katmandan gösterilen en güçlü özellik haritası sayısı


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _forward(net, x, steps=False, r0=None):
    """Her bloğun çıktısı (MaxPool sonrası), global havuzlama vektörü ve skor.

    steps=True: ayrıca her bloğun Conv çıktısı (ReLU öncesi, işaretli; ReLU tarayıcıda max(0, x) ile bulunur)
    ve sınıflandırıcı başının ara değerleri (Dense 64 çıktısı, logit). Bölüm 6a adım adım gösterir.
    """
    import torch
    from torch import nn
    feats, convs = [], []
    hooks = [m.register_forward_hook(lambda _m, _i, o: feats.append(o.detach()))
             for m in net.features if isinstance(m, nn.MaxPool2d)]
    hooks += [m.register_forward_hook(lambda _m, _i, o: convs.append(o.detach()))
              for m in net.features if isinstance(m, nn.Conv2d)]
    try:
        with torch.no_grad():
            r = torch.tensor([float(r0 or 0.0)], dtype=torch.float32)
            score = float(torch.sigmoid(net(x, r) if getattr(net, "coord", False) else net(x)).item())
            pooled_t = net.pool(feats[-1]).flatten(1)
            lin = [m for m in net.head if isinstance(m, nn.Linear)]
            z = lin[0](pooled_t)[0].numpy()                     # Dense 64, ReLU öncesi
            logit = float(lin[-1](torch.relu(lin[0](pooled_t)))[0, 0])
    finally:
        for h in hooks:
            h.remove()
    layers = []
    for b, f in enumerate(feats):
        a = f[0].numpy()                                        # (C, H, W), ReLU sonrası ≥ 0
        peak = a.reshape(len(a), -1).max(axis=1)
        top = np.argsort(-peak)[:TOP_MAPS]
        m = a[top] / np.maximum(peak[top], 1e-6)[:, None, None]
        L = {"c": int(a.shape[0]), "h": int(a.shape[1]), "w": int(a.shape[2]), "n": int(len(top)),
             "x": _b64((m * 255).round().astype(np.uint8))}
        if steps and b < len(convs):
            c = convs[b][0].numpy()[top]                        # aynı kanallar, Conv çıktısı (işaretli)
            ab = np.abs(c).reshape(len(c), -1).max(axis=1)
            L.update(ch=int(c.shape[1]), cw=int(c.shape[2]),
                     conv=_b64((c / np.maximum(ab, 1e-6)[:, None, None] * 127 + 128).round().clip(0, 255).astype(np.uint8)))
        layers.append(L)
    pooled = pooled_t[0].numpy()
    p = pooled / max(float(pooled.max()), 1e-6)
    out = {"layers": layers, "pooled": _b64((p * 255).round().astype(np.uint8)), "score": score}
    if steps:
        zs = z / max(float(np.abs(z).max()), 1e-6)
        out.update(dense=_b64((zs * 127 + 128).round().clip(0, 255).astype(np.uint8)), logit=logit,
                   dropout=float(next((m.p for m in net.head if isinstance(m, nn.Dropout)), 0.0)))
        if getattr(net, "coord", False):                        # CoordConv: girdiye eklenen r ve 1/r (azimut boyunca sabit)
            from src.helper.model import range_channels
            rc = range_channels(torch.tensor([float(r0 or 0.0)]), x.shape[2], x.shape[3])[0, :, 0, :].numpy()
            out["coord"] = [_b64(((c + 1) / 2 * 255).round().clip(0, 255).astype(np.uint8)) for c in rc]
    return out


@st.cache_data(show_spinner="Model katmanları hesaplanıyor…", max_entries=4)
def _cnn_acts(model_key: str, stamp: float, storm_key: str, _q: np.ndarray, variables: tuple, r0: float) -> dict | None:
    """Bölüm 3'ün fırtınası TornadoCNN'den geçer: eğitilmemiş (rastgele ağırlık, seed 0) ve eğitilmiş (best.pt)."""
    try:
        import torch
    except ImportError:
        return None
    from src.helper.model import build_model, count_params
    from src.helper.train import decode_batch
    info = next(m for m in cache.models() if m.key == model_key)
    trained = cache.net(model_key, stamp)
    torch.manual_seed(0)
    fresh = build_model("tornado_cnn", in_channels=len(variables), dropout=float(info.results.get("config", {}).get("dropout", 0.3)),
                        **info.kwargs).eval()
    x = decode_batch(torch.from_numpy(np.ascontiguousarray(_q[None])), list(variables))
    return {"before": _forward(fresh, x, steps=True, r0=r0), "after": _forward(trained, x, r0=r0), "params": int(count_params(trained)),
            "threshold": float(info.threshold), "label": info.label, "key": model_key, "coord": bool(info.kwargs.get("coord"))}


def _cnn(store, key) -> dict | None:
    info = next((m for m in cache.models() if m.key == key), None)
    if info is None or info.arch not in ("", "tornado_cnn"):
        return None
    pick = _pick_storm(cache._mtime(D.npy_dir() / D.SAMPLES_FILE), store)
    if not pick:
        return None
    rows = store.samples.loc[[pick["best"]]]
    return _cnn_acts(info.key, cache._mtime(info.folder / "best.pt"), str(pick["best"]), store.images(rows)[0],
                     tuple(store.variables), float(rows["rng_min"].iloc[0]))


def _egitim(key) -> dict | None:
    """Bölüm 6b: seçilen TornadoCNN'in eğitim geçmişi (history.json) ve kayıp ağırlığı (results.json)."""
    info = next((m for m in cache.models() if m.key == key), None)
    if info is None or not info.history:
        return None
    h, cfg = info.history, info.results.get("config", {})
    col = lambda k: [None if r.get(k) is None else float(r[k]) for r in h]   # noqa: E731
    return {"epoch": [int(r.get("tur", i + 1)) for i, r in enumerate(h)], "train_loss": col("eğitim kaybı"),
            "val_loss": col("doğrulama kaybı"), "val_pr": col("doğrulama PR-AUC"), "val_auc": col("doğrulama AUC"),
            "best": int(info.results.get("best_epoch") or 0), "patience": int(cfg.get("patience", 0)),
            "max_epochs": int(cfg.get("epochs", len(h))), "pos_weight": info.results.get("pos_weight"),
            "weighting": cfg.get("weighting"), "lr": cfg.get("lr"), "batch": cfg.get("batch_size")}


def _kind(spec: dict) -> str | None:
    """Deneyin temel modelden farkı (experiments.json): mini animasyonun türü."""
    kw, tr = spec.get("model_kwargs", {}), spec.get("train", {})
    if spec.get("arch") != "tornado_cnn":
        return None
    if kw.get("pool") == "avg":
        return "avg"
    if kw.get("batch_norm"):
        return "bn"
    if kw.get("coord"):
        return "coord"
    if tr.get("augment"):
        return "aug"
    if tr.get("label_smoothing"):
        return "ls"
    if tr.get("weighting") == "mit":
        return "mit"
    return None


def _varyant(store, chosen) -> dict | None:
    """Bölüm 6d: TornadoCNN varyantlarının doğrulama PR-AUC'si, başlangıç modelinin (03_baseline) seed ortalamasına göre.
    chosen: doğrulamada en iyi olan (sunumun anlattığı model) işaretlenir. Test sonuçları bölüm 8'de."""
    from src.helper.train import AUG_MAX_AZ, AUG_MAX_RNG, class_weights, seed_group
    allm = cache.models()

    def pr(m):
        r = E.metrics_for(m, store, "val") or {}
        v = r.get("PR-AUC")
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    base = [(m, pr(m)) for m in allm if seed_group(m.key)[0] == BASE_KEY and not m.key.endswith("_hizli")]
    base = [(m, v) for m, v in base if v is not None]
    if not base:
        return None
    vals = [v for _, v in base]
    exps = M._experiments()
    rows = []
    for name, spec in exps.items():
        k = _kind(spec)
        m = next((x for x in allm if x.key == name), None)
        if k is None or m is None or name == BASE_KEY:
            continue
        v = pr(m)
        if v is not None:
            rows.append({"key": name, "kind": k, "label": m.label, "v": v, "d": v - float(np.mean(vals)),
                         "ls": float(spec.get("train", {}).get("label_smoothing", 0) or 0), "chosen": name == chosen})
    meta = pd.DataFrame({"category": ["NUL", "WRN", "TOR", "TOR", "TOR", "TOR"], "ef_number": [-1, -1, 0, 1, 2, 3]})
    w, _ = class_weights(meta, "mit")
    return {"base": {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)), "n": len(vals)},
            "chosen": chosen, "base_chosen": chosen == BASE_KEY,
            "rows": rows, "mit": {"NUL": float(w[0]), "WRN": float(w[1]), "EF0–1": float(w[2]), "EF2+": float(w[4])},
            "aug": {"az": AUG_MAX_AZ, "rng": AUG_MAX_RNG}}


# ---------------------------------------------------------------- bölüm 6d: hiperparametre denemeleri (04)
HP_GROUPS = [   # değişen ayar → grup (öncelik sırasıyla: SGD denemesinde lr de değişir, grup optimizer'dır)
    ("optimizer", "Optimizer"), ("filters", "Model boyutu"), ("pos_weight", "Kayıp ağırlığı"), ("weighting", "Kayıp ağırlığı"),
    ("batch_size", "Batch boyutu"), ("weight_decay", "Düzenlileştirme"), ("dropout", "Düzenlileştirme"),
    ("scheduler", "Öğrenme oranı"), ("lr", "Öğrenme oranı"),
]
HP_IGNORE = {"epochs", "seed", "num_workers", "monitor"}     # en fazla tur sayısı bütün denemelerde 30 (sonucu etkilemez)
OPT_NAMES = {"sgd": "SGD", "momentum": "SGD + momentum", "rmsprop": "RMSprop", "adam": "Adam"}


def _tr(v) -> str:
    if isinstance(v, float):
        return f"{v:g}".replace(".", ",")
    if isinstance(v, list):
        return f"{v[0]}…{v[-1]}" if len(v) > 2 else ", ".join(map(str, v))
    return str(v)


def _hp_change(cfg: dict, base: dict) -> tuple[str, str] | None:
    """Denemenin baseline'dan farkı (results.json config): (grup, etiket)."""
    diff = {k: v for k, v in cfg.items() if k not in HP_IGNORE and base.get(k) != v}
    if not diff:
        return None
    group = next((g for k, g in HP_GROUPS if k in diff), "Diğer")
    parts = []
    if "optimizer" in diff:
        parts.append(OPT_NAMES.get(diff["optimizer"], diff["optimizer"]))
    if "scheduler" in diff:
        parts.append(f"lr planlama ({diff['scheduler']})")
    for k, nm in (("lr", "lr"), ("weight_decay", "L2"), ("dropout", "dropout"), ("batch_size", "batch"),
                  ("pos_weight", "pos_weight"), ("filters", "filtreler")):
        if k in diff:
            parts.append(f"{nm} = {_tr(diff[k])}")
    return group, " · ".join(parts) or ", ".join(f"{k} = {_tr(v)}" for k, v in diff.items())


def _hp(store) -> dict | None:
    """04 denemeleri (her biri baseline'ın tek bir ayarı değişmiş): doğrulama PR-AUC'si, baseline seed ortalamasına göre."""
    from src.helper.train import seed_group
    allm = cache.models()
    by = {m.key: m for m in allm}
    if BASE_KEY not in by:
        return None
    vals = [v for v in (E._val_pr(m, store) for m in allm if seed_group(m.key)[0] == BASE_KEY and not m.key.endswith("_hizli")) if v is not None]
    if not vals:
        return None
    base_cfg = by[BASE_KEY].results.get("config", {})
    rows = []
    for m in allm:
        if m.role != "hp":
            continue
        ch = _hp_change(m.results.get("config", {}), base_cfg)
        v = E._val_pr(m, store)
        if ch is None or v is None:
            continue
        rows.append({"key": m.key, "group": ch[0], "label": ch[1], "v": v, "d": v - float(np.mean(vals)),
                     "epochs": len(m.history), "best": m.results.get("best_epoch")})
    order = [g for _, g in HP_GROUPS]
    rows.sort(key=lambda r: (order.index(r["group"]) if r["group"] in order else 99, -r["d"]))
    return {"base": {"mean": float(np.mean(vals)), "min": float(min(vals)), "max": float(max(vals)), "n": len(vals)}, "rows": rows}


# ---------------------------------------------------------------- bölüm 7: ResNet-18


def _top_maps(a: np.ndarray, top=None, signed=False) -> tuple[dict, np.ndarray]:
    """(C, H, W) → en güçlü TOP_MAPS harita, harita başına ölçeklenmiş uint8 (işaretli: 128 = 0)."""
    if top is None:
        top = np.argsort(-a.reshape(len(a), -1).max(axis=1))[:TOP_MAPS]
    m = a[top]
    if signed:
        ab = np.abs(m).reshape(len(m), -1).max(axis=1)
        q = (m / np.maximum(ab, 1e-6)[:, None, None] * 127 + 128).round().clip(0, 255)
    else:
        pk = m.reshape(len(m), -1).max(axis=1)
        q = (np.clip(m, 0, None) / np.maximum(pk, 1e-6)[:, None, None] * 255).round()
    return {"c": int(a.shape[0]), "h": int(a.shape[1]), "w": int(a.shape[2]), "n": int(len(top)),
            "x": _b64(q.astype(np.uint8))}, top


def _forward_resnet(net, x, steps=False):
    """ResNet-18 aşamaları: kök (conv1, bn1 → relu, maxpool), layer1–4, global ortalama, skor."""
    import torch
    got = {}
    names = ["conv1", "bn1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]
    hooks = [getattr(net, n).register_forward_hook(lambda _m, _i, o, n=n: got.__setitem__(n, o.detach())) for n in names]
    try:
        with torch.no_grad():
            logit = float(net(x)[0, 0])
    finally:
        for h in hooks:
            h.remove()
    out = {"score": float(1 / (1 + np.exp(-logit))), "logit": logit, "stages": []}
    pool, top = _top_maps(got["maxpool"][0].numpy())
    if steps:                                                  # kök: aynı kanallar conv1 (işaretli) ve bn1 (ReLU tarayıcıda)
        out["conv1"], _ = _top_maps(got["conv1"][0].numpy(), top, signed=True)
        out["bn1"], _ = _top_maps(got["bn1"][0].numpy(), top, signed=True)
    out["pool"] = pool
    for n in ("layer1", "layer2", "layer3", "layer4"):
        out["stages"].append(_top_maps(got[n][0].numpy())[0])
    v = got["avgpool"].flatten().numpy()
    out["avg"] = _b64((v / max(float(v.max()), 1e-6) * 255).round().clip(0, 255).astype(np.uint8))
    out["n_avg"] = int(len(v))
    return out


def _filters(net) -> str:
    """İlk katmanın 64 filtresi (64, 3, 7, 7) → her biri kendi aralığına ölçeklenmiş RGB uint8 (64, 7, 7, 3)."""
    w = net.conv1.weight.detach().numpy()
    lo, hi = w.min(axis=(1, 2, 3), keepdims=True), w.max(axis=(1, 2, 3), keepdims=True)
    return _b64(((w - lo) / np.maximum(hi - lo, 1e-9) * 255).round().astype(np.uint8).transpose(0, 2, 3, 1))


@st.cache_data(show_spinner="ResNet katmanları hesaplanıyor…", max_entries=4)
def _res_acts(res_key: str, other_key: str | None, stamp: tuple, storm_key: str, _q: np.ndarray, variables: tuple) -> dict | None:
    """Bölüm 3'ün fırtınası ResNet-18'den geçer: başlangıç ağırlıklarıyla (ön-eğitimliyse yalnız ImageNet, değilse
    rastgele; son katman yeni, seed 0) ve radarla eğitilmiş (best.pt). Ayrıca ilk katmanın filtreleri."""
    try:
        import torch
        from src.helper.model import count_params, make_resnet18
        from src.helper.train import decode_batch
    except ImportError:
        return None
    by = {m.key: m for m in cache.models()}
    info = by[res_key]
    pre = bool(info.kwargs.get("pretrained", False))
    tuned = cache.net(res_key, cache._mtime(info.folder / "best.pt"))
    drop = float(info.results.get("config", {}).get("dropout", 0.3))
    torch.manual_seed(0)
    rnd = make_resnet18(in_channels=len(variables), pretrained=False, dropout=drop).eval()
    torch.manual_seed(0)
    try:
        imnet = make_resnet18(in_channels=len(variables), pretrained=True, dropout=drop).eval()
    except Exception:  # noqa: BLE001  ImageNet ağırlıkları indirilemezse
        imnet = None
    start = imnet if pre else rnd
    if start is None:
        return None
    x = decode_batch(torch.from_numpy(np.ascontiguousarray(_q[None])), list(variables))
    # rnd: rastgele ağırlıklı ResNet-18 = ResNet baseline'ının başlangıç noktası (7a mimariyi bununla anlatır)
    out = {"before": _forward_resnet(start, x, steps=True), "after": _forward_resnet(tuned, x), "pre": pre,
           "rnd": _forward_resnet(rnd, x, steps=True),
           "params": int(count_params(tuned)), "threshold": float(info.threshold), "label": info.label, "key": res_key,
           "filters": {"start": _filters(start), "tuned": _filters(tuned), "random": _filters(rnd)}}
    if imnet is not None:
        out["filters"]["imagenet"] = _filters(imnet)
    out["base_key"] = _family_base("resnet18")
    if other_key in by:
        out["other_label"] = by[other_key].label
        out["other_pre"] = bool(by[other_key].kwargs.get("pretrained", False))
        other = cache.net(other_key, cache._mtime(by[other_key].folder / "best.pt"))
        out["filters"]["other_tuned"] = _filters(other)
        out["other_last"] = _forward_resnet(other, x)["stages"][-1]        # 7e: son aşamanın en güçlü haritası
        out["other_key"] = other_key
    return out


def _resnet(store, res_key, other_key) -> dict | None:
    by = {m.key: m for m in cache.models()}
    if res_key not in by or store is None:
        return None
    pick = _pick_storm(cache._mtime(D.npy_dir() / D.SAMPLES_FILE), store)
    if not pick:
        return None
    q = store.images(store.samples.loc[[pick["best"]]])[0]
    stamp = tuple(cache._mtime(by[k].folder / "best.pt") for k in (res_key, other_key) if k in by)
    return _res_acts(res_key, other_key, stamp, str(pick["best"]), q, tuple(store.variables))


def _hist(info) -> dict:
    h, cfg = info.history, info.results.get("config", {})
    col = lambda k: [None if r.get(k) is None else float(r[k]) for r in h]   # noqa: E731
    return {"epoch": [int(r.get("tur", i + 1)) for i, r in enumerate(h)], "train_loss": col("eğitim kaybı"),
            "val_loss": col("doğrulama kaybı"), "val_pr": col("doğrulama PR-AUC"),
            "best": int(info.results.get("best_epoch") or 0), "patience": int(cfg.get("patience", 0)),
            "max_epochs": int(cfg.get("epochs", len(h))), "lr": cfg.get("lr"),
            "sec": float(np.mean([r.get("süre (sn)", np.nan) for r in h])) if h else None}


def _res_egitim(res_key, cnn_key) -> dict | None:
    """Bölüm 7b: seçilen ResNet'in eğitim geçmişi + karşılaştırma için seçilen CNN'inki."""
    by = {m.key: m for m in cache.models()}
    if res_key not in by or not by[res_key].history:
        return None
    out = _hist(by[res_key])
    out["label"] = by[res_key].label
    base = _family_base("resnet18")
    if base and base != res_key and base in by and by[base].history:      # ResNet baseline'ı (sıfırdan) karşılaştırma için
        out["base"] = _hist(by[base])
        out["base_label"] = by[base].label
    if cnn_key in by and by[cnn_key].history:
        out["cnn"] = _hist(by[cnn_key])
        out["cnn_label"] = by[cnn_key].label
    return out


def _res_karsi(store, keys) -> dict | None:
    """Bölüm 7d–e: temel 3 modelin seed'lerinin doğrulama PR-AUC'si, parametre, tur süresi (keys sırasıyla)."""
    from src.helper.train import seed_group
    allm = cache.models()

    def pr(m):
        r = E.metrics_for(m, store, "val") or {}
        v = r.get("PR-AUC")
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    groups = {}
    for key in keys:
        ms = [m for m in allm if seed_group(m.key)[0] == key and not m.key.endswith("_hizli")]
        if not ms:
            continue
        head = next((m for m in ms if m.key == key), ms[0])
        vals = [v for v in (pr(m) for m in ms) if v is not None]
        if vals:
            groups[key] = {"label": head.label, "vals": vals, "mean": float(np.mean(vals)),
                           "params": head.params or None, "sec": _hist(head)["sec"] if head.history else None,
                           "arch": head.arch or "tornado_cnn", "pre": bool(head.kwargs.get("pretrained", False))}
    return {"order": [k for k in keys if k in groups], "groups": groups} if groups else None


# ---------------------------------------------------------------- bölüm 8: test sonuçları (yalnız temel 3 model + baseline referansı)
N_BOOT_TEST = 500         # fırtına bazlı bootstrap (Karşılaştırma sekmesiyle aynı)


def _family_base(arch: str) -> str | None:
    """Mimarinin baseline deneyi: TornadoCNN → 03_baseline; ResNet-18 → ön-eğitimsiz (sıfırdan) eğitilen."""
    if arch == "tornado_cnn":
        return BASE_KEY
    from src.helper.train import seed_group
    for m in cache.models():
        if (m.arch or "tornado_cnn") == arch and m.role in ("main", "ablation") and not m.kwargs.get("pretrained", False):
            return seed_group(m.key)[0]
    return None


def _thin(x, y, n=240):
    """Eğriyi en fazla n noktaya indirir (uçlar korunur)."""
    k = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return [round(float(v), 4) for v in np.asarray(x)[k]], [round(float(v), 4) for v in np.asarray(y)[k]]


@st.cache_data(show_spinner="Test sonuçları hesaplanıyor…", max_entries=2)
def _test_cached(stamps: tuple, keys: tuple, _store) -> dict | None:
    from sklearn.metrics import precision_recall_curve, roc_curve
    from src.helper import evaluate as ev
    from src.helper.train import seed_group
    by = {m.key: m for m in cache.models()}
    data = {}
    for k in keys:
        s, p = E.split_scores(by[k], _store, "test")
        if s is not None and s["label"].nunique() > 1:
            data[k] = (s, p)
    if not data:
        return None
    common = set.intersection(*(set(s.index) for s, _ in data.values()))
    idx0 = sorted(common)
    al = {k: pd.Series(p, index=s.index).loc[idx0].to_numpy() for k, (s, p) in data.items()}
    s0 = _store.samples.loc[idx0]
    y = s0["label"].to_numpy()
    meta = E.meta(s0)
    boot_idx = ev.bootstrap_indices(s0["episode_id"].to_numpy(), n_boot=N_BOOT_TEST, seed=0)
    # Her mimarinin kendi baseline'ı (_family_base): TornadoCNN → 03_baseline, ResNet-18 → sıfırdan eğitilen.
    # Δ ve etiket modelin kendi ailesinin baseline'ına göre (Karşılaştırma sekmesindeki kural; tek seed'li
    # baseline'da seed sapması yok, yalnız güven aralığı kullanılır).
    bases = {}
    for arch in {(by[k].arch or "tornado_cnn") for k in al}:
        bk = _family_base(arch)
        if bk is None:
            continue
        bp = []
        for m in cache.models():
            if seed_group(m.key)[0] != bk or m.key.endswith("_hizli"):
                continue
            s, p = E.split_scores(m, _store, "test")
            if s is not None:
                pp = pd.Series(p, index=s.index).reindex(idx0).to_numpy()
                if not np.isnan(pp).any():
                    bp.append((m, pp))
        if not bp:
            continue
        vals = [float(ev.summarize(y, p, m.threshold)["PR-AUC"]) for m, p in bp]
        bases[arch] = {"key": bk, "label": by[bk].label, "vals": vals, "mean": float(np.mean(vals)),
                       "boot": np.mean([ev.bootstrap(y, p, m.threshold, boot_idx)["PR-AUC"].to_numpy() for m, p in bp], axis=0),
                       "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")}
    models = []
    for k in keys:
        if k not in al:
            continue
        m, p = by[k], al[k]
        sm = ev.summarize(y, p, m.threshold)
        bt = ev.bootstrap(y, p, m.threshold, boot_idx)
        fpr, tpr, _ = roc_curve(y, p)
        pre, rec, _ = precision_recall_curve(y, p)
        rx, ry = _thin(fpr, tpr)
        px, py = _thin(rec[::-1], pre[::-1])
        vt = ev.view_table(meta, p, m.threshold)
        ef = ev.ef_table(meta, p, m.threshold)
        arch = m.arch or "tornado_cnn"
        row = {"key": k, "label": m.label, "thr": float(m.threshold), "arch": arch,
               **{x: float(sm[x]) for x in ("PR-AUC", "ROC-AUC", "POD", "SR", "CSI", "F1", "Bias")},
               "TP": int(sm["TP"]), "FP": int(sm["FP"]), "FN": int(sm["FN"]), "TN": int((y == 0).sum() - sm["FP"]),
               "pr_ci": [float(x) for x in ev.ci(bt["PR-AUC"])], "csi_ci": [float(x) for x in ev.ci(bt["CSI"])],
               "roc": {"x": rx, "y": ry}, "pr": {"x": px, "y": py},
               "views": {v: {"pr": float(r["PR-AUC"]), "rate": float(r["hortum oranı"]), "n": int(r["n"])} for v, r in vt.iterrows()},
               "ef": {c: float(r["eşiği geçen oran"]) for c, r in ef.iterrows() if c.startswith("EF")},
               "ef_n": {c: int(r["n"]) for c, r in ef.iterrows() if c.startswith("EF")}}
        B = bases.get(arch)
        if B is not None and seed_group(k)[0] == B["key"]:
            row.update(is_base=True, base_label=B["label"])
        elif B is not None:
            d = row["PR-AUC"] - B["mean"]
            lo, hi = ev.ci(bt["PR-AUC"].to_numpy() - B["boot"])
            sd, out = B["sd"], not (lo <= 0 <= hi)
            verdict = ("fark" if out and abs(d) > 2 * sd else "eğilim" if abs(d) > sd else "gürültü") if sd == sd else ("fark" if out else "belirsiz")
            row.update(d=d, d_ci=[float(lo), float(hi)], verdict=verdict, base_label=B["label"], base_vals=B["vals"])
        models.append(row)
    # tarayıcıya: önyükleme dizisi gitmez, NaN (tek seed'li baseline'ın sapması) JSON'da geçersiz → None
    bases = {a: {k: (None if isinstance(v, float) and v != v else v) for k, v in b_.items() if k != "boot"} for a, b_ in bases.items()}
    return {"n": int(len(y)), "n_tor": int(y.sum()), "storms": int(s0["episode_id"].nunique()), "rate": float(y.mean()),
            "n_boot": N_BOOT_TEST, "models": models, "bases": bases}


def _test(store, keys) -> dict | None:
    by = {m.key: m for m in cache.models()}
    keys = [k for k in keys if k in by]
    if store is None or not keys:
        return None
    stamps = tuple((m.key, m.preds_path.stat().st_mtime if m.has_preds else 0.0) for m in cache.models())
    return _test_cached(stamps, tuple(keys), store)


# ---------------------------------------------------------------- bölüm 9: açıklanabilirlik ve kapanış
XAI_NEAR_KM = 3.0         # "dönmenin üstünde" sayılan mesafe (Açıklanabilirlik sekmesiyle aynı)


def _xai_storm(store, keys) -> dict | None:
    """Bölüm 3'ün fırtınasının bütün karelerinde üç modelin Grad-CAM'i (uint8, 2× küçültülmüş: 60 × 120)."""
    if store is None:
        return None
    pick = _pick_storm(cache._mtime(D.npy_dir() / D.SAMPLES_FILE), store)
    if not pick:
        return None
    rows = store.samples.loc[pick["seq"]]
    q = store.images(rows)
    r0 = rows["rng_min"].to_numpy(dtype=np.float32)
    by = {m.key: m for m in cache.models()}
    out = []
    for k in keys:
        m = by.get(k)
        if m is None:
            continue
        try:
            c = cache.cam(k, cache._mtime(m.folder / "best.pt"), f"sunum:{pick['best']}", q, tuple(store.variables), r0)
        except Exception:  # noqa: BLE001  torch yoksa ya da model açılamazsa
            continue
        F_, H, W = c.shape
        small = c.reshape(F_, H // 2, 2, W // 2, 2).mean(axis=(2, 4)).round().astype(np.uint8)
        out.append({"key": k, "label": m.label, "h": H // 2, "w": W // 2, "x": _b64(small)})
    return {"models": out} if out else None


DEL_STEPS = 10           # silme testi: %0 → %50, 10 adım (5 puanlık)


@st.cache_data(show_spinner="Açıklama adımları hesaplanıyor…", max_entries=2)
def _xai_steps_cached(model_key: str, stamp: float, storm_key: str, _q: np.ndarray, variables: tuple, r0: float) -> dict | None:
    """Bölüm 9'un "nasıl çalışır" slaytları için, seçilen CNN ve bölüm 3'ün karesiyle gerçek ara değerler:
    Grad-CAM'in adımları, kanal Shapley'nin 8 alt kümesi, silme testinin iki sırası (src/helper/xai ile aynı yöntem)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None
    from src.helper import xai as X
    from src.helper.model import bind_range
    net = cache.net(model_key, stamp)
    wrapped = bind_range(net, [r0])
    q = np.ascontiguousarray(_q)
    # --- Grad-CAM adım adım (son evrişim bloğunun ReLU çıktısı, havuzlamadan önce)
    layer, got = X.cam_layer(net), {}

    def keep(_m, _i, o):
        got["a"] = o
        o.register_hook(lambda g: got.__setitem__("g", g))
    h = layer.register_forward_hook(keep)
    try:
        x = X._x(q[None], variables, X._device(net)).requires_grad_(True)
        net.zero_grad()
        wrapped(x).squeeze(1).sum().backward()
        A, G = got["a"][0].detach(), got["g"][0].detach()               # (K, h, w)
    finally:
        h.remove()
    w = G.mean(dim=(1, 2))                                                # her haritanın ağırlığı
    raw = (w[:, None, None] * A).sum(0)                                   # ağırlıklı toplam (işaretli)
    pos = F.relu(raw)
    up = F.interpolate(pos[None, None], size=q.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    up = up / max(float(up.max()), 1e-9)
    K, hh, ww = A.shape
    order = torch.argsort(w, descending=True)[:TOP_MAPS].numpy()
    An = A.numpy()[order]
    An = An / np.maximum(An.reshape(len(An), -1).max(1), 1e-9)[:, None, None]
    sgn = lambda m: _b64((m / max(float(np.abs(m).max()), 1e-9) * 127 + 128).round().clip(0, 255).astype(np.uint8))   # noqa: E731
    wn = (w / max(float(w.abs().max()), 1e-9)).numpy()
    cam = {"k": int(K), "h": int(hh), "w": int(ww), "maps": _b64((An * 255).round().astype(np.uint8)), "n": int(len(order)),
           "map_w": [round(float(wn[i]), 3) for i in order], "weights": [round(float(v), 3) for v in np.sort(wn)[::-1]],
           "raw": sgn(raw.numpy()), "relu": _b64((pos.numpy() / max(float(pos.max()), 1e-9) * 255).round().astype(np.uint8)),
           "up": _b64((up * 255).round().astype(np.uint8))}
    # --- kanal Shapley: 2^C alt küme (kapalı kanal = boş, uint8 0)
    sh = X.channel_shapley(wrapped, q[None], list(variables))
    import itertools
    subs = []
    for sset in itertools.product([0, 1], repeat=len(variables)):
        qs = q.copy()
        for c, on in enumerate(sset):
            if not on:
                qs[c] = 0
        subs.append({"on": list(sset), "score": float(X.predict(wrapped, qs[None], list(variables))[0])})
    subs.sort(key=lambda r: (sum(r["on"]), r["on"][::-1]))
    shap = {"subsets": subs, "base": float(sh["base"][0]), "score": float(sh["score"][0]),
            "phi": {v: float(sh["phi"][0, i]) for i, v in enumerate(variables)}}
    # --- silme testi: Grad-CAM sırası ve rastgele sıra; her piksel hangi adımda siliniyor (255: hiç)
    n = q.shape[1] * q.shape[2]
    f1, s1 = X.deletion_curve(wrapped, q, list(variables), up, steps=DEL_STEPS)
    rng_order = np.random.default_rng(0).permutation(n)
    f2, s2 = X.deletion_curve(wrapped, q, list(variables), None, steps=DEL_STEPS, rng=np.random.default_rng(0))

    def step_of(order_):
        st_ = np.full(n, 255, dtype=np.uint8)
        for k, fr in enumerate(f1):
            m = order_[:int(round(fr * n))]
            st_[m] = np.minimum(st_[m], k)
        return _b64(st_)
    cam_order = np.argsort(-up.ravel(), kind="stable")
    dele = {"fracs": [round(float(v), 3) for v in f1], "cam": [round(float(v), 4) for v in s1], "rnd": [round(float(v), 4) for v in s2],
            "cam_step": step_of(cam_order), "rnd_step": step_of(rng_order),
            "auc_cam": X.curve_area(f1, s1), "auc_rnd": X.curve_area(f2, s2)}
    return {"cam": cam, "shap": shap, "deletion": dele, "variables": list(variables)}


def _xai_steps(store, key) -> dict | None:
    by = {m.key: m for m in cache.models()}
    if store is None or key not in by:
        return None
    pick = _pick_storm(cache._mtime(D.npy_dir() / D.SAMPLES_FILE), store)
    if not pick:
        return None
    rows = store.samples.loc[[pick["best"]]]
    return _xai_steps_cached(key, cache._mtime(by[key].folder / "best.pt"), str(pick["best"]), store.images(rows)[0],
                             tuple(store.variables), float(rows["rng_min"].iloc[0]))


def _xai_global(keys) -> dict | None:
    """Önceden hesaplanmış açıklama özetleri (scripts/make_xai.py, make_xai_global.py):
    kanal payları (bütün test kümesi, kanal SHAP), Grad-CAM zirvesinin dönmeye uzaklığı ve silme testi."""
    by = {m.key: m for m in cache.models()}
    variables = ["DBZ", "VEL", "WIDTH"]

    def shares(d):
        a = np.abs(d[[f"phi_{v}" for v in variables]].to_numpy())
        ssum = a.sum(1, keepdims=True)
        a = np.divide(a, ssum, out=np.zeros_like(a), where=ssum > 1e-6)
        return {v: float(a[:, i].mean()) for i, v in enumerate(variables)} if len(a) else None

    out = []
    for k in keys:
        m = by.get(k)
        if m is None:
            continue
        row = {"key": k, "label": m.label}
        sp, xp = m.folder / "shap_all.parquet", m.folder / "xai.parquet"
        if sp.exists():
            sh = pd.read_parquet(sp)
            row["shap"] = {kind: shares(sh[sh["kind"] == kind]) for kind in ("tp", "fn", "fp", "tn")}
            row["n_shap"] = int(len(sh))
        if xp.exists():
            df = pd.read_parquet(xp)
            tp, pos = df[df["kind"] == "tp"], df[df["kind"].isin(["tp", "fp"])]
            d = tp["dist_cam"].dropna() if "dist_cam" in tp else pd.Series(dtype=float)
            row["dist"] = {"median": float(d.median()) if len(d) else None, "near": float((d <= XAI_NEAR_KM).mean()) if len(d) else None, "n": int(len(d))}
            if len(pos) and "auc_cam" in pos and "auc_rnd" in pos:
                row["deletion"] = {"cam": float(pos["auc_cam"].mean()), "rnd": float(pos["auc_rnd"].mean()), "n": int(len(pos))}
        out.append(row)
    return {"models": out, "near_km": XAI_NEAR_KM} if out else None


def state(ss, store, models) -> dict:
    core = E.core_models(store) if store is not None else []
    arch = lambda m: m.arch or "tornado_cnn"   # noqa: E731
    cnn_key = next((m.key for m in core if arch(m) == "tornado_cnn"), BASE_KEY)
    res = [m.key for m in core if arch(m) == "resnet18"]
    res_key, other_key = (res + [None, None])[:2]
    return {"kapak": _kapak(store), "veri": _veri(store), "storm": _storm(store), "girdi": _girdi(),
            "onisleme": _onisleme(store), "bolme": _bolme(store),
            "cnn": _cnn(store, cnn_key) if store is not None else None, "egitim": _egitim(cnn_key), "varyant": _varyant(store, cnn_key), "hp": _hp(store) if store is not None else None,
            "resnet": _resnet(store, res_key, other_key) if res_key else None,
            "res_egitim": _res_egitim(res_key, cnn_key) if res_key else None,
            "res_karsi": _res_karsi(store, [k for k in (cnn_key, res_key, other_key) if k]),
            "test": _test(store, [k for k in (cnn_key, res_key, other_key) if k]),
            "xai_storm": _xai_storm(store, [k for k in (cnn_key, res_key, other_key) if k]) if ss.get("torch_ok", True) else None,
            "xai": _xai_global([k for k in (cnn_key, res_key, other_key) if k]),
            "xai_steps": _xai_steps(store, cnn_key) if ss.get("torch_ok", True) else None}


def body(ss, store, models):
    pass                      # sunumun tamamı bileşende çizilir
