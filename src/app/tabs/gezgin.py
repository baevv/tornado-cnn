"""Fırtına Gezgini: filtre/bölüm/fırtına seçimi ve bileşene gönderilen fırtına verisi."""
from __future__ import annotations

import base64

import numpy as np
import streamlit as st

from src.app import cache
from src.app import data as D
from src.helper.tornet import CHANNEL_RANGES

SPLIT_TR = {"train": "eğitim", "val": "doğrulama", "test": "test", "dropped": "çıkarılan"}
DAZ, DR = 0.5, 0.25          # TorNet: açı adımı (derece), mesafe adımı (km)


def defaults(ss):
    ss.setdefault("gz_filter", "tor")
    ss.setdefault("gz_split", "test")
    ss.setdefault("gz_ep", None)
    ss.setdefault("gz_frame", None)       # Hata Analizi'nden gelince açılacak kare


def handle(ev: dict, ss, store) -> None:
    a = ev.get("a")
    if a == "filter":
        ss.gz_filter, ss.gz_ep = ev["filter"], None
    elif a == "split":
        ss.gz_split, ss.gz_ep = ev["split"], None
    elif a == "goto":
        ss.gz_ep = int(ev["id"])
    elif a == "random" and store is not None:
        ids = D.pool(store.episodes, ss.gz_filter, ss.gz_split)
        if len(ids):
            ss.gz_ep = int(np.random.default_rng().choice(ids))
    elif a == "epid" and store is not None:
        try:
            ep = int(ev["id"])
        except ValueError:
            return
        if ep in store.episodes.index:
            ss.gz_ep = ep
            ids = D.pool(store.episodes, ss.gz_filter, ss.gz_split)
            if ep not in ids:                       # filtre dışındaysa her şeyi göster
                ss.gz_filter, ss.gz_split = "all", "all"
        else:
            st.toast(f"Episode {ep} bulunamadı.", icon=":material/search_off:")
    ss.gz_frame = None


def _missing_message(root) -> str:
    return (f"<b>Radar verisi bulunamadı.</b><br>Beklenen klasör: <code>{root}</code><br><br>"
            "İşlenmiş .npz dosyalarını bir kez .npy'ye çevir:<br>"
            "<code>export TORNADO_DATA_DIR=\"…/tornado-cnn/data\"</code><br>"
            "<code>python scripts/export_npy.py</code><br>"
            "Skorlar için: <code>python scripts/make_preds.py</code>")


def state(ss, store, root, models: list) -> dict:
    """Bileşene gönderilecek "gezgin" bölümü."""
    if store is None:
        return {"filter": ss.gz_filter, "split": ss.gz_split, "counts": {}, "strip": {"codes": "", "ids": []},
                "pos": 0, "episode": None, "message": _missing_message(root)}

    eps = store.episodes
    counts = D.filter_counts(eps, ss.gz_split)
    ids = D.pool(eps, ss.gz_filter, ss.gz_split)
    sel = next((m for m in models if m.key == ss.sel_model), None)

    codes = "x" * len(ids)
    if sel is not None:
        p = cache.probs(sel, store.samples)
        if p is not None:
            c = cache.strip_codes(sel.key, sel.threshold, cache._mtime(sel.preds_path), len(p), store.samples, p)
            codes = "".join(c.reindex(ids).fillna("x").to_numpy())

    if len(ids) == 0:
        return {"filter": ss.gz_filter, "split": ss.gz_split, "counts": counts, "strip": {"codes": "", "ids": []},
                "pos": 0, "episode": None, "message": "Bu filtre ve bölümde fırtına yok."}
    if ss.gz_ep not in ids:
        ss.gz_ep = int(ids[0])
    pos = int(ids.get_loc(ss.gz_ep))

    return {"filter": ss.gz_filter, "split": ss.gz_split, "counts": counts,
            "strip": {"codes": codes, "ids": [int(i) for i in ids]}, "pos": pos,
            "episode": episode_payload(ss, store, models, ss.gz_ep)}


def episode_payload(ss, store, models: list, ep_id: int) -> dict:
    rows = store.frames(ep_id)
    q = store.images(rows)                                   # uint8 (F, C, 120, 240)
    variables = tuple(store.variables)
    ep = store.episodes.loc[ep_id]
    stamp = cache._mtime(D.npy_dir() / D.SAMPLES_FILE)

    # Model skorları: önce preds.parquet, yoksa (torch varsa) canlı
    scores = {}
    for m in models:
        p = cache.probs(m, store.samples)
        if p is not None and m.results.get("variables", list(variables)) == list(variables):
            v = p[rows.index.to_numpy()]
            scores[m.key] = [None if np.isnan(x) else round(float(x), 4) for x in v]
        elif ss.torch_ok and m.results.get("variables", list(variables)) == list(variables):
            try:
                scores[m.key] = [round(x, 4) for x in cache.live_scores(m.key, cache._mtime(m.folder / "best.pt"), ep_id, q, variables)]
            except Exception as e:  # noqa: BLE001  model açılamazsa kart boş kalır
                st.toast(f"{m.label}: {e}", icon=":material/error:")

    # Grad-CAM: seçili model için
    cam_b64, cam_model = None, None
    sel = next((m for m in models if m.key == ss.sel_model), None)
    if ss.torch_ok and sel is not None:
        try:
            c = cache.cam(sel.key, cache._mtime(sel.folder / "best.pt"), ep_id, q, variables)
            cam_b64, cam_model = base64.b64encode(c.tobytes()).decode("ascii"), sel.key
        except Exception as e:  # noqa: BLE001
            st.toast(f"Grad-CAM: {e}", icon=":material/error:")

    cats = rows["category"].tolist()
    if ss.gz_frame is not None and 0 <= ss.gz_frame < len(rows):
        init = int(ss.gz_frame)
    else:
        init = cats.index("TOR") if "TOR" in cats else len(cats) - 1

    splits = sorted({SPLIT_TR.get(s, str(s)) for s in rows["split"]})
    frames = [{
        "time": f"{t:%H:%M}", "date": f"{t:%Y-%m-%d}", "cat": c, "ef": int(e),
        "az0": float(a), "r0": float(r), "fn": str(fn),
    } for t, c, e, a, r, fn in zip(rows["end_time"], rows["category"], rows["ef"], rows["az_min"], rows["rng_min"], rows["filename"])]

    return {
        "key": f"{ep_id}" + (f"@{ss.gz_frame}" if ss.gz_frame is not None else ""),
        "shape": list(q.shape),
        "x": base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii"),
        "variables": list(variables),
        "ranges": {v: list(CHANNEL_RANGES[v]) for v in variables},
        "daz": DAZ, "dr": DR,
        "frames": frames,
        "rot": cache.rotation(ep_id, stamp, q, variables),
        "scores": scores,
        "cam": cam_b64, "cam_model": cam_model,
        "init_frame": init,
        "meta": {"episode_id": str(ep_id), "split": "/".join(splits), "site": str(rows["site"].iloc[0]) if "site" in rows else "",
                 "n_tor": int(ep["n_tor"]), "n_wrn": int(ep["n_wrn"]), "n_nul": int(ep["n_nul"])},
    }
