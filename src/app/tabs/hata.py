"""Hata Analizi: seçili model ve eşikte kaçırılan hortumlar, yanlış alarmlar ve doğru yakalananlar.

Her kart bir radar karesidir (CNN girdisi, VEL); Grad-CAM açıksa modelin baktığı yer üstüne bindirilir.
"Gezgin'de aç" o fırtınayı o karede açar.
"""
import numpy as np
import streamlit as st

from src.app import cache
from src.app import evaluate as E
from src.app import plots as P

N_CARDS = 12


def defaults(ss, models):
    ss.setdefault("ht_kind", "fn")
    ss.setdefault("ht_model", ss.sel_model)
    ss.setdefault("ht_split", "val")
    ss.setdefault("ht_thr", {})
    ss.setdefault("ht_cam", True)
    ss.setdefault("ht_sort", True)


def handle(ev, ss):
    a = ev["a"]
    if a == "hata_kind":
        ss.ht_kind = ev["kind"]
    elif a == "hata_model":
        ss.ht_model = ev["key"]
    elif a == "hata_split":
        ss.ht_split = ev["split"]
    elif a == "hata_thr":
        ss.ht_thr = {**ss.ht_thr, ss.ht_model: float(ev["thr"])}
    elif a == "hata_cam":
        ss.ht_cam = bool(ev["on"])
    elif a == "hata_sort":
        ss.ht_sort = bool(ev["on"])


def _thr(ss, info):
    return ss.ht_thr.get(info.key, info.threshold) if info else 0.5


def _table(ss, store, info):
    s, p = E.split_scores(info, store, ss.ht_split)
    if s is None:
        return None
    t = s[["episode_id", "filename", "end_time", "category", "ef", "label", "year", "row"]].copy()
    t["prob"] = p
    fire = p >= _thr(ss, info)
    y = t["label"].to_numpy() == 1
    t["kind"] = np.where(y, np.where(fire, "tp", "fn"), np.where(fire, "fp", "tn"))
    return t


def state(ss, store, models) -> dict:
    info = next((m for m in models if m.key == ss.ht_model), None)
    t = _table(ss, store, info) if info else None
    counts = t["kind"].value_counts().to_dict() if t is not None else {}
    return {"kind": ss.ht_kind, "model": ss.ht_model, "split": ss.ht_split, "thr": _thr(ss, info),
            "counts": {k: int(counts.get(k, 0)) for k in ("fn", "fp", "tp")}, "cam": ss.ht_cam, "sort_conf": ss.ht_sort}


def body(ss, store, models):
    info = next((m for m in models if m.key == ss.ht_model), None)
    t = _table(ss, store, info) if info else None
    if t is None:
        st.html(E.no_scores_note(ss.ht_split))
        return
    sel = t[t["kind"] == ss.ht_kind]
    if ss.ht_sort:                                     # en emin hata önce
        sel = sel.sort_values("prob", ascending=ss.ht_kind == "fn")
    else:
        sel = sel.sort_values("end_time")
    if sel.empty:
        st.html("<div class='note'>Bu eşikte bu türden kare yok.</div>")
        return
    show = sel.head(N_CARDS)
    q = store.images(show)
    cam = None
    if ss.ht_cam and ss.torch_ok:
        try:
            n = cache.net(info.key, cache._mtime(info.folder / "best.pt"))
            from src.app.models import gradcam
            cam = gradcam(n, q, store.variables)
        except Exception as e:  # noqa: BLE001
            st.toast(f"Grad-CAM: {e}", icon=":material/error:")
    vi = store.variables.index("VEL") if "VEL" in store.variables else 0
    cols = st.columns(4)
    for i, (idx, r) in enumerate(show.iterrows()):
        with cols[i % 4]:
            img = P.radar_rgb(q[i, vi], store.variables[vi], None if cam is None else cam[i])
            ef = f" · EF{r['ef']}" if r["category"] == "TOR" and r["ef"] >= 0 else ""
            st.image(img, use_container_width=True,
                     caption=f"#{r['episode_id']} · {r['end_time']:%Y-%m-%d %H:%M} · {r['category']}{ef} · skor {r['prob']:.3f}")
            if st.button("Gezgin'de aç", key=f"open_{idx}", icon=":material/radar:", use_container_width=True):
                frames = store.frames(r["episode_id"])
                ss.tab, ss.gz_ep, ss.gz_filter, ss.gz_split = "gezgin", int(r["episode_id"]), "all", "all"
                ss.gz_frame = int(list(frames.index).index(idx))
                ss.sel_model = info.key
                st.rerun()
    if len(sel) > N_CARDS:
        st.html(f"<div class='note'>İlk {N_CARDS} / {len(sel):,} kare gösteriliyor.</div>".replace(",", "."))
