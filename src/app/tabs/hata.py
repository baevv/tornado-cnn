"""Hata Analizi: seçili model ve eşikte kaçırılan hortumlar, yanlış alarmlar ve doğru yakalananlar.

Her kart bir radar karesidir (CNN girdisi, VEL); Grad-CAM açıksa modelin baktığı yer üstüne bindirilir.
"Gezgin'de aç" o fırtınayı o karede açar.

Kompozit düğmesi (karar X1): test kümesinin dört uç grubunun (en iyi isabet, en kötü yanlış alarm, en kötü ıska,
en iyi doğru ret) ortalama radar görüntüsü, ortalama Grad-CAM'i ve ortalama konum haritası
(scripts/make_xai_global.py → models/<ad>/composites.npz). TorNet pencereleri olay konumuna ortalandığı için
ortalama almak anlamlıdır.
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
    ss.setdefault("ht_comp", False)


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
    elif a == "hata_comp":
        ss.ht_comp = bool(ev["on"])


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
            "counts": {k: int(counts.get(k, 0)) for k in ("fn", "fp", "tp")}, "cam": ss.ht_cam, "sort_conf": ss.ht_sort,
            "comp": ss.ht_comp}


def body(ss, store, models):
    info = next((m for m in models if m.key == ss.ht_model), None)
    if ss.ht_comp and info is not None:
        composites(info)
        return
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
            cam = gradcam(n, q, store.variables, r0_km=show["rng_min"].to_numpy(dtype="float32"))
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
            b1, b2 = st.columns(2, gap="small")
            if b2.button("Açıkla", key=f"xai_{idx}", icon=":material/lightbulb:", use_container_width=True):
                from src.app.tabs import xai
                frames = store.frames(r["episode_id"])
                xai.open_frame(ss, r["episode_id"], list(frames.index).index(idx), info.key)
                st.rerun()
            if b1.button("Gezgin'de aç", key=f"open_{idx}", icon=":material/radar:", use_container_width=True):
                frames = store.frames(r["episode_id"])
                ss.tab, ss.gz_ep, ss.gz_filter, ss.gz_split = "gezgin", int(r["episode_id"]), "all", "all"
                ss.gz_frame = int(list(frames.index).index(idx))
                ss.sel_model = info.key
                st.rerun()
    if len(sel) > N_CARDS:
        st.html(f"<div class='note'>İlk {N_CARDS} / {len(sel):,} kare gösteriliyor.</div>".replace(",", "."))


# ---------------------------------------------------------------- kompozitler (X1)

COMP_FILE = "composites.npz"


@st.cache_data(show_spinner=False, max_entries=8)
def _comp(path: str, stamp: float) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _to_q(img: np.ndarray, var: str) -> np.ndarray:
    """Fiziksel ortalama (H, W) → arayüzün uint8 ölçeği (renk tabloları için); NaN = boş."""
    from src.helper.tornet import CHANNEL_RANGES
    lo, hi = CHANNEL_RANGES[var]
    q = np.round((np.clip(img, lo, hi) - lo) / (hi - lo) * 254 + 1)
    return np.where(np.isfinite(img), q, 0).astype(np.uint8)


def composites(info):
    from src.helper.xai import COMPOSITE_GROUPS
    path = info.folder / COMP_FILE
    if not path.exists():
        st.html(f"<div class='note'>{info.label} için kompozit yok. Bir kez hesapla: "
                f"<code>python scripts/make_xai_global.py --models {info.key} --only composites</code></div>")
        return
    z = _comp(str(path), cache._mtime(path))
    variables = [str(v) for v in z["variables"]]
    base = "DBZ" if "DBZ" in variables else variables[0]               # Grad-CAM ve konum haritasının zemini
    groups = [k for k in COMPOSITE_GROUPS if f"{k}_img" in z]
    rot_max = max((float(np.nanmax(z[f"{k}_rot"])) for k in groups if f"{k}_rot" in z), default=1.0)
    st.html("<div class='note'>Test kümesinin dört uç grubu; her gruptan aynı fırtınadan yalnızca bir kare. Satırlar: "
            "DBZ ve WIDTH ortalaması (fiziksel birimde, boş pikseller dışarıda); <b>dönme</b> ortalaması (VEL'de 5 × 3 "
            "hücrelik pencerede en büyük hız farkı; VEL'in kendi ortalaması alınmaz, çünkü hız çiftinin yönü olaydan "
            "olaya değiştiği için + ve − birbirini götürür); DBZ üstünde ortalama Grad-CAM ve ortalama konum haritası "
            "(X4, skor ölçeği). Pencereler olay konumuna ortalı olduğu için ortalama, grubun tipik görünümünü "
            "verir.</div>")
    cols = st.columns(len(groups), gap="small")
    for c, k in zip(cols, groups):
        name, rule = COMPOSITE_GROUPS[k]
        pr = z[f"{k}_prob"]
        img = z[f"{k}_img"]
        qb = _to_q(img[variables.index(base)], base)
        with c:
            st.html(f"<div class='h-panel'>{name}</div><div class='xd'>{rule} · {len(pr)} kare · ortalama skor "
                    f"{pr.mean():.3f}</div>")
            for v_i, v in enumerate(variables):
                if v == "VEL" and f"{k}_rot" in z:
                    r = z[f"{k}_rot"]
                    st.image(P.heat_rgb(r, rot_max), use_container_width=True,
                             caption=f"Dönme ortalaması (en yüksek {np.nanmax(r):.0f} m/s; ölçek ortak)")
                elif v != "VEL":
                    st.image(P.radar_rgb(_to_q(img[v_i], v), v), use_container_width=True, caption=f"{v} ortalaması")
            st.image(P.radar_rgb(qb, base, z[f"{k}_cam"] / max(float(z[f"{k}_cam"].max()), 1e-6)),
                     use_container_width=True, caption=f"Grad-CAM ortalaması ({base} üstünde)")
            if f"{k}_loc" in z:
                st.image(P.radar_rgb(qb, base, z[f"{k}_loc"]), use_container_width=True,
                         caption=f"Konum haritası ortalaması (en yüksek skor {float(z[f'{k}_loc'].max()):.2f})")
