"""Harita (kararlar V1–V3): fırtınalar ve istasyonlar ABD haritasında.

  Genel     (V1) Her fırtına (episode) bir işaret; hortumlu fırtınalar EF'ye göre renklenir ve izleri çizilir.
            Renk ayrıca seçili modelin en yüksek skoruna ya da fırtına düzeyindeki karara göre seçilebilir.
            Bir fırtınaya tıklanınca aynı sekmede bilgi kartı açılır: zaman, istasyon, EF, kare sayıları, modelin kare
            kare skoru ve kararı. Kartta seçilen kare (V2) kutupsal ızgaradan coğrafi koordinatlara yeniden
            örneklenip haritaya yarı saydam katman olarak oturtulur (DBZ, VEL, Grad-CAM ya da konum haritası).
  İstasyon  (V3) İstasyon başına PR-AUC / CSI; en az 30 TOR karesi olan istasyonlar. Modelin eğitimde görmediği
            bölümler (doğrulama + test) birlikte kullanılır; test tek başına çok az istasyonda 30 hortuma ulaşır.

Altlık Carto Voyager (OpenStreetMap), API anahtarı gerektirmez; harita karoları için internet gerekir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app import cache
from src.app import data as D
from src.app import evaluate as E
from src.app import geo
from src.app import plots as P
from src.app import theme as T

CFG = {"displayModeBar": False, "scrollZoom": True}
MIN_TOR = 30
CAT_TR = {"TOR": "Hortum", "WRN": "Uyarı", "NUL": "Fırtına"}
LAYERS = {"DBZ": "DBZ", "VEL": "VEL", "cam": "Grad-CAM", "loc": "Konum haritası"}
US_CENTER = dict(lat=38.0, lon=-96.0)
US_BOUNDS = dict(west=-128.0, east=-64.0, south=23.0, north=50.5)   # yalnız ABD verisi var
# EF ölçeği: sarıdan koyu kırmızıya (güçlü hortum = koyu ve büyük)
EF_COLORS = {0: "#F2C230", 1: "#F08A24", 2: "#DE4B2B", 3: "#9E1B32"}
EF_SIZE = {0: 9, 1: 11, 2: 13, 3: 16}
EP_KIND = {"p": ("Yakaladı", T.GOOD), "n": ("Kaçırdı", T.TOR), "f": ("Yanlış alarm", T.ACCENT), "t": ("Doğru ret", T.NUL)}


def defaults(ss, models):
    ss.setdefault("hr_view", "genel")
    ss.setdefault("hr_color", "cat")
    ss.setdefault("hr_model", ss.sel_model)
    ss.setdefault("hr_split", "test")
    ss.setdefault("hr_ep", None)           # seçilen fırtına (episode_id)
    ss.setdefault("hr_frame", None)        # o fırtınada haritaya oturtulan kare (sıra)
    ss.setdefault("hr_layer", "DBZ")


def handle(ev, ss):
    a = ev["a"]
    if a == "harita_view":
        ss.hr_view = ev["view"]
        ss.hr_color = "cat" if ss.hr_view == "genel" else "prauc"
    elif a == "harita_color":
        ss.hr_color = ev["color"]
    elif a == "harita_model":
        ss.hr_model = ev["key"]
    elif a == "harita_split":
        ss.hr_split, ss.hr_ep, ss.hr_frame = ev["split"], None, None


def state(ss, store, models) -> dict:
    if ss.hr_model not in {m.key for m in models} and models:
        ss.hr_model = models[0].key
    return {"view": ss.hr_view, "color": ss.hr_color, "model": ss.hr_model, "split": ss.hr_split}


def body(ss, store, models):
    if store is None:
        return
    info = next((m for m in models if m.key == ss.hr_model), None)
    if ss.hr_view == "istasyon":
        stations(store, info, ss.hr_color)
    else:
        overview(ss, store, info)


# ---------------------------------------------------------------- ortak

@st.cache_data(show_spinner=False)
def _radars(stamp: float, _samples: pd.DataFrame) -> pd.DataFrame:
    return geo.radar_positions(_samples)


def radars(store) -> pd.DataFrame:
    return _radars(cache._mtime(D.npy_dir() / D.SAMPLES_FILE), store.samples)


def _base_fig(height=640, center=None, zoom=3.5, rev="hr") -> go.Figure:
    f = go.Figure()
    f.update_layout(**T.plotly_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                                      legend=dict(orientation="h", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)",
                                                  font=dict(size=12))))
    f.update_layout(map=dict(style="carto-voyager", center=center or US_CENTER, zoom=zoom, bounds=US_BOUNDS),
                    uirevision=rev)
    return f


def _radar_trace(r: pd.DataFrame, size=5, name="Radar") -> go.Scattermap:
    return go.Scattermap(lat=r["lat"], lon=r["lon"], mode="markers", name=name,
                         marker=dict(size=size, color="#3A4352", opacity=0.55, symbol="circle"), text=r["site"],
                         hovertemplate="%{text}<extra>radar</extra>")


def _first(cd):
    """Seçilen noktanın customdata'sından ilk değer: Plotly bunu liste, sözlük ya da tek değer olarak döndürebilir."""
    if isinstance(cd, dict):
        cd = [cd[k] for k in sorted(cd, key=lambda k: int(k) if str(k).isdigit() else 0)]
    if isinstance(cd, (list, tuple)):
        cd = cd[0] if cd else None
    return None if cd is None else int(float(cd))


# ---------------------------------------------------------------- fırtına tablosu

@st.cache_data(show_spinner=False, max_entries=16)
def _episodes(key: tuple, split: str, _samples: pd.DataFrame, _p) -> pd.DataFrame:
    """Fırtına başına: konum (hortum karelerinin ya da bütün karelerin ortası), zaman, istasyon, EF, kare sayıları,
    seçili modelin en yüksek skoru ve fırtına düzeyindeki karar (Gezgin şeridiyle aynı kural)."""
    s = _samples if split == "all" else _samples[_samples["split"] == split]
    s = s.assign(prob=np.nan if _p is None else _p[s.index.to_numpy()] if len(s) else [])
    s = s.assign(tor=s["label"] == 1)
    g = s.groupby("episode_id", sort=False)
    ep = pd.DataFrame({
        "start": g["start_time"].min(), "end": g["end_time"].max(), "site": g["site"].first(),
        "n": g.size(), "n_tor": g["tor"].sum(),
        "n_wrn": g["category"].apply(lambda c: int((c == "WRN").sum())),
        "max_ef": g["ef"].max(), "max_p": g["prob"].max(),
    })
    tor = s[s["tor"]].groupby("episode_id")[["lat", "lon"]].mean()
    allc = g[["lat", "lon"]].mean()
    ep[["lat", "lon"]] = allc
    ep.loc[tor.index, ["lat", "lon"]] = tor
    ep["cat"] = np.where(ep["n_tor"] > 0, "TOR", np.where(ep["n_wrn"] > 0, "WRN", "NUL"))
    return ep


def _outcomes(info, store, ep_index, p) -> pd.Series:
    if info is None or p is None:
        return pd.Series("x", index=ep_index)
    stamp = tuple(m.preds_path.stat().st_mtime for m in (info.members or [info]) if m.has_preds)
    c = cache.strip_codes(info.key, info.threshold, stamp, len(p), store.samples, p)
    return c.reindex(ep_index).fillna("x")


TRACK_LINK_KM = 40.0       # bir hortum karesi, en yakın izin son noktasına bundan yakınsa o ize eklenir


def _split_tracks(la, lo):
    """Zaman sırasındaki hortum karelerini izlere ayırır: her kare en yakın izin son noktasına (≤ TRACK_LINK_KM)
    bağlanır, yoksa yeni iz başlar. Aynı fırtına kaydında aynı anda birden fazla hortum varsa çizgi zikzak çizmez."""
    tracks = []
    for a, o in zip(la, lo):
        best, bd = None, TRACK_LINK_KM
        for t in tracks:
            pa, po = t[-1]
            d = np.hypot((a - pa) * geo.KM_PER_DEG, (o - po) * geo.KM_PER_DEG * np.cos(np.radians(a)))
            if d <= bd:
                best, bd = t, d
        if best is None:
            tracks.append([(a, o)])
        else:
            best.append((a, o))
    return [t for t in tracks if len(t) >= 2]


def _tracks(store, eps: pd.DataFrame, ef: int):
    """Hortumlu fırtınaların izleri (izler arası None)."""
    lat, lon = [], []
    s = store.samples
    for ep_id in eps.index:
        rows = s.loc[store.episodes.loc[ep_id, "rows"]]
        rows = rows[rows["label"] == 1].sort_values("end_time")
        for t in _split_tracks(rows["lat"].to_numpy(), rows["lon"].to_numpy()):
            lat += [float(a) for a, _ in t] + [None]
            lon += [float(o) for _, o in t] + [None]
    return lat, lon


# ---------------------------------------------------------------- V1 + V2: genel bakış

def overview(ss, store, info):
    p = E._probs(info, store) if info is not None else None
    key = (info.key if info else "", ss.hr_split, E._stamp(info) if info else ())
    ep = _episodes(key, ss.hr_split, store.samples, p)
    if ep.empty:
        st.html(f"<div class='note'>Bu bölümde ({ss.hr_split}) fırtına yok.</div>")
        return
    sel = ss.hr_ep if ss.hr_ep in ep.index else None
    r = radars(store)
    color = ss.hr_color if p is not None else "cat"

    center, zoom = US_CENTER, 3.5
    if sel is not None:
        center, zoom = dict(lat=float(ep.loc[sel, "lat"]), lon=float(ep.loc[sel, "lon"])), 7.0
    f = _base_fig(center=center, zoom=zoom, rev=f"hr-{ss.hr_split}-{sel}")
    f.add_trace(_radar_trace(r))
    traces = []                                        # (iz, episode_id dizisi): seçimi customdata'sız da çözmek için

    def add(df, name, color_, size, opacity=0.9, **marker):
        if df.empty:
            return
        txt = (df["site"] + " · " + df["start"].dt.strftime("%Y-%m-%d %H:%M")).to_numpy()
        mk = dict(size=size, color=color_, opacity=opacity, **marker)
        f.add_trace(go.Scattermap(lat=df["lat"], lon=df["lon"], mode="markers", name=name, marker=mk, text=txt,
                                  customdata=df.index.to_numpy(), hovertemplate="%{text}<extra>" + name.split(" ·")[0] + "</extra>"))
        traces.append(df.index.to_numpy())

    nul, wrn, tor = ep[ep["cat"] == "NUL"], ep[ep["cat"] == "WRN"], ep[ep["cat"] == "TOR"]
    if color == "cat":
        add(nul, f"Hortumsuz fırtına · {len(nul):,}".replace(",", "."), T.NUL, 5, 0.45)
        add(wrn, f"Uyarı · {len(wrn):,}".replace(",", "."), T.WRN, 6, 0.6)
        for k in (0, 1, 2, 3):
            g = tor[tor["max_ef"].clip(lower=0).clip(upper=3) == k]
            lat, lon = _tracks(store, g, k)
            if lat:
                f.add_trace(go.Scattermap(lat=lat, lon=lon, mode="lines", line=dict(width=2.5, color=EF_COLORS[k]),
                                          hoverinfo="skip", showlegend=False))
                traces.append(np.array([], dtype=int))
            add(g, f"{'EF3+' if k == 3 else f'EF{k}'} · {len(g)}", EF_COLORS[k], EF_SIZE[k], 0.95)
    elif color == "score":
        order = ep.sort_values("max_p", na_position="first")
        add(order, "en yüksek skor", None, 8, 0.9)
        f.data[-1].marker.update(color=order["max_p"], colorscale=[[0, "#C9D3DF"], [0.5, "#F0A04B"], [1, "#8A1A0E"]],
                                 cmin=0, cmax=1, colorbar=dict(title=dict(text="skor"), thickness=10, len=0.45, x=0.99))
        f.data[-1].name = "fırtınanın en yüksek skoru"
    else:
        out = _outcomes(info, store, ep.index, p)
        for code in ("t", "p", "f", "n"):
            g = ep[out == code]
            name, col = EP_KIND[code]
            add(g, f"{name} · {len(g):,}".replace(",", "."), col, 6 if code == "t" else 10, 0.5 if code == "t" else 0.95)

    layer_note = None
    if sel is not None:
        f.add_trace(go.Scattermap(lat=[ep.loc[sel, "lat"]], lon=[ep.loc[sel, "lon"]], mode="markers",
                                  marker=dict(size=26, color="rgba(19,26,36,0.18)"), hoverinfo="skip", showlegend=False))
        traces.append(np.array([], dtype=int))
        try:
            layer, layer_note = _overlay(ss, store, info, sel, r)
            f.update_layout(map_layers=[layer])
        except Exception as e:  # noqa: BLE001
            layer_note = f"Katman çizilemedi: {e}"

    c1, c2 = st.columns([2.6, 1.25], gap="small")
    with c1:
        evt = st.plotly_chart(f, use_container_width=True, config=CFG, on_select="rerun", selection_mode="points",
                              key=f"hr_map_{ss.hr_split}_{color}_{info.key if info else ''}")
        _pick(ss, evt, traces, n_fixed=1)
    with c2:
        if sel is None:
            _legend_card(ep, ss.hr_split)
        else:
            _storm_card(ss, store, info, sel, ep, p, layer_note)


def _pick(ss, evt, traces, n_fixed):
    sel = getattr(evt, "selection", None) if not isinstance(evt, dict) else evt.get("selection")
    pts = (sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)) or []
    for pt in pts:
        ep_id = None
        try:
            ep_id = _first(pt.get("customdata"))
        except (TypeError, ValueError):
            pass
        if ep_id is None:                              # yedek: iz ve nokta sırasından
            ci, pi = pt.get("curve_number"), pt.get("point_index")
            if ci is not None and pi is not None and 0 <= ci - n_fixed < len(traces):
                arr = traces[ci - n_fixed]
                ep_id = int(arr[pi]) if pi < len(arr) else None
        if ep_id is not None and ep_id != ss.hr_ep:
            ss.hr_ep, ss.hr_frame = ep_id, None
            st.rerun()
        break


def _legend_card(ep, split):
    tor = ep[ep["cat"] == "TOR"]
    by_ef = tor["max_ef"].clip(lower=0, upper=3).value_counts()
    chips = "".join(f"<span style='display:inline-flex;align-items:center;gap:6px;margin:0 10px 6px 0'>"
                    f"<i style='width:12px;height:12px;border-radius:50%;background:{EF_COLORS[k]};display:inline-block'></i>"
                    f"{'EF3+' if k == 3 else f'EF{k}'} <b>{int(by_ef.get(k, 0))}</b></span>" for k in (0, 1, 2, 3))
    split_tr = {"all": "bütün bölümler", "train": "eğitim", "val": "doğrulama", "test": "test"}.get(split, split)
    st.html("<div class='xbox'><div class='h-panel'>Fırtınalar</div>"
            f"<div class='xd' style='font-size:13px;line-height:1.55'>{len(ep):,} fırtına ({split_tr}) · "
            f"<b>{len(tor)}</b> hortumlu, {int((ep['cat'] == 'WRN').sum()):,} uyarılı, "
            f"{int((ep['cat'] == 'NUL').sum()):,} hortumsuz.<br><br>{chips}<br>"
            "Renkli çizgiler hortumun kareler boyunca izidir. Bir fırtınaya tıkla: bilgileri burada açılır ve "
            "radar karesi haritaya oturtulur. Koyu gri noktalar radarlar.</div></div>".replace(",", "."))


def _score_fig(rows: pd.DataFrame, scores, thr, color) -> go.Figure:
    f = go.Figure(layout=T.plotly_layout(height=170, margin=dict(l=36, r=8, t=8, b=26), showlegend=False))
    x = rows["end_time"].dt.strftime("%H:%M").tolist()
    band = [T.CATEGORY_COLORS.get(c, T.NUL) for c in rows["category"]]
    f.add_bar(x=x, y=[1.0] * len(x), marker_color=band, opacity=0.18, hoverinfo="skip")
    if scores is not None:
        f.add_scatter(x=x, y=scores, mode="lines+markers", line=dict(color=color, width=2.2), marker=dict(size=6),
                      hovertemplate="%{x} · skor %{y:.3f}<extra></extra>")
        f.add_hline(y=thr, line=dict(color=T.INK, width=1, dash="dot"))
    f.update_yaxes(range=[0, 1.02], dtick=0.5)
    f.update_xaxes(type="category", tickfont=dict(size=10))
    return f


def _storm_card(ss, store, info, ep_id, ep, p, layer_note):
    rows = store.frames(ep_id)
    e = ep.loc[ep_id]
    cats = rows["category"].tolist()
    if ss.hr_frame is None or not 0 <= ss.hr_frame < len(rows):
        sc = None if p is None else p[rows.index.to_numpy()]
        ss.hr_frame = (cats.index("TOR") if "TOR" in cats else int(np.nanargmax(sc)) if sc is not None and np.isfinite(sc).any()
                       else len(rows) - 1)
    ef = int(e["max_ef"])
    if e["cat"] == "TOR":
        head_col, title = EF_COLORS[min(max(ef, 0), 3)], f"Hortum · {'EF?' if ef < 0 else f'EF{ef}'}"
    else:
        head_col, title = (T.WRN, "Uyarı fırtınası") if e["cat"] == "WRN" else (T.NUL, "Hortumsuz fırtına")
    verdict = ""
    if info is not None and p is not None:
        code = _outcomes(info, store, pd.Index([ep_id]), p).iloc[0]
        if code in EP_KIND:
            name, col = EP_KIND[code]
            verdict = (f"<span style='background:{col};color:#fff;border-radius:6px;padding:2px 8px;font-weight:600'>"
                       f"{name}</span>")
    dur = (e["end"] - e["start"]).total_seconds() / 60
    stats = [("Tarih", f"{e['start']:%Y-%m-%d}"), ("Saat", f"{e['start']:%H:%M}–{e['end']:%H:%M} ({dur:.0f} dk)"),
             ("İstasyon", e["site"]), ("Kareler", f"{len(rows)} · <span style='color:{T.TOR}'>{int(e['n_tor'])} TOR</span>"),
             ("En yüksek skor", "—" if not np.isfinite(e["max_p"]) else f"{e['max_p']:.3f}"),
             ("Eşik", "—" if info is None else f"{info.threshold:.3f}")]
    grid = "".join(f"<div><div class='k'>{k}</div><div class='n' style='font-size:15px'>{v}</div></div>" for k, v in stats)
    st.html(f"<div class='xbox' style='border-top:6px solid {head_col}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px'>"
            f"<div class='h-panel' style='margin:0'>{title}</div>{verdict}</div>"
            f"<div class='xd' style='margin:2px 0 8px'>Fırtına #{ep_id} · {info.label if info else ''}</div>"
            f"<div class='xk' style='grid-template-columns:1fr 1fr;margin:0'>{grid}</div></div>")
    scores = None if p is None else [None if np.isnan(x) else float(x) for x in p[rows.index.to_numpy()]]
    st.plotly_chart(_score_fig(rows, scores, info.threshold if info else 0.5, info.color if info else T.INK),
                    use_container_width=True, config={"displayModeBar": False})
    if len(rows) > 1:
        labels = [f"{i + 1} · {t:%H:%M} · {c}" for i, (t, c) in enumerate(zip(rows["end_time"], cats))]
        pick = st.select_slider("Haritadaki kare", options=list(range(len(rows))), value=int(ss.hr_frame),
                                format_func=lambda i: labels[i], key=f"hr_fr_{ep_id}")
        if pick != ss.hr_frame:
            ss.hr_frame = int(pick)
            st.rerun()
    opts = ["DBZ", "VEL"] + (["cam", "loc"] if ss.torch_ok and info is not None and not info.members else [])
    lay = st.segmented_control("Katman", options=opts, format_func=LAYERS.get,
                               default=ss.hr_layer if ss.hr_layer in opts else "DBZ", key="hr_layer_ctl",
                               label_visibility="collapsed")
    if lay and lay != ss.hr_layer:
        ss.hr_layer = lay
        st.rerun()
    b1, b2, b3 = st.columns(3, gap="small")
    if b1.button("Gezgin", icon=":material/radar:", use_container_width=True, key="hr_open", help="Gezgin'de aç"):
        ss.tab, ss.gz_ep, ss.gz_filter, ss.gz_split, ss.gz_frame = "gezgin", int(ep_id), "all", "all", int(ss.hr_frame)
        if info is not None and not info.members:
            ss.sel_model = info.key
        st.rerun()
    if b2.button("Açıkla", icon=":material/lightbulb:", use_container_width=True, key="hr_explain",
                 disabled=info is None or bool(info.members), help="Bu kareyi Açıklanabilirlik sekmesinde aç"):
        from src.app.tabs import xai
        xai.open_frame(ss, int(ep_id), int(ss.hr_frame), info.key)
        st.rerun()
    if b3.button("Kapat", icon=":material/close:", use_container_width=True, key="hr_clear"):
        ss.hr_ep, ss.hr_frame = None, None
        st.rerun()
    if layer_note:
        st.html(f"<div class='note'>{layer_note}</div>")


@st.cache_data(show_spinner="Kare haritaya oturtuluyor…", max_entries=32)
def _overlay_png(key: str, layer: str, _q, _variables, az0, r0, lat0, lon0, _info):
    variables = list(_variables)
    base = "DBZ" if "DBZ" in variables else variables[0]
    if layer in ("DBZ", "VEL"):
        v = layer if layer in variables else base
        qi = _q[variables.index(v)]
        return geo.frame_image(P.radar_rgb(qi, v), az0, r0, lat0, lon0, valid=qi > 0)
    from src.helper.model import bind_range
    net = cache.net(_info.key, cache._mtime(_info.folder / "best.pt"))
    nb = bind_range(net, np.array([r0], dtype=np.float32))
    if layer == "cam":
        from src.helper.xai import gradcam
        m = gradcam(nb, _q[None], variables)[0]
    else:
        from src.helper.xai import location_map
        m = location_map(nb, _q[None], variables)["score_map"][0]
    qb = _q[variables.index(base)]
    return geo.frame_image(P.radar_rgb(qb, base, m), az0, r0, lat0, lon0, valid=(qb > 0) | (m > 0.15))


def _overlay(ss, store, info, ep_id, r):
    rows = store.frames(ep_id)
    row = rows.iloc[int(ss.hr_frame or 0)] if ss.hr_frame is not None else rows.iloc[0]
    q = store.images(row.to_frame().T)[0]
    rad = r.set_index("site").loc[row["site"]]
    layer = ss.hr_layer
    if layer in ("cam", "loc") and (info is None or info.members or not ss.torch_ok):
        layer = "DBZ"
    img = _overlay_png(f"{row['filename']}|{info.key if info else ''}|{layer}", layer, q, tuple(store.variables),
                       float(row["az_min"]), float(row["rng_min"]), float(rad["lat"]), float(rad["lon"]), info)
    note = (f"Haritadaki kare ({row['end_time']:%H:%M}, {LAYERS[layer]}) kutupsal ızgaradan coğrafi koordinatlara "
            "yeniden örneklenmiştir; model kutupsal veriyle çalışır. Radar konumu katalogdan hesaplandı "
            f"({row['site']}, tutarlılık ortanca {rad['spread_km']:.1f} km).")
    return {"sourcetype": "image", "source": img["source"], "coordinates": img["coordinates"], "opacity": 0.8,
            "below": "traces"}, note


# ---------------------------------------------------------------- V3: istasyona göre başarım

@st.cache_data(show_spinner="İstasyon ölçütleri hesaplanıyor…", max_entries=16)
def _site_table(key: tuple, _samples, _p, thr: float) -> pd.DataFrame:
    return geo.site_metrics(_samples, _p, thr, min_tor=MIN_TOR)


def stations(store, info, color):
    r = radars(store)
    if info is None:
        st.html("<div class='note'>Model seçilmedi.</div>")
        return
    p = E._probs(info, store)
    if p is None:
        st.html(E.no_scores_note("test"))
        return
    s = store.samples
    held = [x for x in ("val", "test") if (s["split"] == x).any()]
    m = s["split"].isin(held).to_numpy() & ~np.isnan(p)
    t = _site_table((info.key, tuple(held), E._stamp(info), info.threshold), s[m], p[m], info.threshold)
    col = "PR-AUC" if color == "prauc" else "CSI"
    f = _base_fig(height=600, rev=f"st-{info.key}-{col}")
    f.add_trace(_radar_trace(r[~r["site"].isin(t["site"])] if len(t) else r, size=5,
                             name=f"< {MIN_TOR} hortum karesi"))
    if len(t):
        tt = t.merge(r[["site", "lat", "lon"]], on="site")
        size = 10 + 22 * np.sqrt(tt["n_tor"] / tt["n_tor"].max())
        lo, hi = (0.2, 0.8) if col == "PR-AUC" else (0.1, 0.6)
        f.add_trace(go.Scattermap(
            lat=tt["lat"], lon=tt["lon"], mode="markers+text", name=col, text=tt["site"], textposition="top center",
            textfont=dict(size=10, color=T.INK),
            marker=dict(size=size, color=tt[col], colorscale="RdYlGn", cmin=lo, cmax=hi, opacity=0.9,
                        colorbar=dict(title=dict(text=col), thickness=10, len=0.5, x=0.99)),
            customdata=np.stack([tt["n_tor"], tt["n"], tt["PR-AUC"], tt["CSI"], tt["ROC-AUC"], tt["hortumlu fırtına"]], 1),
            hovertemplate=("%{text}<br>hortum karesi %{customdata[0]:.0f} / %{customdata[1]:.0f} · "
                           "%{customdata[5]:.0f} hortumlu fırtına<br>PR-AUC "
                           "%{customdata[2]:.3f} · CSI %{customdata[3]:.3f} · ROC-AUC %{customdata[4]:.3f}<extra></extra>")))
    st.plotly_chart(f, use_container_width=True, config=CFG)
    split_txt = " + ".join({"val": "doğrulama", "test": "test"}[x] for x in held)
    st.html(f"<div class='note'>{info.label} · bölüm: {split_txt} (modelin eğitimde görmediği kareler) · en az "
            f"{MIN_TOR} hortum karesi olan <b>{len(t)}</b> istasyon / {len(r)}. Daire büyüklüğü hortum karesi sayısı; "
            "gri noktalar eşiğin altında kalan radarlar. Hortum oranı istasyondan istasyona değiştiği için PR-AUC "
            "istasyonlar arasında dikkatli kıyaslanmalı; ROC-AUC orandan bağımsızdır. Bir istasyonun hortum kareleri "
            "birkaç fırtınadan gelir (tablodaki <i>hortumlu fırtına</i>): kareler bağımsız değildir, tek bir iyi ya da "
            "kötü yakalanan fırtına değeri belirler. Bu harita bölgesel bir eğilimi değil, istasyon başına örneklemi "
            "gösterir."
            + ("" if "val" in held else " <b>Doğrulama bölümü yok</b> (veri klasöründe splits.csv): yalnız test kullanıldı, "
               "çok az istasyon eşiği geçer.") + "</div>")
    if len(t):
        show = t.sort_values(col, ascending=False).round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)
