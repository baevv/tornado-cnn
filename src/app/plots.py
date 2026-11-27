"""Plotly grafikleri (Özet, Model Karşılaştırma, Hata Analizi) ve küçük radar resimleri."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.app import theme as T
from src.helper.tornet import CHANNEL_RANGES


def _fig(**kw) -> go.Figure:
    return go.Figure(layout=T.plotly_layout(**kw))


def roc(curves: list) -> go.Figure:
    """curves: [(etiket, renk, fpr, tpr, auc)]"""
    f = _fig(title=dict(text="ROC"), height=440)
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.LINE, dash="dash", width=1), showlegend=False, hoverinfo="skip")
    for lab, col, x, y, auc in curves:
        f.add_scatter(x=x, y=y, mode="lines", name=f"{lab} · {auc:.3f}", line=dict(color=col, width=2.2),
                      hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra>" + lab + "</extra>")
    f.update_xaxes(title="yanlış alarm oranı (FPR)", range=[0, 1])
    f.update_yaxes(title="yakalama (TPR)", range=[0, 1.01])
    return f


def pr(curves: list, base_rate: float | None = None) -> go.Figure:
    """curves: [(etiket, renk, recall, precision, ap)]"""
    f = _fig(title=dict(text="Precision–Recall"), height=440)
    if base_rate is not None:
        f.add_hline(y=base_rate, line=dict(color=T.LINE, dash="dash", width=1))
    for lab, col, r, p, ap in curves:
        f.add_scatter(x=r, y=p, mode="lines", name=f"{lab} · {ap:.3f}", line=dict(color=col, width=2.2),
                      hovertemplate="recall %{x:.3f}<br>precision %{y:.3f}<extra>" + lab + "</extra>")
    f.update_xaxes(title="recall (yakalama)", range=[0, 1])
    f.update_yaxes(title="precision (isabet)", range=[0, 1.01])
    return f


def confusion(tp, fp, fn, tn, title="Karmaşıklık matrisi", color=T.INK, height=330) -> go.Figure:
    z = np.array([[tn, fp], [fn, tp]], dtype=float)
    share = z / max(z.sum(), 1)
    txt = [[f"{int(v):,}".replace(",", ".") + f"<br><span style='font-size:11px'>%{100 * s:.1f}</span>" for v, s in zip(r, sr)] for r, sr in zip(z, share)]
    f = _fig(title=dict(text=title), height=height, margin=dict(l=70, r=12, t=44, b=50))
    f.add_heatmap(z=np.log1p(z), x=["hortum yok", "hortum"], y=["hortum yok", "hortum"], text=txt, texttemplate="%{text}",
                  textfont=dict(family="IBM Plex Mono, monospace", size=15), colorscale=[[0, T.PANEL_2], [1, color]],
                  showscale=False, hoverinfo="skip", xgap=3, ygap=3)
    f.update_xaxes(title="tahmin", side="bottom", showgrid=False)
    f.update_yaxes(title="gerçek", autorange="reversed", showgrid=False)
    return f


def learning(series: list, metric: str, title: str) -> go.Figure:
    """series: [(etiket, renk, tur listesi, değer listesi, kesikli mi)]"""
    f = _fig(title=dict(text=title), height=400)
    for lab, col, x, y, dash in series:
        f.add_scatter(x=x, y=y, mode="lines+markers", name=lab, line=dict(color=col, width=2, dash="dot" if dash else "solid"),
                      marker=dict(size=5), hovertemplate="tur %{x}<br>%{y:.4f}<extra>" + lab + "</extra>")
    f.update_xaxes(title="tur (epoch)", dtick=1)
    f.update_yaxes(title=metric)
    return f


def score_hist(probs: np.ndarray, cats: np.ndarray, thr: float, title="Skor dağılımı") -> go.Figure:
    f = _fig(title=dict(text=title), height=330, barmode="overlay")
    bins = dict(start=0, end=1, size=0.025)
    for c in ("NUL", "WRN", "TOR"):
        v = probs[cats == c]
        if len(v):
            f.add_histogram(x=v, xbins=bins, name=f"{c} · {len(v):,}".replace(",", "."), marker_color=T.CATEGORY_COLORS[c],
                            opacity=0.75 if c == "TOR" else 0.55, histnorm="probability",
                            hovertemplate="skor %{x}<br>oran %{y:.3f}<extra>" + c + "</extra>")
    f.add_vline(x=thr, line=dict(color=T.INK, width=1.5, dash="dash"))
    f.update_xaxes(title="model skoru", range=[0, 1])
    f.update_yaxes(title="oran (kategori içinde)")
    return f


# ---------------------------------------------------------------- değerlendirme grafikleri (M4)

def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def perf_diagram(curves: list, height=520) -> go.Figure:
    """Performans diyagramı (Roebber, 2009). curves: [(etiket, renk, y, p, eşik)].
    x = SR (isabet), y = POD (yakalama); dolgu CSI, kesikli doğrular frekans yanlılığı; nokta = seçilen eşik."""
    from sklearn.metrics import precision_recall_curve

    from src.helper.evaluate import decision_scores
    f = _fig(title=dict(text="Performans diyagramı"), height=height)
    g = np.linspace(0.01, 1, 120)
    S, P = np.meshgrid(g, g)
    f.add_contour(x=g, y=g, z=1 / (1 / S + 1 / P - 1), contours=dict(start=0.1, end=0.9, size=0.1, coloring="fill",
                  showlabels=True, labelfont=dict(size=10, color=T.MUTED)),
                  colorscale=[[0, T.PANEL], [1, "#C3CCD8"]], showscale=False, line=dict(width=0.6, color=T.LINE),
                  hoverinfo="skip", name="CSI")
    for b in (0.5, 1, 1.5, 2, 4):
        x = np.array([0, min(1, 1 / b)])
        f.add_scatter(x=x, y=b * x, mode="lines", line=dict(color=T.FAINT, dash="dash", width=1), showlegend=False,
                      hoverinfo="skip")
        f.add_annotation(x=min(0.97, 0.97 / b), y=b * min(0.97, 0.97 / b), text=f"{b:g}", showarrow=False,
                         font=dict(size=10, color=T.MUTED), yshift=8)
    for lab, col, y, p, thr in curves:
        prec, rec, _ = precision_recall_curve(y, p)
        f.add_scatter(x=prec[:-1], y=rec[:-1], mode="lines", name=lab, line=dict(color=col, width=2.2),
                      hovertemplate="SR %{x:.3f}<br>POD %{y:.3f}<extra>" + lab + "</extra>")
        d = decision_scores(y, p, thr)
        f.add_scatter(x=[d["SR"]], y=[d["POD"]], mode="markers", showlegend=False,
                      marker=dict(color=col, size=11, line=dict(color=T.INK, width=1.2)),
                      hovertemplate=(f"eşik {thr:.3f}<br>SR {d['SR']:.3f} · POD {d['POD']:.3f}<br>CSI {d['CSI']:.3f}"
                                     f" · Bias {d['Bias']:.2f}<extra>{lab}</extra>"))
    f.update_xaxes(title="başarı oranı SR = 1 − FAR", range=[0, 1], constrain="domain")
    f.update_yaxes(title="yakalama POD", range=[0, 1], scaleanchor="x", scaleratio=1, constrain="domain")
    return f


def reliability(items: list, bins=10, height=520) -> go.Figure:
    """Güvenilirlik diyagramı. items: [(etiket, renk, y, p)]. Altta skor histogramı (log)."""
    from plotly.subplots import make_subplots

    from src.helper.evaluate import brier, reliability as rel
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.04)
    f.update_layout(**T.plotly_layout(title=dict(text="Güvenilirlik diyagramı"), height=height))
    f.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.LINE, dash="dash", width=1), showlegend=False,
                  hoverinfo="skip", row=1, col=1)
    edges = np.linspace(0, 1, bins + 1)
    for lab, col, y, p in items:
        r = rel(y, p, bins)
        ok = r["n"] > 0
        f.add_scatter(x=r.loc[ok, "ortalama skor"], y=r.loc[ok, "gerçekleşen oran"], mode="lines+markers",
                      name=f"{lab} · Brier {brier(y, p):.3f}", line=dict(color=col, width=2.2), marker=dict(size=6),
                      customdata=r.loc[ok, "n"],
                      hovertemplate="skor %{x:.2f}<br>gerçekleşen %{y:.3f}<br>%{customdata} kare<extra>" + lab + "</extra>",
                      row=1, col=1)
        cnt, _ = np.histogram(p, bins=edges)
        f.add_scatter(x=np.repeat(edges, 2)[1:-1], y=np.repeat(cnt, 2), mode="lines", line=dict(color=col, width=1.5),
                      showlegend=False, hoverinfo="skip", row=2, col=1)
    f.update_yaxes(title="gerçekleşen hortum oranı", range=[0, 1.01], row=1, col=1)
    f.update_yaxes(title="kare", type="log", row=2, col=1)
    f.update_xaxes(range=[0, 1], row=1, col=1)
    f.update_xaxes(title="model skoru", range=[0, 1], row=2, col=1)
    return f


def ef_box(groups: list, height=440) -> go.Figure:
    """EF'ye göre skor dağılımı. groups: [(etiket, renk, sınıflar, skorlar, eşik)] (sınıf: EF0 … EF3+, WRN, NUL)."""
    from src.helper.evaluate import EF_ORDER
    f = _fig(title=dict(text="EF'ye göre skor dağılımı"), height=height, boxmode="group")
    for lab, col, cls, p, thr in groups:
        keep = np.isin(cls, EF_ORDER)
        f.add_box(x=cls[keep], y=p[keep], name=f"{lab} · eşik {thr:.2f}", marker_color=col, boxpoints=False,
                  line=dict(width=1.5), fillcolor=_rgba(col, 0.25))
    f.update_xaxes(title="sınıf", categoryorder="array", categoryarray=EF_ORDER)
    f.update_yaxes(title="model skoru", range=[0, 1.01])
    return f


def range_lines(tables: list, height=400) -> go.Figure:
    """Radara uzaklığa göre ROC-AUC (sol) ve POD (sağ). tables: [(etiket, renk, range_table)]."""
    from plotly.subplots import make_subplots
    f = make_subplots(rows=1, cols=2, subplot_titles=("ROC-AUC", "POD (seçilen eşikte)"), horizontal_spacing=0.08)
    f.update_layout(**T.plotly_layout(title=dict(text="Radara uzaklığa göre başarım"), height=height))
    for lab, col, t in tables:
        x = [f"{i}<br>{h} hortum" for i, h in zip(t.index, t["hortum"])]
        for c, k in ((1, "ROC-AUC"), (2, "POD")):
            f.add_scatter(x=x, y=t[k], mode="lines+markers", name=lab, legendgroup=lab, showlegend=c == 1,
                          line=dict(color=col, width=2.2), marker=dict(size=7),
                          hovertemplate="%{x}<br>" + k + " %{y:.3f}<extra>" + lab + "</extra>", row=1, col=c)
    f.update_xaxes(title="pencere merkezinin radara uzaklığı", title_font=dict(size=11))
    f.update_layout(legend=dict(orientation="h", x=0, y=-0.38, yanchor="top"), margin=dict(b=120))
    return f


# ---------------------------------------------------------------- küçük radar resimleri (Hata Analizi)

def _ramp(stops):
    v = np.array([s[0] for s in stops], dtype=np.float32)
    c = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for _, h in stops], dtype=np.float32)
    return v, c


_VEL = _ramp([(-60, "#065C33"), (-40, "#0FA552"), (-22, "#5CD38C"), (-6, "#C9EBD5"), (0, "#D7DBE1"), (6, "#F4D0CE"),
              (22, "#F0867F"), (40, "#DA2B27"), (60, "#7E0D12")])
_DBZ_STEPS = [(-20, "#DCE2EA"), (-5, "#C3CCD8"), (5, "#04B9C7"), (10, "#0D84E0"), (15, "#2A3FE0"), (20, "#11D44A"), (25, "#0BA33A"),
              (30, "#087A2C"), (35, "#F2E22B"), (40, "#E0B417"), (45, "#F28A12"), (50, "#EE2B22"), (55, "#C11A1A"), (60, "#E23BD9")]
_WID = _ramp([(0, "#F1EDF9"), (3, "#CDBDEB"), (6, "#9C79D8"), (9, "#7539A8"), (12, "#B8306A"), (16, "#6E0D2B")])
_CAM = _ramp([(0, "#E6ECF4"), (0.3, "#8FB1DA"), (0.55, "#F0A04B"), (0.8, "#DD4A2A"), (1, "#8A1A0E")])
NODATA = np.array([0xE9, 0xED, 0xF2], dtype=np.float32)


def _lut(var: str) -> np.ndarray:
    """uint8 değer → RGB tablosu (256 × 3); 0 = boş piksel."""
    lo, hi = CHANNEL_RANGES[var]
    vals = (np.arange(256, dtype=np.float32) - 1) / 254 * (hi - lo) + lo
    if var == "DBZ":
        th = np.array([s[0] for s in _DBZ_STEPS])
        cols = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for _, h in _DBZ_STEPS], dtype=np.float32)
        out = cols[np.clip(np.searchsorted(th, vals, side="right") - 1, 0, len(th) - 1)]
    else:
        v, c = _VEL if var == "VEL" else _WID
        out = np.stack([np.interp(vals, v, c[:, k]) for k in range(3)], 1)
    out[0] = NODATA
    return out


def radar_rgb(q: np.ndarray, var: str, cam: np.ndarray | None = None) -> np.ndarray:
    """CNN görünümü (120 × 240) RGB; cam verilirse (0..1) üstüne Grad-CAM ısı haritası bindirilir."""
    rgb = _lut(var)[q]
    if cam is not None:
        v, c = _CAM
        heat = np.stack([np.interp(cam, v, c[:, k]) for k in range(3)], -1)
        a = np.clip(cam * 1.15, 0, 0.8)[..., None]
        rgb = rgb * 0.6 + 255 * 0.4                 # radar renkleri soluk kalır, ısı haritası öne çıkar
        rgb = rgb * (1 - a) + heat * a
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- Açıklanabilirlik

CHANNEL_COLORS = {"DBZ": "#0072B2", "VEL": "#E8950E", "WIDTH": "#7B5BE0"}
METHOD_COLORS = {"cam": "#D55E00", "ig": "#0072B2", "occ": "#009E73", "rnd": "#8791A1"}
METHOD_NAMES = {"cam": "Grad-CAM", "ig": "Integrated Gradients", "occ": "Occlusion", "rnd": "Rastgele sıra"}


def shap_waterfall(base: float, phi, variables, score: float, thr: float, height=330) -> go.Figure:
    """Kanal SHAP şelalesi: taban → +her kanal → skor (0–1 skor ölçeği)."""
    labels, bases, widths, colors, texts = ["taban"], [0.0], [base], [T.NUL], [f"{base:.3f}"]
    cur = base
    for v, d in zip(variables, phi):
        labels.append(f"{'+' if d >= 0 else '−'} {v}")
        bases.append(cur if d >= 0 else cur + d)
        widths.append(abs(d))
        colors.append(CHANNEL_COLORS.get(v, T.MUTED))
        texts.append(f"{d:+.3f}")
        cur += d
    labels.append("= skor"); bases.append(0.0); widths.append(score); colors.append(T.INK); texts.append(f"{score:.3f}")
    f = _fig(title=dict(text="Kanal katkısı · SHAP"), height=height, margin=dict(l=70, r=56, t=40, b=36), showlegend=False)
    f.add_bar(y=labels, x=widths, base=bases, orientation="h", marker=dict(color=colors), text=texts, textposition="outside",
              cliponaxis=False, textfont=dict(family="IBM Plex Mono, monospace", size=12),
              hovertemplate="%{y}: %{text}<extra></extra>")
    f.add_vline(x=thr, line=dict(color=T.INK, width=1.5, dash="dash"), annotation_text="eşik", annotation_position="top",
                annotation_font=dict(size=10, color=T.MUTED))
    f.update_yaxes(autorange="reversed", showgrid=False)
    f.update_xaxes(range=[0, 1.05], title="model skoru")
    return f


def deletion_plot(curves: dict, method: str, thr: float, height=320) -> go.Figure:
    """curves: {yöntem: (silinen oran, skor)}; seçili yöntem kalın çizilir."""
    f = _fig(title=dict(text="Sadakat testi · silme eğrisi"), height=height, margin=dict(l=52, r=16, t=40, b=96),
             legend=dict(orientation="h", x=0, y=-0.32, yanchor="top", font=dict(size=11)))
    for k in ("ig", "occ", "cam", "rnd"):
        if k not in curves:
            continue
        x, y = curves[k]
        on = k == method
        f.add_scatter(x=np.asarray(x) * 100, y=y, mode="lines", name=METHOD_NAMES[k],
                      line=dict(color=METHOD_COLORS[k], width=3.2 if on else 1.6, dash="dash" if k == "rnd" else "solid"),
                      opacity=1 if on or k == "rnd" else 0.5,
                      hovertemplate="%{x:.0f}% silindi · skor %{y:.3f}<extra>" + METHOD_NAMES[k] + "</extra>")
    f.add_hline(y=thr, line=dict(color=T.INK, width=1, dash="dot"))
    f.update_xaxes(title="silinen piksel (%)", range=[0, 50], ticksuffix="")
    f.update_yaxes(title="model skoru", range=[0, 1.02])
    return f


def channel_share_bars(rows: list, variables, height=300) -> go.Figure:
    """rows: [(etiket, {kanal: pay})]; yatay yığılmış çubuklar, ortalama |SHAP| payı."""
    f = _fig(title=dict(text="Kanal katkısı · karar türüne göre (ortalama |SHAP| payı)"), height=height, barmode="stack",
             margin=dict(l=120, r=16, t=40, b=60), legend=dict(orientation="h", y=-0.16, x=0, traceorder="normal"))
    labels = [r[0] for r in rows]
    for v in variables:
        vals = [r[1].get(v, 0.0) for r in rows]
        f.add_bar(y=labels, x=vals, name=v, orientation="h", marker_color=CHANNEL_COLORS.get(v, T.MUTED),
                  text=[f"%{100 * x:.0f}" for x in vals], textposition="inside", insidetextanchor="start",
                  textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#FFFFFF" if v != "VEL" else T.INK),
                  hovertemplate="%{y} · " + v + " %{x:.1%}<extra></extra>")
    f.update_yaxes(autorange="reversed", showgrid=False)
    f.update_xaxes(range=[0, 1], tickformat=".0%")
    return f


def distance_hist(groups: list, method: str, height=300) -> go.Figure:
    """groups: [(etiket, renk, km dizisi)]; açıklama zirvesi ↔ en güçlü dönme mesafesi."""
    f = _fig(title=dict(text=f"Zirve ↔ en güçlü dönme · {METHOD_NAMES[method]}"), height=height, barmode="group",
             margin=dict(l=52, r=16, t=40, b=84), legend=dict(orientation="h", y=-0.3, x=0))
    for lab, col, d in groups:
        d = np.clip(np.asarray(d, dtype=float), 0, 20)
        f.add_histogram(x=d, xbins=dict(start=0, end=20.001, size=1), name=f"{lab} · {len(d)}", marker_color=col,
                        histnorm="probability", hovertemplate="%{x} km<br>oran %{y:.2f}<extra>" + lab + "</extra>")
    f.add_vline(x=3, line=dict(color=T.INK, width=1.2, dash="dash"), annotation_text="3 km", annotation_position="top right",
                annotation_font=dict(size=10))
    f.update_xaxes(title="mesafe (km, 20+ birleşik)", range=[0, 20.5])
    f.update_yaxes(title="oran", tickformat=".0%")
    return f


def sanity_bars(vals: dict, method: str, height=230) -> go.Figure:
    """vals: {yöntem: ortalama sıra korelasyonu (eğitilmiş vs rastgele ağırlık)}."""
    keys = [k for k in ("cam", "ig", "occ") if k in vals]
    f = _fig(title=dict(text="Sağlamlık kontrolü · rastgele ağırlıkla harita benzerliği"), height=height,
             margin=dict(l=150, r=50, t=40, b=40), showlegend=False)
    f.add_bar(y=[METHOD_NAMES[k] for k in keys], x=[vals[k] for k in keys], orientation="h",
              marker_color=[T.INK if k == method else T.NUL for k in keys], text=[f"{vals[k]:.2f}" for k in keys],
              textposition="outside", cliponaxis=False, textfont=dict(family="IBM Plex Mono, monospace", size=12),
              hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    f.update_xaxes(range=[min(0, *[vals[k] for k in keys]) - 0.05, 1], title="Spearman ρ (düşük = harita modele bağlı)")
    f.update_yaxes(autorange="reversed", showgrid=False)
    return f
