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
    f.update_xaxes(title="hortum olasılığı", range=[0, 1])
    f.update_yaxes(title="oran (kategori içinde)")
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
