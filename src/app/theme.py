"""Görsel tema: renkler, yazı tipleri ve Streamlit sayfasına eklenen CSS.

Renkler onaylı Fırtına Gezgini maketiyle aynıdır (frontend/style.css :root).
"""
import streamlit as st

GROUND = "#EDF0F4"
PANEL = "#FFFFFF"
PANEL_2 = "#F4F6F9"
LINE = "#D6DCE4"
LINE_SOFT = "#E7EBF0"
INK = "#131A24"
MUTED = "#55606F"
FAINT = "#8791A1"
ACCENT = "#E8950E"
TOR = "#E3344E"
WRN = "#7B5BE0"
NUL = "#94A0B2"
GOOD = "#11906A"

CATEGORY_COLORS = {"TOR": TOR, "WRN": WRN, "NUL": NUL}

FONT_DISPLAY = '"Big Shoulders Display", "Oswald", "Arial Narrow", sans-serif'
FONT_BODY = '"IBM Plex Sans", "Inter", system-ui, sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace'

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
html, body, [data-testid="stAppViewContainer"] {{ background: {GROUND}; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {{ display: none !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding: 12px 16px 18px !important; max-width: none !important; }}
[data-testid="stVerticalBlock"] {{ gap: 8px; }}
html, body, .stMarkdown p, [data-testid="stCaptionContainer"] {{ font-family: {FONT_BODY}; }}
iframe[title="src.app.shell.firtina_shell"], iframe[title*="firtina_shell"] {{ display: block; }}
/* Plotly ve tablo kartları maketteki panellerle aynı görünür */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], .panel-box {{
  background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; box-shadow: 0 1px 2px rgba(19,26,36,.05); padding: 6px;
}}
.h-panel {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 19px; letter-spacing: .03em; margin: 2px 0 0 2px; color: {INK}; }}
.note {{ color: {MUTED}; font-size: 13px; }}
.note code {{ font-family: {FONT_MONO}; font-size: 12px; background: {PANEL_2}; padding: 1px 5px; border-radius: 4px; }}
[data-testid="stImage"] img {{ border-radius: 6px; border: 1px solid {LINE}; }}
.stButton > button {{ border-radius: 999px; border: 1px solid {LINE}; background: {PANEL}; }}
.stButton > button:hover {{ border-color: {ACCENT}; color: {INK}; }}
</style>
"""


def inject():
    st.html(_CSS)


def plotly_layout(**kw) -> dict:
    """Bütün Plotly grafiklerinde ortak düzen."""
    base = dict(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=12, color=INK),
        margin=dict(l=52, r=16, t=40, b=44),
        title=dict(font=dict(family="Big Shoulders Display, Arial Narrow, sans-serif", size=19, color=INK), x=0.01, y=0.97),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#FFFFFF", family="IBM Plex Mono, monospace", size=11), bordercolor=INK),
        xaxis=dict(gridcolor=LINE_SOFT, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(family="IBM Plex Mono, monospace", size=10, color=MUTED)),
        yaxis=dict(gridcolor=LINE_SOFT, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(family="IBM Plex Mono, monospace", size=10, color=MUTED)),
    )
    base.update(kw)
    return base
