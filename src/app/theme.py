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
/* Açıklanabilirlik: kare bilgi şeridi, özet kutuları, yorum maddeleri */
.xs {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 6px; background: {PANEL}; border: 2px solid var(--st, {LINE}); border-radius: 8px; padding: 5px; }}
.xs .pl {{ display: flex; flex-direction: column; justify-content: center; background: var(--st); color: #fff; border-radius: 6px; padding: 3px 12px; }}
.xs .pl b {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 22px; line-height: 1; text-transform: uppercase; }}
.xs .pl span {{ font-family: {FONT_MONO}; font-size: 10.5px; }}
.xs .c {{ display: flex; flex-direction: column; justify-content: center; padding: 0 12px; border-left: 1px solid {LINE_SOFT}; }}
.xs .c:first-of-type {{ border-left: 0; }}
.xs .k, .xk .k {{ font-size: 10px; letter-spacing: .09em; text-transform: uppercase; color: #5E6878; line-height: 1.2; }}
.xs .v {{ font-family: {FONT_MONO}; font-size: 12.5px; white-space: nowrap; }}
.xs .pr {{ display: flex; margin-left: auto; background: {PANEL_2}; border-radius: 6px; padding: 3px 0; }}
.xs .pr .c {{ border-left-color: {LINE}; }}
.xs .pr .v {{ font-size: 15px; font-weight: 500; }}
.xbox {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; box-shadow: 0 1px 2px rgba(19,26,36,.05); padding: 10px 14px; }}
.xk {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 6px 0 10px; }}
.xk > div {{ background: {PANEL_2}; border-radius: 6px; padding: 8px 10px; }}
.xk .n {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 28px; line-height: 1; }}
.xk .s {{ font-size: 11.5px; color: {MUTED}; }}
.xb {{ display: flex; gap: 10px; align-items: flex-start; margin: 6px 0; font-size: 13px; color: #2A3340; line-height: 1.45; }}
.xb i {{ width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; flex: none; }}
.xd {{ font-size: 12.5px; color: #3E4857; margin: 2px 2px 0; }}
.xsplit {{ display: grid; grid-template-columns: minmax(250px, 0.85fr) 2fr; gap: 16px; align-items: start; margin-top: 6px; }}
.xsplit .xk {{ grid-template-columns: 1fr; margin: 0; gap: 6px; }}
.xsplit .xk > div {{ display: grid; grid-template-columns: 1fr auto; align-items: baseline; column-gap: 10px; padding: 7px 10px; }}
.xsplit .xk .k {{ grid-column: 1; }}
.xsplit .xk .n {{ grid-column: 2; grid-row: 1 / span 2; font-size: 24px; text-align: right; align-self: center; }}
.xsplit .xk .s {{ grid-column: 1; }}
.xsplit .xc {{ border-left: 1px solid {LINE}; padding-left: 16px; }}
.xsplit .xc .xb {{ font-size: 14px; margin: 4px 0 10px; }}
.xsplit .xc .xb:first-child {{ margin-top: 2px; }}
@media (max-width: 900px) {{ .xsplit {{ grid-template-columns: 1fr; }} .xsplit .xc {{ border-left: none; padding-left: 0; }} }}
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
