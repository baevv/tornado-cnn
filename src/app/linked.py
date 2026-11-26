"""Bağlı radar görünümleri: büyük açıklama haritası + kanal küçük resimleri, ortak imleç (artı işareti).

İmleç herhangi bir görüntünün üstünde gezdirilince aynı (açı, mesafe) hücresi bütün görüntülerde işaretlenir ve
üstteki satırda o hücrenin fiziksel değerleri ile harita değeri yazılır. Tıklamak noktayı sabitler.
Kendi başına bir HTML parçasıdır (st.components.v1.html); Python'a değer döndürmez.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import streamlit.components.v1 as components

from src.helper.tornet import CHANNEL_RANGES

UNITS = {"DBZ": "dBZ", "VEL": "m/s", "WIDTH": "m/s", "KDP": "°/km", "RHOHV": "", "ZDR": "dB"}
MAIN_H = 420                 # büyük görüntünün en fazla yüksekliği (px); iframe yüksekliği buna göre sabit


def _png(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render(main: tuple, small: list, q: np.ndarray, variables, heat: np.ndarray, heat_name: str,
           rot=None, peak=None, az0=0.0, r0=0.0, daz=0.5, dr=0.25):
    """main: (başlık, RGB 120×240×3); small: [(alt yazı, RGB)]; q: uint8 (C, H, W); heat: 0..1 (H, W)."""
    data = {
        "main": {"title": main[0], "png": _png(main[1])},
        "small": [{"cap": c, "png": _png(img)} for c, img in small],
        "shape": list(q.shape),
        "q": base64.b64encode(np.ascontiguousarray(q, dtype=np.uint8).tobytes()).decode("ascii"),
        "heat": base64.b64encode(np.round(np.clip(heat, 0, 1) * 255).astype(np.uint8).tobytes()).decode("ascii"),
        "heatName": heat_name,
        "vars": list(variables),
        "ranges": {v: list(CHANNEL_RANGES[v]) for v in variables},
        "units": {v: UNITS.get(v, "") for v in variables},
        "rot": None if rot is None else [int(rot[0]), int(rot[1])],
        "peak": None if peak is None else [int(peak[0]), int(peak[1])],
        "az0": float(az0), "r0": float(r0), "daz": daz, "dr": dr, "mainH": MAIN_H,
    }
    height = 30 + MAIN_H + 50 + 12 + 175 + 26
    components.html(_HTML.replace("__DATA__", json.dumps(data)), height=height, scrolling=False)


_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
html,body{margin:0;background:transparent;color:#131A24;font-family:"IBM Plex Sans",system-ui,sans-serif}
.ro{display:flex;flex-wrap:wrap;align-items:center;gap:4px 12px;min-height:26px;font-family:"IBM Plex Mono",monospace;font-size:12px;color:#55606F;padding:0 2px 4px}
.ro b{color:#131A24;font-weight:500}
.ro .lg{margin-left:auto;display:flex;gap:12px;font-family:"IBM Plex Sans",sans-serif}
.ro .lg span{display:flex;align-items:center;gap:5px}
.ring{width:10px;height:10px;border:2px solid #131A24;border-radius:50%}
.dia{width:8px;height:8px;background:#131A24;transform:rotate(45deg)}
.main{display:flex;justify-content:center}
canvas{display:block;cursor:crosshair;touch-action:none}
.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
figure{margin:0;display:flex;flex-direction:column;align-items:center}
figcaption{font-size:12.5px;color:#55606F;margin-top:5px;text-align:center}
</style></head><body>
<div class="ro" id="ro"></div>
<div class="main"><canvas id="c0"></canvas></div>
<div class="row" id="row"></div>
<script>
(() => {
const D = __DATA__;
const [C, H, W] = D.shape;
const b64 = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
const Q = b64(D.q), HEAT = b64(D.heat);
const dpr = window.devicePixelRatio || 1;
const fmt = (v, d) => v.toFixed(d).replace('.', ',');
const ML = 54, MB = 40, MT = 6, MR = 8;          // büyük görüntünün eksen payları

const views = [];          // {canvas, img, rect: [x0, y0, w, h], axes}
function addView(canvas, png, axes) {
  const img = new Image(); img.src = 'data:image/png;base64,' + png;
  const v = {canvas, img, axes, rect: [0, 0, 1, 1]};
  img.onload = () => draw(v);
  views.push(v);
  canvas.addEventListener('pointermove', e => { const c = cellAt(v, e); hover = c; drawAll(); });
  canvas.addEventListener('pointerleave', () => { hover = null; drawAll(); });
  canvas.addEventListener('click', e => { const c = cellAt(v, e); pin = (pin && c && pin[0] === c[0] && pin[1] === c[1]) ? null : c; drawAll(); });
  return v;
}
addView(document.getElementById('c0'), D.main.png, true);
const row = document.getElementById('row');
D.small.forEach(s => {
  const f = document.createElement('figure'), c = document.createElement('canvas'), cap = document.createElement('figcaption');
  cap.textContent = s.cap; f.appendChild(c); f.appendChild(cap); row.appendChild(f);
  addView(c, s.png, false);
});

let hover = null, pin = null;

function size(v, cw, ch) {
  v.canvas.style.width = cw + 'px'; v.canvas.style.height = ch + 'px';
  v.canvas.width = Math.round(cw * dpr); v.canvas.height = Math.round(ch * dpr);
}
function layout() {
  const total = document.body.clientWidth;
  // büyük görüntü: 2:1, yükseklik en fazla mainH
  let pw = Math.min(total - ML - MR, 2 * D.mainH), ph = pw / 2;
  size(views[0], ML + pw + MR, MT + ph + MB);
  views[0].rect = [ML, MT, pw, ph];
  const sw = Math.min((total - 16) / 3, 340), sh = sw / 2;
  views.slice(1).forEach(v => { size(v, sw, sh); v.rect = [0, 0, sw, sh]; });
  drawAll();
}
function cellAt(v, e) {
  const b = v.canvas.getBoundingClientRect(), [x0, y0, w, h] = v.rect;
  const x = e.clientX - b.left - x0, y = e.clientY - b.top - y0;
  if (x < 0 || y < 0 || x >= w || y >= h) return null;
  return [Math.floor(y / h * H), Math.floor(x / w * W)];
}
function px(v, c) { const [x0, y0, w, h] = v.rect; return [x0 + (c[1] + 0.5) * w / W, y0 + (c[0] + 0.5) * h / H]; }

function cross(ctx, v, c, solid) {
  const [x0, y0, w, h] = v.rect, [x, y] = px(v, c);
  for (const [col, lw] of [['rgba(255,255,255,.85)', 3], [solid ? '#131A24' : 'rgba(19,26,36,.8)', 1.2]]) {
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(solid ? [] : [4, 3]);
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + w, y); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + h); ctx.stroke();
  }
  ctx.setLineDash([]);
}
function marker(ctx, v, c, kind) {
  const [x, y] = px(v, c), s = v.axes ? 1 : 0.6;
  ctx.lineWidth = 2.5 * s;
  if (kind === 'ring') {
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 4.5 * s; ctx.beginPath(); ctx.arc(x, y, 10 * s, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = '#131A24'; ctx.lineWidth = 2.5 * s; ctx.beginPath(); ctx.arc(x, y, 10 * s, 0, 2 * Math.PI); ctx.stroke();
  } else {
    const r = 6.5 * s;
    ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
    ctx.fillStyle = '#131A24'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * s; ctx.stroke();
  }
}
function axes(ctx, v) {
  const [x0, y0, w, h] = v.rect;
  ctx.fillStyle = '#55606F'; ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.strokeStyle = '#D6DCE4'; ctx.lineWidth = 1;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let j = 0; j <= W; j += 50) { const x = x0 + j * w / W; ctx.beginPath(); ctx.moveTo(x, y0 + h); ctx.lineTo(x, y0 + h + 4); ctx.stroke(); ctx.fillText(j, x, y0 + h + 6); }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i <= H; i += 20) { const y = y0 + i * h / H; ctx.beginPath(); ctx.moveTo(x0 - 4, y); ctx.lineTo(x0, y); ctx.stroke(); ctx.fillText(i, x0 - 6, y); }
  ctx.font = '12px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.fillText('menzil hücresi (0,25 km) →', x0 + w / 2, y0 + h + MB - 2);
  ctx.save(); ctx.translate(14, y0 + h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle';
  ctx.fillText('azimut hücresi (0,5°)', 0, 0); ctx.restore();
}
function draw(v) {
  const ctx = v.canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, v.canvas.width, v.canvas.height);
  const [x0, y0, w, h] = v.rect;
  if (v.img.complete && v.img.naturalWidth) { ctx.imageSmoothingEnabled = false; ctx.drawImage(v.img, x0, y0, w, h); }
  ctx.strokeStyle = '#D6DCE4'; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
  if (v.axes) axes(ctx, v);
  if (D.rot) marker(ctx, v, D.rot, 'ring');
  if (D.peak) marker(ctx, v, D.peak, 'dia');
  if (pin) cross(ctx, v, pin, true);
  if (hover && !(pin && hover[0] === pin[0] && hover[1] === pin[1])) cross(ctx, v, hover, false);
}
function value(ci, c) {
  const u = Q[(ci * H + c[0]) * W + c[1]];
  if (u === 0) return null;
  const [lo, hi] = D.ranges[D.vars[ci]];
  return (u - 1) / 254 * (hi - lo) + lo;
}
function readout() {
  const c = hover || pin;
  const lg = '<span class="lg"><span><i class="ring"></i>en güçlü dönme</span><span><i class="dia"></i>açıklama zirvesi</span></span>';
  if (!c) { document.getElementById('ro').innerHTML = '<span>İmleci bir görüntünün üstünde gezdir; tıklayınca nokta sabitlenir.</span>' + lg; return; }
  const az = D.az0 + (c[0] + 0.5) * D.daz, r = D.r0 + (c[1] + 0.5) * D.dr;
  let s = `<span>hücre <b>${c[0]}, ${c[1]}</b></span><span>az <b>${fmt(((az % 360) + 360) % 360, 1)}°</b> · menzil <b>${fmt(r, 1)} km</b></span>`;
  D.vars.forEach((v, i) => { const x = value(i, c); s += `<span>${v} <b>${x === null ? 'boş' : fmt(x, 1) + ' ' + D.units[v]}</b></span>`; });
  s += `<span>${D.heatName} <b>${fmt(HEAT[c[0] * W + c[1]] / 255, 2)}</b></span>`;
  document.getElementById('ro').innerHTML = s + (pin && !hover ? '<span>(sabit)</span>' : '') + lg;
}
function drawAll() { views.forEach(draw); readout(); }
new ResizeObserver(layout).observe(document.body);
layout();
})();
</script></body></html>"""
