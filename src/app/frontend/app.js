/* Fırtına Gezgini: Streamlit çift yönlü bileşeni (derleme gerektirmez).
   Python her çalıştırmada args gönderir (render); kullanıcı eylemleri setComponentValue ile Python'a döner.
   Kare oynatma, imleç, işaret ve model seçimi tarayıcıda çalışır; fırtına/sekme değişimi Python'a gider. */
(() => {
'use strict';

// ---------------------------------------------------------------- Streamlit protokolü
const post = (type, data) => window.parent.postMessage({isStreamlitMessage: true, type, ...data}, '*');
let evN = 0;
const send = (a, extra = {}) => post('streamlit:setComponentValue', {value: {n: Date.now() * 100 + (evN++ % 100), a, ...extra}, dataType: 'json'});
let lastH = 0;
const fitHeight = () => {
  const h = Math.ceil(document.getElementById('app').getBoundingClientRect().height) + 6;
  if (Math.abs(h - lastH) > 1) { lastH = h; post('streamlit:setFrameHeight', {height: h}); }
};
new ResizeObserver(() => { fitHeight(); }).observe(document.body);
window.addEventListener('message', e => { if (e.data && e.data.type === 'streamlit:render') render(e.data.args.state); });
post('streamlit:componentReady', {apiVersion: 1});
try { if (window.frameElement) window.frameElement.removeAttribute('title'); } catch (_) { /* farklı köken */ }

// ---------------------------------------------------------------- yardımcılar
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const rad = d => d * Math.PI / 180;
const toXY = (az, r) => [r * Math.sin(rad(az)), r * Math.cos(rad(az))];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtN = n => Number(n).toLocaleString('tr-TR');
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const ico = (id, style = '') => `<svg class="ico"${style ? ` style="${style}"` : ''}><use href="#i-${id}"/></svg>`;
const b64 = s => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

// ---------------------------------------------------------------- renk haritaları (meteorolojik)
function ramp(stops) {
  const s = stops.map(([v, h]) => [v, hex(h)]);
  return v => {
    if (v <= s[0][0]) return s[0][1];
    for (let k = 1; k < s.length; k++) if (v <= s[k][0]) {
      const t = (v - s[k - 1][0]) / (s[k][0] - s[k - 1][0]), a = s[k - 1][1], b = s[k][1];
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    }
    return s[s.length - 1][1];
  };
}
function steps(stops) {
  const s = stops.map(([v, h]) => [v, hex(h)]);
  return v => { let c = s[0][1]; for (const [t, col] of s) if (v >= t) c = col; return c; };
}
const CH = {
  DBZ: {name: 'DBZ', unit: 'dBZ', cmap: steps([[-20, '#DCE2EA'], [-5, '#C3CCD8'], [5, '#04B9C7'], [10, '#0D84E0'], [15, '#2A3FE0'], [20, '#11D44A'], [25, '#0BA33A'], [30, '#087A2C'], [35, '#F2E22B'], [40, '#E0B417'], [45, '#F28A12'], [50, '#EE2B22'], [55, '#C11A1A'], [60, '#E23BD9']])},
  VEL: {name: 'VEL', unit: 'm/s', cmap: ramp([[-60, '#065C33'], [-40, '#0FA552'], [-22, '#5CD38C'], [-6, '#C9EBD5'], [0, '#D7DBE1'], [6, '#F4D0CE'], [22, '#F0867F'], [40, '#DA2B27'], [60, '#7E0D12']])},
  WIDTH: {name: 'WIDTH', unit: 'm/s', cmap: ramp([[0, '#F1EDF9'], [3, '#CDBDEB'], [6, '#9C79D8'], [9, '#7539A8'], [12, '#B8306A'], [16, '#6E0D2B']])},
  CAM: {name: 'GRAD-CAM', unit: 'önem', lo: 0, hi: 1, cmap: ramp([[0, '#E6ECF4'], [0.3, '#8FB1DA'], [0.55, '#F0A04B'], [0.8, '#DD4A2A'], [1, '#8A1A0E']])},
};
const NODATA = hex('#E9EDF2');
const MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';

// ---------------------------------------------------------------- durum
let A = null;               // son args
let EP = null;              // çözülmüş fırtına
const S = {frame: 0, stage: 'VEL', view: 'both', hover: null, pin: null, playing: false, sel: null, tab: null};
let timer = null;

// ---------------------------------------------------------------- fırtına verisi
function decodeCam(e, F, N) {
  if (!e.cam) return null;
  const cx = b64(e.cam), cam = {frames: []};
  for (let f = 0; f < F; f++) cam.frames.push(cx.subarray(f * N, (f + 1) * N));
  const col = new Uint8ClampedArray(256 * 3);
  for (let q = 0; q < 256; q++) { const c = CH.CAM.cmap(q / 255); col[q * 3] = c[0]; col[q * 3 + 1] = c[1]; col[q * 3 + 2] = c[2]; }
  cam.col = col;
  return cam;
}
function decodeEpisode(e) {
  const [F, C, NA, NR] = e.shape, x = b64(e.x), vars = e.variables;
  const N = NA * NR;
  const chans = {};
  vars.forEach((v, ci) => {
    const [lo, hi] = e.ranges[v];
    const val = new Float32Array(256); val[0] = NaN;
    for (let q = 1; q < 256; q++) val[q] = (q - 1) / 254 * (hi - lo) + lo;
    const col = new Uint8ClampedArray(256 * 3);
    for (let q = 0; q < 256; q++) { const c = q === 0 ? NODATA : CH[v].cmap(val[q]); col[q * 3] = c[0]; col[q * 3 + 1] = c[1]; col[q * 3 + 2] = c[2]; }
    chans[v] = {ci, lo, hi, val, col, frames: []};
    for (let f = 0; f < F; f++) chans[v].frames.push(x.subarray((f * C + ci) * N, (f * C + ci + 1) * N));
  });
  const cam = decodeCam(e, F, N);
  const ep = {key: e.key, F, NA, NR, DAZ: e.daz, DR: e.dr, frames: e.frames, chans, cam, camModel: e.cam_model, vars, rot: e.rot, meta: e.meta};
  // Ortak sınır kutusu (oynatırken görünüm kaymasın); fırtına çok yer değiştiriyorsa kare kare
  const boxOf = fr => {
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    const rA = Math.max(fr.r0, 0), rB = fr.r0 + NR * ep.DR;
    for (let a = 0; a <= NA * ep.DAZ; a += 1) for (const r of [rA, (rA + rB) / 2, rB]) {
      const [x, y] = toXY(fr.az0 + a, r); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    if (fr.r0 < 0) { x0 = Math.min(x0, 0); x1 = Math.max(x1, 0); y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
    return [x0, x1, y0, y1];
  };
  const boxes = ep.frames.map(boxOf);
  const u = boxes.reduce((b, c) => [Math.min(b[0], c[0]), Math.max(b[1], c[1]), Math.min(b[2], c[2]), Math.max(b[3], c[3])]);
  const area = b => (b[1] - b[0]) * (b[3] - b[2]);
  // Ortak kutu yalnız her kare onu iyi dolduruyorsa: radara yakın fırtınalarda kareler çok farklı olduğundan
  // ortak kutu en küçük kareyi köşede küçücük bırakırdı (en küçük kareye göre karşılaştırılır).
  const smallest = Math.min(...boxes.map(area));
  ep.boxes = area(u) <= 1.6 * smallest ? boxes.map(() => u) : boxes;
  ep.proj = {};
  return ep;
}
const val = (ch, f, k) => EP.chans[ch].val[EP.chans[ch].frames[f][k]];

// ---------------------------------------------------------------- projeksiyon (gerçek x, y km)
function makeProj(W, H, pad, f) {
  const key = `${W}x${H}:${f}`; if (EP.proj[key]) return EP.proj[key];
  const fr = EP.frames[f], [x0, x1, y0, y1] = EP.boxes[f];
  const bw = x1 - x0, bh = y1 - y0, s = Math.min((W - pad[1] - pad[3]) / bw, (H - pad[0] - pad[2]) / bh);
  const ox = pad[3] + ((W - pad[1] - pad[3]) - bw * s) / 2, oy = pad[0] + ((H - pad[0] - pad[2]) - bh * s) / 2;
  const az0 = ((fr.az0 % 360) + 360) % 360;
  const inv = (px, py) => {
    const x = x0 + (px - ox) / s, y = y1 - (py - oy) / s;
    let az = Math.atan2(x, y) * 180 / Math.PI; if (az < 0) az += 360;
    const r = Math.hypot(x, y), d = ((az - az0) % 360 + 360) % 360;
    const i = Math.floor(d / EP.DAZ), j = Math.floor((r - fr.r0) / EP.DR);
    return {x, y, az, r, i, j, ok: i >= 0 && i < EP.NA && j >= 0 && j < EP.NR};
  };
  const lut = new Int32Array(W * H).fill(-1);
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) { const q = inv(px + .5, py + .5); if (q.ok) lut[py * W + px] = q.i * EP.NR + q.j; }
  return (EP.proj[key] = {W, H, s, lut, inv, fwd: (x, y) => [ox + (x - x0) * s, oy + (y1 - y) * s]});
}
function cellPoint(i, j) {
  const fr = EP.frames[S.frame], az = fr.az0 + (i + .5) * EP.DAZ, r = fr.r0 + (j + .5) * EP.DR, [x, y] = toXY(az, r);
  return {i, j, az: ((az % 360) + 360) % 360, r, x, y};
}

function paint(ctx, proj, ch, f) {
  const img = ctx.createImageData(proj.W, proj.H), d = img.data, lut = proj.lut;
  if (ch === 'CAM') {
    const z = EP.chans.DBZ ? EP.chans.DBZ.frames[f] : null, c = EP.cam.frames[f], col = EP.cam.col;
    for (let p = 0; p < lut.length; p++) {
      const k = lut[p]; if (k < 0) continue;
      const zq = z ? z[k] : 0, g = zq === 0 ? NODATA : [236 - zq * .47, 238 - zq * .455, 242 - zq * .44];
      const cq = c[k], a = Math.min(cq / 255 * 1.15, .85);
      d[p * 4] = g[0] * (1 - a) + col[cq * 3] * a; d[p * 4 + 1] = g[1] * (1 - a) + col[cq * 3 + 1] * a; d[p * 4 + 2] = g[2] * (1 - a) + col[cq * 3 + 2] * a; d[p * 4 + 3] = 255;
    }
  } else {
    const q = EP.chans[ch].frames[f], col = EP.chans[ch].col;
    for (let p = 0; p < lut.length; p++) { const k = lut[p]; if (k < 0) continue; const o = q[k] * 3; d[p * 4] = col[o]; d[p * 4 + 1] = col[o + 1]; d[p * 4 + 2] = col[o + 2]; d[p * 4 + 3] = 255; }
  }
  const tmp = document.createElement('canvas'); tmp.width = proj.W; tmp.height = proj.H;
  tmp.getContext('2d').putImageData(img, 0, 0); ctx.drawImage(tmp, 0, 0);
}
function paintRaw(ch, f) {   // CNN girdisi: 240 × 120 ham dizi
  const raw = document.createElement('canvas'); raw.width = EP.NR; raw.height = EP.NA;
  const rx = raw.getContext('2d'), img = rx.createImageData(EP.NR, EP.NA), d = img.data, N = EP.NA * EP.NR;
  if (ch === 'CAM') {
    const z = EP.chans.DBZ ? EP.chans.DBZ.frames[f] : null, c = EP.cam.frames[f], col = EP.cam.col;
    for (let k = 0; k < N; k++) {
      const zq = z ? z[k] : 0, g = zq === 0 ? NODATA : [236 - zq * .47, 238 - zq * .455, 242 - zq * .44], cq = c[k], a = Math.min(cq / 255 * 1.15, .85);
      d[k * 4] = g[0] * (1 - a) + col[cq * 3] * a; d[k * 4 + 1] = g[1] * (1 - a) + col[cq * 3 + 1] * a; d[k * 4 + 2] = g[2] * (1 - a) + col[cq * 3 + 2] * a; d[k * 4 + 3] = 255;
    }
  } else {
    const q = EP.chans[ch].frames[f], col = EP.chans[ch].col;
    for (let k = 0; k < N; k++) { const o = q[k] * 3; d[k * 4] = col[o]; d[k * 4 + 1] = col[o + 1]; d[k * 4 + 2] = col[o + 2]; d[k * 4 + 3] = 255; }
  }
  rx.putImageData(img, 0, 0); return raw;
}

function grid(ctx, proj, f, labels) {
  const fr = EP.frames[f], rA = Math.max(fr.r0, 0), rB = fr.r0 + EP.NR * EP.DR, aA = fr.az0, aB = fr.az0 + EP.NA * EP.DAZ;
  ctx.save(); ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(40,55,80,.16)'; ctx.setLineDash([3, 5]);
  const r1 = Math.ceil(rA / 10) * 10, a1 = Math.ceil(aA / 10) * 10;
  for (let r = r1; r <= rB; r += 10) { if (r <= 0) continue; ctx.beginPath(); for (let a = aA - 4; a <= aB + 4; a += 1) { const p = proj.fwd(...toXY(a, r)); a === aA - 4 ? ctx.moveTo(...p) : ctx.lineTo(...p); } ctx.stroke(); }
  for (let a = a1; a <= aB; a += 10) { const p1 = proj.fwd(...toXY(a, Math.max(rA - 3, 0))), p2 = proj.fwd(...toXY(a, rB + 3)); ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.strokeStyle = 'rgba(40,55,80,.35)'; ctx.beginPath();
  for (let a = aA; a <= aB + 1e-6; a += .5) { const p = proj.fwd(...toXY(a, rB)); a === aA ? ctx.moveTo(...p) : ctx.lineTo(...p); }
  for (let a = aB; a >= aA - 1e-6; a -= .5) ctx.lineTo(...proj.fwd(...toXY(a, rA)));
  ctx.closePath(); ctx.stroke();
  if (labels) {
    ctx.fillStyle = css('--muted'); ctx.font = '500 21px ' + MONO; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    // Mesafe etiketleri: sektörün iki kenarından etiketleri daha seyrek düşeni seçilir, kenarın dışına yazılır
    const rings = []; for (let r = r1; r <= rB; r += 10) if (r > 0) rings.push(r);
    const place = (edge, out) => rings.map(r => {
      const p = proj.fwd(...toXY(edge, r)), q = proj.fwd(...toXY(edge + out, r));
      const dx = q[0] - p[0], dy = q[1] - p[1], n = Math.hypot(dx, dy) || 1;
      return [p[0] + dx / n * 34, p[1] + dy / n * 34];
    });
    const gap = pts => { let g = 1e9; for (let k = 1; k < pts.length; k++) g = Math.min(g, Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])); return g; };
    const inside = pts => pts.every(([x, y]) => x > 20 && x < proj.W - 20 && y > 12 && y < proj.H - 12);
    const pa = place(aA, -3), pb = place(aB, 3);
    const score = pts => gap(pts) + (inside(pts) ? 0 : -1000);
    const pts = score(pa) >= score(pb) ? pa : pb;
    ctx.textAlign = 'center';
    rings.forEach((r, k) => ctx.fillText(k === rings.length - 1 ? r + ' km' : r, pts[k][0], pts[k][1]));
    // Açı etiketleri sektörün dışına yazılır; sektör tuvali doldurunca kenardan taşmasın diye içeri kıstırılır
    for (let a = a1; a <= aB; a += 10) { const p = proj.fwd(...toXY(a, rB + 7)); ctx.fillText((((a % 360) + 360) % 360) + '°', clamp(p[0], 30, proj.W - 30), clamp(p[1], 14, proj.H - 14)); }
    // radara doğru ok
    const cx = toXY((aA + aB) / 2, (rA + rB) / 2), ang = Math.atan2(cx[1], -cx[0]);   // ekran: +x sağ, +y aşağı
    const ax = proj.W - 70, ay = 26, L = 15;
    ctx.strokeStyle = css('--faint'); ctx.fillStyle = css('--faint'); ctx.lineWidth = 2.2;
    const ex = ax + Math.cos(ang) * L, ey = ay + Math.sin(ang) * L;
    ctx.beginPath(); ctx.moveTo(ax - Math.cos(ang) * L, ay - Math.sin(ang) * L); ctx.lineTo(ex, ey); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - Math.cos(ang - .5) * 9, ey - Math.sin(ang - .5) * 9); ctx.lineTo(ex - Math.cos(ang + .5) * 9, ey - Math.sin(ang + .5) * 9); ctx.closePath(); ctx.fill();
    ctx.textAlign = 'left'; ctx.font = '500 19px "IBM Plex Sans", sans-serif'; ctx.fillText('radar', ax + 22, ay + 1);
  }
  ctx.restore();
}

// ---------------------------------------------------------------- sahne
const cReal = $('c-real'), oReal = $('o-real'), cCnn = $('c-cnn'), oCnn = $('o-cnn');
const PAD = [40, 72, 40, 72];
const CR = {x: 66, y: 34, w: 820 - 66 - 18, h: 740 - 34 - 46};
const TW = 360, TH = 300;

function chStats(ch, f) {
  const q = EP.chans[ch].frames[f], v = EP.chans[ch].val; let lo = 255, hi = 0, nan = 0;
  for (let k = 0; k < q.length; k++) { const b = q[k]; if (b === 0) { nan++; continue; } if (b < lo) lo = b; if (b > hi) hi = b; }
  if (nan === q.length) return 'boş';
  return `${v[lo].toFixed(1)}…${v[hi].toFixed(1)} · boş %${(100 * nan / q.length).toFixed(0)}`;
}
const camName = () => { const m = A.models.find(m => m.key === EP.camModel); return m ? m.label : ''; };

function drawStage() {
  const f = S.frame, stage = S.stage;
  const ctx = cReal.getContext('2d'); ctx.fillStyle = css('--panel'); ctx.fillRect(0, 0, cReal.width, cReal.height);
  const P = makeProj(cReal.width, cReal.height, PAD, f); paint(ctx, P, stage, f); grid(ctx, P, f, true);
  const cx = cCnn.getContext('2d'); cx.fillStyle = css('--panel'); cx.fillRect(0, 0, cCnn.width, cCnn.height);
  cx.imageSmoothingEnabled = false; cx.drawImage(paintRaw(stage, f), CR.x, CR.y, CR.w, CR.h);
  cx.strokeStyle = 'rgba(40,55,80,.35)'; cx.lineWidth = 1; cx.strokeRect(CR.x + .5, CR.y + .5, CR.w - 1, CR.h - 1);
  cx.fillStyle = css('--muted'); cx.font = '500 21px ' + MONO; cx.textAlign = 'right'; cx.textBaseline = 'middle';
  const fr = EP.frames[f], span = EP.NA * EP.DAZ, rspan = EP.NR * EP.DR;
  for (let a = Math.ceil(fr.az0 / 10) * 10; a <= fr.az0 + span; a += 10) { const y = CR.y + (a - fr.az0) / span * CR.h; cx.fillText((((a % 360) + 360) % 360) + '°', CR.x - 8, y); cx.fillRect(CR.x - 4, y, 4, 1); }
  cx.textAlign = 'center'; cx.textBaseline = 'top';
  const rEnd = fr.r0 + rspan;
  for (let r = Math.ceil(fr.r0 / 10) * 10; r <= rEnd; r += 10) { const x = CR.x + (r - fr.r0) / rspan * CR.w; cx.fillText(r + 10 > rEnd ? r + ' km' : r, Math.min(x, CR.x + CR.w - 36), CR.y + CR.h + 8); cx.fillRect(x, CR.y + CR.h, 1, 4); }
  // renk çubuğu
  const lo = stage === 'CAM' ? 0 : EP.chans[stage].lo, hi = stage === 'CAM' ? 1 : EP.chans[stage].hi;
  const b = $('c-bar'), bx = b.getContext('2d');
  for (let x = 0; x < b.width; x++) { const c = CH[stage].cmap(lo + (hi - lo) * x / (b.width - 1)); bx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; bx.fillRect(x, 0, 1, b.height); }
  $('bar-lo').textContent = lo; $('bar-hi').textContent = hi + ' ' + CH[stage].unit;
  $('st-name').textContent = CH[stage].name;
  $('st-stats').textContent = stage === 'CAM' ? `${camName()} · DBZ üzerinde` : chStats(stage, f);
  drawOverlays();
}

function drawChannels() {
  const wrap = $('channels'); wrap.innerHTML = '';
  const all = [...EP.vars.filter(v => CH[v]), 'CAM'];
  for (const c of all.filter(c => c !== S.stage).slice(0, 3)) {
    const b = document.createElement('button'); b.className = 'ch';
    if (c === 'CAM' && !EP.cam) {
      b.classList.add('off'); b.disabled = true;
      b.innerHTML = `<span class="nm">GRAD-CAM</span><span>${A.torch ? 'model seç' : 'torch kurulu değil'}</span>`;
      wrap.appendChild(b); continue;
    }
    b.setAttribute('aria-label', CH[c].name + ' kanalını büyüt'); b.title = 'Büyüt';
    const cv = document.createElement('canvas'); cv.width = TW; cv.height = TH; cv.style.position = 'absolute'; cv.style.inset = '0';
    const ctx = cv.getContext('2d'); ctx.fillStyle = css('--panel'); ctx.fillRect(0, 0, TW, TH);
    const P = makeProj(TW, TH, [6, 6, 6, 6], S.frame); paint(ctx, P, c, S.frame); grid(ctx, P, S.frame, false);
    const pt = S.hover || S.pin;
    if (pt) { const p = P.fwd(pt.x, pt.y); ctx.strokeStyle = S.hover ? css('--ink') : css('--accent'); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 7); ctx.stroke(); }
    b.appendChild(cv);
    b.insertAdjacentHTML('beforeend', `<span class="nm">${CH[c].name}</span><span class="stt">${c === 'CAM' ? esc(camName()) : chStats(c, S.frame)}</span>`);
    b.onclick = () => { S.stage = c; drawStage(); drawChannels(); };
    wrap.appendChild(b);
  }
}

function drawOverlays() {
  const o = oReal.getContext('2d'); o.clearRect(0, 0, oReal.width, oReal.height);
  const oc = oCnn.getContext('2d'); oc.clearRect(0, 0, oCnn.width, oCnn.height);
  const P = makeProj(cReal.width, cReal.height, PAD, S.frame);
  const mark = (pt, col, hov) => {
    const [px, py] = P.fwd(pt.x, pt.y);
    o.save(); o.strokeStyle = col; o.lineWidth = 2;
    if (hov) { o.setLineDash([4, 4]); o.globalAlpha = .5; o.beginPath(); o.moveTo(px, 0); o.lineTo(px, oReal.height); o.moveTo(0, py); o.lineTo(oReal.width, py); o.stroke(); o.setLineDash([]); o.globalAlpha = 1; }
    o.lineWidth = 5; o.strokeStyle = 'rgba(255,255,255,.9)'; o.beginPath(); o.arc(px, py, 12, 0, 7); o.stroke();
    o.lineWidth = 2.5; o.strokeStyle = col; o.beginPath(); o.arc(px, py, 12, 0, 7); o.stroke(); o.beginPath(); o.arc(px, py, 3, 0, 7); o.fillStyle = col; o.fill(); o.restore();
    const cx = CR.x + (pt.j + .5) * CR.w / EP.NR, cy = CR.y + (pt.i + .5) * CR.h / EP.NA;
    oc.save(); oc.strokeStyle = col; oc.lineWidth = 2;
    if (hov) { oc.setLineDash([4, 4]); oc.globalAlpha = .5; oc.beginPath(); oc.moveTo(cx, CR.y); oc.lineTo(cx, CR.y + CR.h); oc.moveTo(CR.x, cy); oc.lineTo(CR.x + CR.w, cy); oc.stroke(); oc.setLineDash([]); oc.globalAlpha = 1; }
    oc.strokeStyle = 'rgba(255,255,255,.9)'; oc.lineWidth = 5; oc.strokeRect(cx - 10, cy - 10, 20, 20);
    oc.strokeStyle = col; oc.lineWidth = 2.5; oc.strokeRect(cx - 10, cy - 10, 20, 20); oc.restore();
  };
  if (S.pin) mark(S.pin, css('--accent'), false);
  if (S.hover) mark(S.hover, css('--ink'), true);
  const pt = S.hover || S.pin, ro = $('readout');
  if (!pt) { ro.innerHTML = ''; ro.hidden = true; return; }
  ro.hidden = false;
  const k = pt.i * EP.NR + pt.j;
  const cell = (key, v, cls = '', style = '') => `<div class="cell ${cls}"${style ? ` style="${style}"` : ''}><span class="k">${key}</span><span class="v">${v}</span></div>`;
  const vcell = c => {
    const ch = EP.chans[c], q = ch.frames[S.frame][k], v = ch.val[q], o = q * 3;
    return cell(c === 'WIDTH' ? 'WID' : c, isNaN(v) ? '—' : v.toFixed(1), 'val', `--sw:rgb(${ch.col[o]},${ch.col[o + 1]},${ch.col[o + 2]})`);
  };
  ro.style.setProperty('--pc', S.hover ? css('--ink') : css('--accent'));
  ro.innerHTML = cell(`${ico('crosshair')} ${S.hover ? 'İmleç' : 'İşaret'}`, `${pt.az.toFixed(1)}° · ${pt.r.toFixed(1)} km`) +
    cell('x, y km', `${pt.x.toFixed(1)}, ${pt.y.toFixed(1)}`) +
    EP.vars.filter(v => CH[v]).map(vcell).join('') + (EP.cam ? cell('CAM', (EP.cam.frames[S.frame][k] / 255).toFixed(2)) : '');
}
function evtPoint(e, canvas) { const b = canvas.getBoundingClientRect(); return [(e.clientX - b.left) * canvas.width / b.width, (e.clientY - b.top) * canvas.height / b.height]; }
oReal.addEventListener('pointermove', e => { if (!EP) return; const [px, py] = evtPoint(e, oReal); const q = makeProj(cReal.width, cReal.height, PAD, S.frame).inv(px, py); S.hover = q.ok ? cellPoint(q.i, q.j) : null; drawOverlays(); });
oCnn.addEventListener('pointermove', e => { if (!EP) return; const [px, py] = evtPoint(e, oCnn); const i = Math.floor((py - CR.y) / CR.h * EP.NA), j = Math.floor((px - CR.x) / CR.w * EP.NR); S.hover = (i >= 0 && i < EP.NA && j >= 0 && j < EP.NR) ? cellPoint(i, j) : null; drawOverlays(); });
for (const c of [oReal, oCnn]) {
  c.addEventListener('pointerleave', () => { S.hover = null; drawOverlays(); drawChannels(); });
  c.addEventListener('click', () => { if (S.hover) { S.pin = S.hover; drawOverlays(); drawChannels(); } });
}

// ---------------------------------------------------------------- model kararları
const VD = {tp: ['check', 'Yakaladı'], fn: ['x', 'Kaçırdı'], fp: ['alert', 'Yanlış alarm'], tn: ['minus', 'Doğru ret']};
const frameY = f => EP.frames[f].cat === 'TOR' ? 1 : 0;
const scoreOf = (m, f) => { const s = EP.scores[m.key]; return s && s[f] != null ? s[f] : null; };
function drawModels() {
  const f = S.frame, y = frameY(f); let yes = 0, n = 0;
  const box = $('models'); box.innerHTML = '';
  A.models.forEach(m => {
    const s = scoreOf(m, f), has = s != null, p = has && s >= m.thr;
    if (has) { n++; if (p) yes++; }
    const cls = !has ? null : y ? (p ? 'tp' : 'fn') : (p ? 'fp' : 'tn');
    const b = document.createElement('button'); b.className = 'mc'; b.setAttribute('aria-pressed', m.key === S.sel);
    b.title = `${m.long} · eşik ${m.thr.toFixed(3)}${cls ? ' · ' + VD[cls][1] : ''}`;
    b.innerHTML = `<span class="sw" style="background:${m.color}"></span><span class="nm">${esc(m.label)}</span><span class="sc${has ? '' : ' na'}">${has ? s.toFixed(2) : '—'}</span>` +
      (cls ? `<span class="vd ${cls}" aria-label="${VD[cls][1]}">${ico(VD[cls][0])}</span>` : '<span></span>') +
      `<span class="meter"><span class="track"></span><span class="fill" style="width:${has ? s * 100 : 0}%;background:${m.color}"></span><span class="thr" style="left:calc(${m.thr * 100}% - 1px)"></span></span>`;
    b.onclick = () => { if (S.sel === m.key) return; S.sel = m.key; drawModels(); drawTimeline(); send('model', {key: m.key}); };
    // Kartın yanındaki simge: bu modelin bu karedeki kararını Açıklanabilirlik sekmesinde açar
    const w = document.createElement('div'); w.className = 'mcw';
    const x = document.createElement('button'); x.className = 'mx'; x.dataset.tip = `${m.label} · bu kareyi açıkla`;
    x.setAttribute('aria-label', `${m.label}: bu kareyi açıkla`); x.title = x.dataset.tip; x.innerHTML = ico('bulb');
    x.onclick = () => { stopPlay(); send('explain', {id: EP.meta.episode_id, frame: S.frame, model: m.key}); };
    w.appendChild(b); w.appendChild(x);
    box.appendChild(w);
  });
  $('cons').innerHTML = `${yes}<small>/${n}</small>`;
  $('cons-t').innerHTML = `hortum diyor<br>gerçek: <b style="color:var(${y ? '--tor' : EP.frames[f].cat === 'WRN' ? '--wrn' : '--nul'})">y=${y}</b>`;
  $('pips').innerHTML = A.models.map(m => { const s = scoreOf(m, f); return `<i style="${s != null && s >= m.thr ? `background:${m.color}` : ''}"></i>`; }).join('');
  drawPlacard();
}
function drawPlacard() {
  const fr = EP.frames[S.frame], cat = fr.cat, root = document.documentElement;
  const map = {TOR: ['--tor', 'Hortum', 'tornado'], WRN: ['--wrn', 'Uyarı', 'alert'], NUL: ['--nul', 'Fırtına', 'circle-dashed']}[cat] || ['--nul', cat, 'circle-dashed'];
  root.style.setProperty('--state', css(map[0])); root.dataset.state = cat.toLowerCase();
  $('pl-big').textContent = map[1]; $('pl-ic').setAttribute('href', '#i-' + map[2]);
  const ef = $('pl-ef'); ef.hidden = !(cat === 'TOR' && fr.ef >= 0); ef.textContent = 'EF' + fr.ef;
  $('pl-y').textContent = `${cat} · y=${cat === 'TOR' ? 1 : 0}`;
  const m = EP.meta;
  const cell = (k, v) => `<div class="cell"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  const aA = ((fr.az0 % 360) + 360) % 360, aB = (aA + EP.NA * EP.DAZ) % 360;
  $('cells').innerHTML = cell(`Episode · ${esc(m.split)}`, esc(m.episode_id)) + cell('Tarih', esc(fr.date)) + cell('İstasyon', esc(m.site || '—')) +
    cell('Sektör', `${aA.toFixed(0)}–${aB.toFixed(0)}° · ${Math.max(fr.r0, 0).toFixed(0)}–${(fr.r0 + EP.NR * EP.DR).toFixed(0)} km`) +
    cell('Kareler', `<b style="color:var(--tor)">${m.n_tor}</b> <i>TOR</i> <b style="color:var(--wrn)">${m.n_wrn}</b> <i>WRN</i> <b style="color:var(--nul)">${m.n_nul}</b> <i>NUL</i>`);
}

// ---------------------------------------------------------------- zaman çizelgesi
const tl = $('tl');
function drawTimeline() {
  const NF = EP.F, W = tl.clientWidth || 1000, H = 112, L = 34, R = 8, T = 2, band = 8, top = T + band + 6, bot = H - 18, cw = (W - L - R) / NF;
  tl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const X = t => L + cw * (t + .5), Y = v => bot - (bot - top) * v;
  let h = '';
  for (const v of [0, .5, 1]) h += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line-soft)"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--faint)">${v}</text>`;
  h += `<rect x="${L + cw * S.frame}" y="${T}" width="${cw}" height="${bot - T}" fill="rgba(232,149,14,.10)"/>`;
  for (let t = 0; t < NF; t++) {
    const c = EP.frames[t].cat === 'TOR' ? 'var(--tor)' : EP.frames[t].cat === 'WRN' ? 'var(--wrn)' : 'var(--nul)';
    h += `<rect x="${L + cw * t + 1}" y="${T}" width="${Math.max(cw - 2, 1)}" height="${band}" rx="2" fill="${c}" opacity="${t === S.frame ? 1 : .5}"/>`;
  }
  const sm = A.models.find(m => m.key === S.sel);
  if (sm) h += `<line x1="${L}" x2="${W - R}" y1="${Y(sm.thr)}" y2="${Y(sm.thr)}" stroke="${sm.color}" stroke-dasharray="5 5" stroke-width="1.3"/>`;
  // Zaman çizelgesi kalabalık olmasın: yalnız ana karşılaştırmanın modelleri + seçili model (kartlarda hepsi var)
  const order = A.models.filter(m => m.key !== S.sel && (m.role || 'main') === 'main').concat(sm ? [sm] : []);
  for (const m of order) {
    const s = EP.scores[m.key]; if (!s) continue;
    const on = m.key === S.sel;
    const pts = s.map((v, t) => v == null ? null : `${X(t).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(' ');
    h += NF > 1 ? `<polyline points="${pts}" fill="none" stroke="${m.color}" stroke-width="${on ? 2.4 : 1.3}" stroke-linejoin="round" opacity="${on ? 1 : .5}"/>` : '';
    if (on && s[S.frame] != null) h += `<circle cx="${X(S.frame)}" cy="${Y(s[S.frame])}" r="4" fill="${m.color}" stroke="var(--panel)" stroke-width="1.5"/>`;
  }
  h += `<line x1="${X(S.frame)}" x2="${X(S.frame)}" y1="${T}" y2="${bot}" stroke="var(--accent)" stroke-width="2"/>`;
  // Saat etiketleri: aynı saat (aynı dakikada iki kare) bir kez yazılır, etiketler üst üste binmez; seçili kare önceliklidir.
  const MINGAP = 40, lab = [{t: S.frame, x: X(S.frame), s: EP.frames[S.frame].time}];
  const every = Math.max(1, Math.ceil(MINGAP / cw));
  for (let t = 0; t < NF; t++) {
    if (t % every || t === S.frame) continue;
    const x = X(t), s = EP.frames[t].time;
    if (lab.some(l => l.s === s || Math.abs(l.x - x) < MINGAP)) continue;
    lab.push({t, x, s});
  }
  for (const l of lab) h += `<text x="${l.x}" y="${H - 4}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="${l.t === S.frame ? 'var(--accent)' : 'var(--muted)'}">${l.s}</text>`;
  tl.innerHTML = h; tl._geo = {L, cw};
}
function pickFrame(e) { const b = tl.getBoundingClientRect(), g = tl._geo; setFrame(clamp(Math.floor((e.clientX - b.left - g.L) / g.cw), 0, EP.F - 1)); }
let dragging = false;
tl.addEventListener('pointerdown', e => { if (!EP) return; dragging = true; tl.setPointerCapture(e.pointerId); pickFrame(e); });
tl.addEventListener('pointermove', e => { if (dragging) pickFrame(e); });
tl.addEventListener('pointerup', () => dragging = false);

// ---------------------------------------------------------------- kare / oynatma
function setFrame(t, force) {
  t = ((t % EP.F) + EP.F) % EP.F; if (t === S.frame && !force) return;
  const keep = S.pin && t !== S.frame ? {i: S.pin.i, j: S.pin.j} : null;
  S.frame = t;
  if (keep) S.pin = cellPoint(keep.i, keep.j);    // işaret aynı hücrede kalır (CNN girdisinde aynı yer)
  if (S.hover) S.hover = cellPoint(S.hover.i, S.hover.j);
  $('clk').textContent = EP.frames[t].time;
  $('frm').textContent = `${t + 1}/${EP.F} · ${EP.frames[t].cat}`;
  drawStage(); drawChannels(); drawModels(); drawTimeline();
}
function stopPlay() { if (S.playing) togglePlay(); }
function togglePlay() {
  S.playing = !S.playing;
  $('play-ic').setAttribute('href', S.playing ? '#i-pause' : '#i-play');
  $('play').setAttribute('aria-label', S.playing ? 'Durdur' : 'Oynat');
  clearInterval(timer); if (S.playing) timer = setInterval(() => setFrame(S.frame + 1), 650);
}
$('play').onclick = togglePlay;
$('fprev').onclick = () => setFrame(S.frame - 1);
$('fnext').onclick = () => setFrame(S.frame + 1);
document.addEventListener('keydown', e => {
  if (!EP || S.tab !== 'gezgin' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') setFrame(S.frame + 1); else if (e.key === 'ArrowLeft') setFrame(S.frame - 1);
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
});
document.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.seg button').forEach(x => x.setAttribute('aria-pressed', x === b)); S.view = b.dataset.v;
  $('v-real').hidden = S.view === 'cnn'; $('v-cnn').hidden = S.view === 'real';
  $('stage-body').style.gridTemplateColumns = S.view === 'both' ? '' : 'minmax(0,1fr)';
  $('stage-body').classList.toggle('single', S.view !== 'both');
});

// ---------------------------------------------------------------- fırtına şeridi
const OUT = {p: '--good', n: '--tor', f: '--accent', t: '--nul', x: '--line'};
const OUTN = {p: 'yakaladı', n: 'kaçırdı', f: 'yanlış alarm', t: 'doğru ret', x: 'skor yok'};
const cStrip = $('c-strip'), track = $('track');
let STRIP = {codes: '', ids: [], pos: 0};
function drawStrip() {
  const W = Math.round(track.clientWidth * 2) || 800, H = 56; cStrip.width = W;
  const g = cStrip.getContext('2d'); g.clearRect(0, 0, W, H);
  const n = STRIP.ids.length;
  g.fillStyle = css('--panel-2'); g.beginPath(); g.roundRect ? g.roundRect(0, 8, W, H - 16, 10) : g.rect(0, 8, W, H - 16); g.fill();
  if (n) {
    const w = W / n;
    for (let i = 0; i < n; i++) {
      const o = STRIP.codes[i] || 'x', tall = o === 'n' || o === 'f';
      g.fillStyle = css(OUT[o]); g.globalAlpha = o === 't' ? .45 : 1;
      g.fillRect(i * w + (w > 3 ? w * .15 : 0), tall ? 10 : 18, Math.max(1, w > 3 ? w * .7 : w), tall ? H - 20 : H - 36);
    }
    g.globalAlpha = 1;
    const x = (STRIP.pos + .5) * w; g.fillStyle = css('--ink'); g.fillRect(x - 2, 0, 4, H);
    g.beginPath(); g.moveTo(x - 8, 0); g.lineTo(x + 8, 0); g.lineTo(x, 9); g.closePath(); g.fill();
  }
  $('spos').textContent = n ? `${fmtN(STRIP.pos + 1)} / ${fmtN(n)}` : '0 / 0';
}
const stripIdx = e => { const b = track.getBoundingClientRect(); return clamp(Math.floor((e.clientX - b.left) / b.width * STRIP.ids.length), 0, STRIP.ids.length - 1); };
track.addEventListener('pointermove', e => {
  if (!STRIP.ids.length) return;
  const i = stripIdx(e), t = $('strip-tip'), b = track.getBoundingClientRect();
  t.hidden = false; t.style.left = (e.clientX - b.left) + 'px'; t.textContent = `#${STRIP.ids[i]} · ${OUTN[STRIP.codes[i] || 'x']}`;
});
track.addEventListener('pointerleave', () => { $('strip-tip').hidden = true; });
const goto = i => { if (!STRIP.ids.length) return; i = clamp(i, 0, STRIP.ids.length - 1); if (i === STRIP.pos) return; stopPlay(); STRIP.pos = i; drawStrip(); send('goto', {pos: i, id: STRIP.ids[i]}); };
track.addEventListener('click', e => goto(stripIdx(e)));
$('sprev').onclick = () => goto(STRIP.pos - 1);
$('snext').onclick = () => goto(STRIP.pos + 1);
$('rnd').onclick = () => { stopPlay(); send('random'); };
$('explain').onclick = () => { if (!EP) return; stopPlay(); send('explain', {id: EP.meta.episode_id, frame: S.frame}); };
$('epid').addEventListener('keydown', e => { if (e.key === 'Enter' && /^-?\d+$/.test(e.target.value.trim())) { stopPlay(); send('epid', {id: e.target.value.trim()}); e.target.value = ''; e.target.blur(); } });

// ---------------------------------------------------------------- ana sayfa: sunum (deck)
// Her slayt: {id, name, frames, build(el, data), show(el, f, data), tick?(el, t)}. build yalnız veri değişince çağrılır.
// show(f) karede görünecekleri ayarlar; data-at="n" olan öğeler n. karede kendiliğinden açılır (CSS).
// → / sağ kenar: sonraki kare, son kareden sonra sonraki slayt · ← / sol kenar: geri · Boşluk: slaytı oynat.
// Ekranda gezinme düğmesi yok; sunum yalnız slayttan ibaret.
const DECK = {i: 0, f: 0, playing: false, timer: null, key: null, raf: 0};
const trNum = (v, d = 0) => Number(v).toLocaleString('tr-TR', {minimumFractionDigits: d, maximumFractionDigits: d});
function countUp(el, to, fmt, ms = 1200) {
  cancelAnimationFrame(el._raf);
  if (to == null) { el.textContent = '—'; return; }
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = t => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3); el.textContent = fmt(to * e); if (k < 1) el._raf = requestAnimationFrame(step); };
  el._raf = requestAnimationFrame(step);
}
function fitCanvas(cv) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return [w, h, dpr];
}

// --- slayt 1: kapak · dönen radar taraması + veri sayıları (sonuç yok: anlatım kronolojik, sonuçlar eğitimden sonra)
// Yankılar sabit (her açılışta aynı): üç fırtına hücresi, birinde kanca şekli ve hortum noktası.
const RADAR_ECHO = (() => {
  let z = 7; const rnd = () => (z = (z * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
  const cells = [{a: -0.9, r: .58, n: 150, s: .09, hook: true}, {a: 1.9, r: .42, n: 80, s: .07}, {a: 3.6, r: .72, n: 60, s: .06}];
  const pts = [];
  for (const c of cells) {
    const cx = Math.cos(c.a) * c.r, cy = Math.sin(c.a) * c.r;
    for (let k = 0; k < c.n; k++) {
      const dx = gauss() * c.s, dy = gauss() * c.s * .7, d = Math.hypot(dx / c.s, dy / (c.s * .7));
      pts.push({x: cx + dx, y: cy + dy, v: Math.max(0, 1 - d / 2.6) + rnd() * .15});
    }
    if (c.hook) for (let k = 0; k < 40; k++) {     // kanca (hook echo): hücrenin güneybatısında kıvrılan kuyruk
      const t = k / 40, ang = c.a + 1.2 + t * 2.2, rr = c.s * (1.6 - t * .9);
      pts.push({x: cx + Math.cos(ang) * rr + gauss() * .012, y: cy + Math.sin(ang) * rr + gauss() * .012, v: .55 + rnd() * .3});
    }
  }
  const c0 = cells[0], hk = c0.a + 1.2 + 2.2;
  return {pts, tor: {x: Math.cos(c0.a) * c0.r + Math.cos(hk) * c0.s * .7, y: Math.sin(c0.a) * c0.r + Math.sin(hk) * c0.s * .7}};
})();
const ECHO_COL = ['#04B9C7', '#11D44A', '#0BA33A', '#F2E22B', '#F28A12', '#EE2B22', '#C11A1A'].map(hex);

function drawRadar(cv, t) {
  const [W, H, dpr] = fitCanvas(cv); const g = cv.getContext('2d');
  g.clearRect(0, 0, W, H);
  const R = Math.min(W, H) * .46, cx = W / 2, cy = H / 2;
  const sweep = (t / 4200) * 2 * Math.PI % (2 * Math.PI);         // bir tur 4,2 sn
  // halkalar ve eksenler
  g.lineWidth = 1 * dpr; g.strokeStyle = 'rgba(40,55,80,.16)';
  for (const k of [.25, .5, .75, 1]) { g.beginPath(); g.arc(cx, cy, R * k, 0, 2 * Math.PI); g.stroke(); }
  g.setLineDash([3 * dpr, 6 * dpr]);
  for (let a = 0; a < 6; a++) { const q = a * Math.PI / 6; g.beginPath(); g.moveTo(cx - Math.cos(q) * R, cy - Math.sin(q) * R); g.lineTo(cx + Math.cos(q) * R, cy + Math.sin(q) * R); g.stroke(); }
  g.setLineDash([]);
  // yankılar: ışın geçtikten sonra sönerek kaybolur
  const age = (x, y) => { let a = Math.atan2(y, x); return ((sweep - a) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI); };
  const ps = Math.max(2, R * .022);
  for (const p of RADAR_ECHO.pts) {
    const fade = Math.exp(-age(p.x, p.y) / 2.4), c = ECHO_COL[Math.min(ECHO_COL.length - 1, Math.floor(p.v * ECHO_COL.length))];
    g.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${.12 + .78 * fade})`;
    g.fillRect(cx + p.x * R - ps / 2, cy + p.y * R - ps / 2, ps, ps);
  }
  // tarama izi (ince dilimler, geriye doğru sönen)
  const N = 48, span = .9;
  for (let k = 0; k < N; k++) {
    const a1 = sweep - span * k / N, a0 = sweep - span * (k + 1) / N;
    g.fillStyle = `rgba(232,149,14,${.2 * (1 - k / N)})`;
    g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, a0, a1); g.closePath(); g.fill();
  }
  g.strokeStyle = 'rgba(232,149,14,.9)'; g.lineWidth = 2 * dpr;
  g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(sweep) * R, cy + Math.sin(sweep) * R); g.stroke();
  // hortum noktası: ışın geçince kırmızı halka yayılır
  const T = RADAR_ECHO.tor, ta = age(T.x, T.y), tx = cx + T.x * R, ty = cy + T.y * R;
  const pulse = Math.min(1, ta / 1.6);
  g.strokeStyle = `rgba(227,52,78,${.9 * (1 - pulse)})`; g.lineWidth = 2.5 * dpr;
  g.beginPath(); g.arc(tx, ty, R * (.02 + .12 * pulse), 0, 2 * Math.PI); g.stroke();
  g.fillStyle = css('--tor'); g.beginPath(); g.arc(tx, ty, R * .018 + dpr, 0, 2 * Math.PI); g.fill();
  // radar
  g.fillStyle = css('--ink'); g.beginPath(); g.arc(cx, cy, 4 * dpr, 0, 2 * Math.PI); g.fill();
}

const SLIDE_KAPAK = {
  id: 'kapak', name: 'Kapak', frames: 1,          // statik slayt: her şey birlikte görünür
  build(el, d) {
    const k = d.kapak || {};
    el.innerHTML = `<div class="kapak">
      <div>
        <h1 data-at="0">Radar görüntülerinden<br><em>hortum</em> tespiti</h1>
        <div class="kstats">
          <div class="kstat" data-at="0"><div class="n" data-v="n">—</div><div class="k">örnek</div><div class="m">${k.years && k.years.length ? k.years.join(' · ') : '&nbsp;'}</div></div>
          <div class="kstat" data-at="0"><div class="n" data-v="pct">—</div><div class="k">hortum</div><div class="m">${k.n_tor != null ? trNum(k.n_tor) + ' örnek' : '&nbsp;'}</div></div>
        </div>
      </div>
      <div class="kradar" data-at="0"><canvas></canvas></div>
    </div>`;
  },
  show(el, f, d, prev) {
    const k = d.kapak || {};
    const vals = {n: [k.n, v => trNum(Math.round(v))], pct: [k.pct != null ? k.pct * 100 : null, v => '%' + trNum(v, 1)]};
    el.querySelectorAll('[data-v]').forEach(n => {
      const at = +n.closest('[data-at]').dataset.at, [v, fmt] = vals[n.dataset.v];
      if (f >= at && (prev == null || prev < at)) countUp(n, v, fmt);   // ileri gelirken sayar
      else if (f >= at) n.textContent = v == null ? '—' : fmt(v);
    });
  },
  tick(el, t) { const cv = el.querySelector('.kradar canvas'); if (cv) drawRadar(cv, t); },
};

// --- slayt 2: TorNet · bütün veri setinin haritası (katalog, hücre toplamları) + sayılar + etiketler
function drawDataMap(cv, v, sites, prog) {
  const [W, H, dpr] = fitCanvas(cv); const g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  if (!v) return;
  const C = v.cells, n = C.n.length;
  let la0 = 1e9, la1 = -1e9, lo0 = 1e9, lo1 = -1e9, nMax = 1, tMax = 1;
  for (let k = 0; k < n; k++) { la0 = Math.min(la0, C.lat[k]); la1 = Math.max(la1, C.lat[k]); lo0 = Math.min(lo0, C.lon[k]); lo1 = Math.max(lo1, C.lon[k]); nMax = Math.max(nMax, C.n[k]); tMax = Math.max(tMax, C.t[k]); }
  const kx = Math.cos(rad((la0 + la1) / 2)), pad = 14 * dpr;
  const s = Math.min((W - 2 * pad) / ((lo1 - lo0) * kx), (H - 2 * pad) / (la1 - la0));
  const ox = (W - (lo1 - lo0) * kx * s) / 2, oy = (H - (la1 - la0) * s) / 2;
  const P = (la, lo) => [ox + (lo - lo0) * kx * s, oy + (la1 - la) * s];
  const cut = lo0 + (lo1 - lo0 + 1) * prog, q = .3 * s * kx * 1.02, h = .3 * s * 1.02;   // batıdan doğuya açılır
  const lnN = Math.log(nMax + 1), lnT = Math.log(tMax + 1);
  for (let k = 0; k < n; k++) {
    if (C.lon[k] > cut) continue;
    const [x, y] = P(C.lat[k], C.lon[k]);
    g.fillStyle = `rgba(85,96,111,${.1 + .5 * Math.log(C.n[k] + 1) / lnN})`;
    g.fillRect(x - q / 2, y - h / 2, q, h);
  }
  for (let k = 0; k < n; k++) {
    if (!C.t[k] || C.lon[k] > cut) continue;
    const [x, y] = P(C.lat[k], C.lon[k]), a = .35 + .65 * Math.log(C.t[k] + 1) / lnT;
    g.fillStyle = `rgba(227,52,78,${a})`; g.beginPath(); g.arc(x, y, Math.max(1.2 * dpr, h * .42), 0, 2 * Math.PI); g.fill();
  }
  g.lineWidth = 1.6 * dpr; g.strokeStyle = css('--ink');
  for (const [la, lo] of sites) {
    if (lo > cut) continue;
    const [x, y] = P(la, lo); g.beginPath(); g.arc(x, y, 3.2 * dpr, 0, 2 * Math.PI); g.stroke();
  }
}
function animate(el, ms, draw) {    // giriş animasyonu: 0 → 1
  cancelAnimationFrame(el._anim);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { draw(1); return; }
  const t0 = performance.now();
  const step = t => { const k = Math.min(1, (t - t0) / ms); draw(1 - Math.pow(1 - k, 2)); if (k < 1) el._anim = requestAnimationFrame(step); };
  el._anim = requestAnimationFrame(step);
}
const CATS = [['tor', 'TOR', '--tor', 'y = 1'], ['wrn', 'WRN', '--wrn', 'y = 0'], ['nul', 'NUL', '--nul', 'y = 0']];
const catRows = v => CATS.map(([k, name, col, y]) => `<div class="crow"><i style="background:var(${col})"></i><b>${name}</b><span class="mono">${v ? trNum(v[k]) : '—'}</span><em>${y}</em></div>`).join('');

const SLIDE_TORNET = {
  id: 'tornet', name: 'TorNet', frames: 1,
  build(el, d) {
    const v = (d.veri || {}).all;
    el.innerHTML = `<div class="s-veri">
      <div class="vmap"><canvas></canvas>
        <div class="vleg"><span><i class="g"></i>örnek</span><span><i class="r"></i>hortum</span><span><i class="o"></i>radar</span></div></div>
      <div class="vside">
        <div class="skick">TorNet</div>
        <div class="vnums">
          <div><b data-v="n">—</b><span>örnek</span></div>
          <div><b>${v ? v.years.join('–') : '—'}</b><span>yıl</span></div>
          <div><b data-v="sites">—</b><span>radar</span></div>
        </div>
        <div class="cstack">${v ? CATS.map(([k, , col]) => `<i style="flex:${v[k]};background:var(${col})"></i>`).join('') : ''}</div>
        <div class="crows">${catRows(v)}</div>
      </div>
    </div>`;
  },
  show(el, f, d, prev) {
    const v = (d.veri || {}).all, sites = (d.veri || {}).sites || [];
    if (prev != null) return;
    el.querySelectorAll('[data-v]').forEach(n => countUp(n, v ? v[n.dataset.v === 'sites' ? 'n_sites' : 'n'] : null, x => trNum(Math.round(x)), 1100));
    const cv = el.querySelector('.vmap canvas');
    animate(el, 1400, p => drawDataMap(cv, v, sites, p));
  },
  resize(el, d) { drawDataMap(el.querySelector('.vmap canvas'), (d.veri || {}).all, (d.veri || {}).sites || [], 1); },
};

// --- slayt 3: kullandığımız veri · 100 örnekte sınıflar (waffle) + yıl seçimi + EF dağılımı
function hundred(u) {    // en büyük kalan yöntemiyle 100'e yuvarlama
  const tot = u.tor + u.wrn + u.nul, raw = CATS.map(([k]) => 100 * u[k] / tot), base = raw.map(Math.floor);
  let rest = 100 - base.reduce((a, b) => a + b, 0);
  raw.map((r, i) => [r - base[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (rest > 0) { base[i]++; rest--; } });
  return base;   // [tor, wrn, nul]
}
const SLIDE_DENGE = {
  id: 'denge', name: 'Kullanılan veri', frames: 1,
  build(el, d) {
    const V = d.veri || {}, u = V.used, all = V.all;
    const h = u ? hundred(u) : [0, 0, 0];
    // NUL → WRN → TOR sırası: hortumlar en son, sağ altta
    const order = [...Array(h[2]).fill('nul'), ...Array(h[1]).fill('wrn'), ...Array(h[0]).fill('tor')];
    const waf = order.map((c, k) => `<i class="${c}" style="--k:${k}"></i>`).join('');
    const yrs = all ? all.by_year : [], yMax = Math.max(1, ...yrs.map(r => r[1])), used = new Set(u ? u.years : []);
    const ybars = yrs.map(([y, n, t], k) => `<div class="yb${used.has(y) ? ' on' : ''}" style="--k:${k}">
        <span class="yn mono">${used.has(y) ? trNum(n) : ''}</span>
        <div class="ycol" style="height:${100 * n / yMax}%"><i style="height:${100 * t / n}%"></i></div>
        <span class="yl mono">${y}</span></div>`).join('');
    let efs = u ? u.ef.slice() : []; while (efs.length && !efs[efs.length - 1]) efs.pop();
    const eMax = Math.max(1, ...efs);
    const ef = efs.map((n, k) => `<div class="efr" style="--k:${k}"><b>EF${k}</b><div class="eft"><i style="width:${100 * n / eMax}%"></i></div><span class="mono">${trNum(n)}</span></div>`).join('');
    el.innerHTML = `<div class="s-denge">
      <div class="wbox">
        <div class="waffle">${waf}</div>
        <div class="wcap"><b>${h[0]}</b><span>/ 100</span></div>
        <div class="crows">${catRows(u)}</div>
      </div>
      <div class="dright">
        <div class="dblk"><div class="dk">yıl</div><div class="ybars">${ybars}</div></div>
        <div class="dblk"><div class="dk">EF · hortum örnekleri</div><div class="efbars">${ef}</div></div>
      </div>
    </div>`;
  },
  show() {},
};

// ---------------------------------------------------------------- bölüm 3: radar görüntüsü (otomatik seçilen fırtına)
// Veri Python'dan (ana.py _storm): gezgin ile aynı biçim, uint8 (F, C, 120, 240); renkler Gezgin'deki CH haritaları.
let STORM = null;
function decodeStorm(e) {
  const [F, C, NA, NR] = e.shape, x = b64(e.x), N = NA * NR, ch = {};
  e.variables.forEach((v, ci) => {
    if (!CH[v]) return;
    const [lo, hi] = e.ranges[v], css = [], rgb = new Uint8ClampedArray(256 * 3);
    for (let q = 0; q < 256; q++) {
      const c = q === 0 ? NODATA : CH[v].cmap((q - 1) / 254 * (hi - lo) + lo);
      rgb[q * 3] = c[0]; rgb[q * 3 + 1] = c[1]; rgb[q * 3 + 2] = c[2]; css.push(`rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`);
    }
    const frames = []; for (let f = 0; f < F; f++) frames.push(x.subarray((f * C + ci) * N, (f * C + ci + 1) * N));
    ch[v] = {lo, hi, css, rgb, frames};
  });
  return {key: e.key, F, NA, NR, DAZ: e.daz, DR: e.dr, frames: e.frames, ch, rot: e.rot, best: e.best, site: e.site, ep: e.episode_id};
}
const stormOf = d => { const e = d && d.storm; if (!e) return (STORM = null); if (!STORM || STORM.key !== e.key) STORM = decodeStorm(e); return STORM; };

// Hücre köşeleri (km, ekran yönü: x doğu, y güney). m: 0 gerçek konum (kutupsal) → 1 dilim döner (orta açı sağa)
// → 2 dikdörtgen (CNN girdisi: satır = açı, sütun = mesafe, her hücre 0,25 km kare). Radar (0, 0).
function stormGeom(St, f, m) {
  const {NA, NR, DAZ, DR} = St, fr = St.frames[f], azc = fr.az0 + NA * DAZ / 2;
  const u = clamp(m, 0, 1), t = clamp(m - 1, 0, 1), NR1 = NR + 1;
  const X = new Float32Array((NA + 1) * NR1), Y = new Float32Array((NA + 1) * NR1);
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (let i = 0; i <= NA; i++) {
    const az = fr.az0 + i * DAZ, th = rad(az - 90 - u * (azc - 90)), c = Math.cos(th), s = Math.sin(th), yf = (i - NA / 2) * DR;
    for (let j = 0; j <= NR; j++) {
      const r = Math.max(0, fr.r0 + j * DR), px = r * c, py = r * s, k = i * NR1 + j;
      const X_ = px + (r - px) * t, Y_ = py + (yf - py) * t; X[k] = X_; Y[k] = Y_;
      if (X_ < x0) x0 = X_; if (X_ > x1) x1 = X_; if (Y_ < y0) y0 = Y_; if (Y_ > y1) y1 = Y_;
    }
  }
  return {X, Y, box: [x0, x1, y0, y1]};
}
const camXf = (cam, W, H) => { const s = Math.min(W / (2 * cam.hw), H / (2 * cam.hh)); return {s, sx: x => W / 2 + (x - cam.cx) * s, sy: y => H / 2 + (y - cam.cy) * s}; };
// Hücreleri dörtgen olarak çizer (renge göre gruplanmış yollar; dikiş görünmesin diye aynı renkle ince kontur)
function drawCells(g, St, f, chName, G, T, W, H, dpr) {
  const ch = St.ch[chName], q = ch.frames[f], NR = St.NR, NR1 = NR + 1, paths = new Map();
  const X = G.X, Y = G.Y, mg = 4 * dpr;
  for (let i = 0; i < St.NA; i++) for (let j = 0; j < NR; j++) {
    const a = i * NR1 + j, b = a + 1, c = a + NR1 + 1, d = a + NR1;
    const ax = T.sx(X[a]), ay = T.sy(Y[a]);
    if ((ax < -mg && T.sx(X[c]) < -mg) || (ax > W + mg && T.sx(X[c]) > W + mg) || (ay < -mg && T.sy(Y[c]) < -mg) || (ay > H + mg && T.sy(Y[c]) > H + mg)) continue;
    const v = q[i * NR + j]; let p = paths.get(v); if (!p) paths.set(v, p = new Path2D());
    p.moveTo(ax, ay); p.lineTo(T.sx(X[b]), T.sy(Y[b])); p.lineTo(T.sx(X[c]), T.sy(Y[c])); p.lineTo(T.sx(X[d]), T.sy(Y[d])); p.closePath();
  }
  g.lineWidth = .7 * dpr; g.lineJoin = 'round';
  for (const [v, p] of paths) { g.fillStyle = ch.css[v]; g.strokeStyle = ch.css[v]; g.fill(p); g.stroke(p); }
}
const ease = k => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
const lerp = (a, b, k) => a + (b - a) * k;
const kmLabel = v => trNum(v, v < 10 ? 1 : 0);

// --- 3a · radar → dilim → dikdörtgen (4 kare, geçişler animasyonlu)
const A3 = {from: null, to: null, t0: 0, dur: 1, layer: null, lkey: ''};
const A3T = f => ({z: f > 0 ? 1 : 0, m: f >= 2 ? 2 : 0, rings: f === 0 ? 1 : f === 1 ? .4 : 0, outline: f === 1 ? 1 : 0, axes: f >= 2 ? 1 : 0, marks: f === 3 ? 1 : 0});
function a3Params(now) {
  if (!A3.to) return A3T(0);
  const k = A3.dur ? ease(clamp((now - A3.t0) / A3.dur, 0, 1)) : 1, o = {};
  for (const key in A3.to) o[key] = lerp(A3.from[key], A3.to[key], k);
  return o;
}
function drawA3(cv, St, t) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  if (!St) return;
  const f = St.best, fr = St.frames[f], P = a3Params(t), G = stormGeom(St, f, P.m), Rd = fr.r0 + St.NR * St.DR + 6;
  const [x0, x1, y0, y1] = G.box, pad = P.marks > 0 ? 1.3 : 1.2;
  const sc = {cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 + P.axes * (y1 - y0) * .04, hw: (x1 - x0) / 2 * pad, hh: (y1 - y0) / 2 * pad};
  const cam = {cx: lerp(0, sc.cx, P.z), cy: lerp(0, sc.cy, P.z), hw: Math.exp(lerp(Math.log(Rd), Math.log(sc.hw), P.z)), hh: Math.exp(lerp(Math.log(Rd), Math.log(sc.hh), P.z))};
  const T = camXf(cam, W, H);
  // halkalar ve tarama ışını (radar etrafında)
  const ra = P.rings * (1 - clamp(P.m, 0, 1));
  if (ra > .01) {
    g.save(); g.globalAlpha = ra; g.strokeStyle = 'rgba(40,55,80,.2)'; g.lineWidth = dpr;
    g.fillStyle = css('--faint'); g.font = `500 ${11 * dpr}px ${MONO}`; g.textAlign = 'left';
    for (let r = 25; r <= Rd; r += 25) { g.beginPath(); g.arc(T.sx(0), T.sy(0), r * T.s, 0, 2 * Math.PI); g.stroke(); g.fillText(r + ' km', T.sx(0) + 4 * dpr, T.sy(-r) - 4 * dpr); }
    g.restore();
  }
  // hücre katmanı: görünüm değişmedikçe önbellekten
  const lkey = [W, H, P.m.toFixed(4), cam.cx.toFixed(3), cam.cy.toFixed(3), cam.hw.toFixed(3), cam.hh.toFixed(3), St.key].join('|');
  if (A3.lkey !== lkey) {
    const L = A3.layer || (A3.layer = document.createElement('canvas')); L.width = W; L.height = H;
    const lg = L.getContext('2d'); lg.clearRect(0, 0, W, H); drawCells(lg, St, f, 'DBZ', G, T, W, H, dpr); A3.lkey = lkey;
  }
  g.drawImage(A3.layer, 0, 0);
  if (ra > .01) {
    const sw = (t / 4200) * 2 * Math.PI % (2 * Math.PI), cx = T.sx(0), cy = T.sy(0), R = Rd * T.s;
    g.save(); g.globalAlpha = ra;
    for (let k = 0; k < 40; k++) { g.fillStyle = `rgba(232,149,14,${.18 * (1 - k / 40)})`; g.beginPath(); g.moveTo(cx, cy); g.arc(cx, cy, R, sw - .8 * (k + 1) / 40, sw - .8 * k / 40); g.closePath(); g.fill(); }
    g.strokeStyle = 'rgba(232,149,14,.9)'; g.lineWidth = 2 * dpr; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(sw) * R, cy + Math.sin(sw) * R); g.stroke();
    g.fillStyle = css('--ink'); g.beginPath(); g.arc(cx, cy, 4.5 * dpr, 0, 2 * Math.PI); g.fill();
    g.font = `600 ${13 * dpr}px "IBM Plex Sans", sans-serif`; g.textAlign = 'left'; g.fillText('radar · ' + St.site, cx + 10 * dpr, cy + 16 * dpr);
    g.restore();
  }
  // dilimin sınırı ve boyutu
  if (P.outline > .01) {
    const {NA, NR} = St, NR1 = NR + 1, pt = k => [T.sx(G.X[k]), T.sy(G.Y[k])];
    g.save(); g.globalAlpha = P.outline; g.strokeStyle = css('--accent'); g.lineWidth = 2.5 * dpr; g.beginPath();
    for (let j = 0; j <= NR; j++) g.lineTo(...pt(j));
    for (let i = 0; i <= NA; i++) g.lineTo(...pt(i * NR1 + NR));
    for (let j = NR; j >= 0; j--) g.lineTo(...pt(NA * NR1 + j));
    for (let i = NA; i >= 0; i--) g.lineTo(...pt(i * NR1));
    g.closePath(); g.stroke();
    const mid = Math.round(NA / 2) * NR1 + NR, [mx, my] = pt(mid), [ix, iy] = pt(Math.round(NA / 2) * NR1);
    const dx = mx - ix, dy = my - iy, n = Math.hypot(dx, dy) || 1;
    g.fillStyle = css('--ink'); g.font = `800 ${26 * dpr}px ${getComputedStyle(document.body).getPropertyValue('--f-display') || 'sans-serif'}`;
    g.textAlign = dx >= 0 ? 'left' : 'right'; g.textBaseline = 'middle';
    g.fillText(`${trNum(NA * St.DAZ)}° × ${trNum(NR * St.DR)} km`, mx + dx / n * 34 * dpr, my + dy / n * 34 * dpr);
    g.restore();
  }
  // dikdörtgen eksenleri
  if (P.axes > .01) {
    const L = T.sx(fr.r0), R = T.sx(fr.r0 + St.NR * St.DR), Tp = T.sy(-St.NA / 2 * St.DR), B = T.sy(St.NA / 2 * St.DR);
    g.save(); g.globalAlpha = P.axes; g.fillStyle = css('--muted'); g.strokeStyle = 'rgba(40,55,80,.35)'; g.lineWidth = dpr;
    g.strokeRect(L, Tp, R - L, B - Tp);
    g.font = `500 ${13 * dpr}px ${MONO}`; g.textBaseline = 'top'; g.textAlign = 'center';
    g.fillText(`mesafe · ${St.NR} × ${trNum(St.DR * 1000)} m`, (L + R) / 2, B + 10 * dpr);
    g.textAlign = 'left'; g.fillText(`${kmLabel(Math.max(fr.r0, 0))} km`, L, B + 10 * dpr);
    g.textAlign = 'right'; g.fillText(`${kmLabel(fr.r0 + St.NR * St.DR)} km`, R, B + 10 * dpr);
    g.translate(L - 12 * dpr, (Tp + B) / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.textBaseline = 'bottom';
    g.fillText(`açı · ${St.NA} × ${trNum(St.DAZ, 1)}°`, 0, 0);
    g.restore();
  }
  // aynı boy piksel, farklı gerçek genişlik: radara en yakın ve en uzak sütun
  if (P.marks > .01) {
    g.save(); g.globalAlpha = P.marks; g.lineWidth = 3 * dpr; g.strokeStyle = css('--ink'); g.fillStyle = css('--ink');
    g.font = `600 ${17 * dpr}px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'bottom';
    const iA = Math.floor(St.NA / 2) - 3, iB = iA + 6;
    for (const [j0, j1] of [[8, 16], [St.NR - 16, St.NR - 8]]) {
      const x = T.sx(fr.r0 + j0 * St.DR), w = T.sx(fr.r0 + j1 * St.DR) - x, y = T.sy((iA - St.NA / 2) * St.DR), h = T.sy((iB - St.NA / 2) * St.DR) - y;
      g.strokeRect(x, y, w, h);
      const r = Math.max(fr.r0 + (j0 + j1) / 2 * St.DR, 0), wm = r * rad(St.DAZ) * 1000;
      const lab = `${trNum(Math.round(wm / 10) * 10)} m`, tw = g.measureText(lab).width;
      g.fillStyle = 'rgba(255,255,255,.92)'; g.fillRect(x + w / 2 - tw / 2 - 6 * dpr, y - 30 * dpr, tw + 12 * dpr, 22 * dpr);
      g.fillStyle = css('--ink'); g.fillText(lab, x + w / 2, y - 11 * dpr);
    }
    g.restore();
  }
}
const SLIDE_POLAR = {
  id: 'polar', name: 'Radar görüntüsü', frames: 4, delay: 2600,
  build(el, d) { stormOf(d); el.innerHTML = `<div class="s-polar"><canvas></canvas>${STORM ? '' : '<div class="snote">Radar verisi yok</div>'}</div>`; A3.lkey = ''; },
  show(el, f, d, prev) {
    const now = performance.now(), to = A3T(f);
    if (prev == null) { A3.from = to; A3.to = to; A3.dur = 0; }
    else { A3.from = a3Params(now); A3.to = to; A3.t0 = now; A3.dur = Math.abs(to.m - A3.from.m) > .5 ? 2400 : 1200; }
  },
  tick(el, t) { drawA3(el.querySelector('canvas'), STORM, t); },
};

// --- 3b · üç ölçüm: modelin gördüğü dikdörtgen, en güçlü dönme halkayla işaretli
const CH_DESC = {DBZ: ['yansıtıcılık', 'dBZ'], VEL: ['radyal hız', 'm/s'], WIDTH: ['spektrum genişliği', 'm/s']};
function drawRect(cv, St, f, chName, ring) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  const ch = St.ch[chName], q = ch.frames[f], raw = document.createElement('canvas'); raw.width = St.NR; raw.height = St.NA;
  const rx = raw.getContext('2d'), img = rx.createImageData(St.NR, St.NA), D_ = img.data;
  for (let k = 0; k < q.length; k++) { const o = q[k] * 3; D_[k * 4] = ch.rgb[o]; D_[k * 4 + 1] = ch.rgb[o + 1]; D_[k * 4 + 2] = ch.rgb[o + 2]; D_[k * 4 + 3] = 255; }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(raw, 0, 0, W, H);
  const rc = St.rot[f];
  if (ring && rc) {
    const x = (rc[1] + .5) / St.NR * W, y = (rc[0] + .5) / St.NA * H;
    g.lineWidth = 5 * dpr; g.strokeStyle = 'rgba(255,255,255,.9)'; g.beginPath(); g.arc(x, y, 16 * dpr, 0, 2 * Math.PI); g.stroke();
    g.lineWidth = 2.5 * dpr; g.strokeStyle = css('--ink'); g.beginPath(); g.arc(x, y, 16 * dpr, 0, 2 * Math.PI); g.stroke();
  }
}
function cbar(chName, St) {
  const ch = St.ch[chName], stops = [];
  for (let k = 0; k <= 24; k++) { const c = CH[chName].cmap(ch.lo + (ch.hi - ch.lo) * k / 24); stops.push(`rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0}) ${(k / 24 * 100).toFixed(1)}%`); }
  const ends = chName === 'VEL' ? ['yaklaşan', 'uzaklaşan'] : [trNum(ch.lo), trNum(ch.hi)];
  return `<div class="cbar3"><i style="background:linear-gradient(90deg,${stops.join(',')})"></i><span>${ends[0]}</span><span>${ends[1]}</span></div>`;
}
const SLIDE_KANAL = {
  id: 'kanal', name: 'Üç ölçüm', frames: 1,
  build(el, d) {
    const St = stormOf(d);
    if (!St) { el.innerHTML = '<div class="snote">Radar verisi yok</div>'; return; }
    const chans = ['DBZ', 'VEL', 'WIDTH'].filter(c => St.ch[c]);
    el.innerHTML = `<div class="s-kanal">${chans.map((c, k) => `<div class="kpan" style="--k:${k}">
        <div class="khead"><b>${c}</b><span>${CH_DESC[c][0]} · ${CH_DESC[c][1]}</span></div>
        <div class="kcv"><canvas data-ch="${c}"></canvas></div>${cbar(c, St)}</div>`).join('')}</div>`;
  },
  show(el) { this.resize(el); },
  resize(el) { if (STORM) el.querySelectorAll('canvas[data-ch]').forEach(cv => drawRect(cv, STORM, STORM.best, cv.dataset.ch, true)); },
};

// --- 3c · hortumun imzası: VEL'de dönüş çifti, fırtınanın kareleri boyunca (→ ya da Boşluk ile oynar)
const C3 = {cache: new Map(), ckey: ''};
function cellXY(St, f, i, j) { const fr = St.frames[f], az = fr.az0 + (i + .5) * St.DAZ, r = fr.r0 + (j + .5) * St.DR, th = rad(az - 90); return [r * Math.cos(th), r * Math.sin(th)]; }
function drawC3(cv, St, f, t) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  if (!St) return;
  // Kamera dönüş çiftini izler: fırtına 5 dakikada birkaç km yol alır, sabit kamerada kadrajdan çıkardı
  const b = St.rot[f] || St.rot[St.best] || [St.NA / 2, St.NR / 2], [cx, cy] = cellXY(St, St.rot[f] ? f : St.best, b[0], b[1]), hw = 11;
  const cam = {cx, cy, hw: hw * Math.max(1, W / H), hh: hw * Math.max(1, H / W)}, T = camXf(cam, W, H);
  const key = `${W}x${H}|${St.key}`;
  if (C3.ckey !== key) { C3.cache.clear(); C3.ckey = key; }
  let L = C3.cache.get(f);
  if (!L) {
    L = document.createElement('canvas'); L.width = W; L.height = H;
    drawCells(L.getContext('2d'), St, f, 'VEL', stormGeom(St, f, 0), T, W, H, dpr); C3.cache.set(f, L);
  }
  g.drawImage(L, 0, 0);
  // ölçek: 5 km
  const s5 = 5 * T.s; g.fillStyle = css('--ink'); g.fillRect(24 * dpr, H - 30 * dpr, s5, 3 * dpr);
  g.font = `500 ${12 * dpr}px ${MONO}`; g.textAlign = 'left'; g.fillText('5 km', 24 * dpr, H - 38 * dpr);
  // dönüş çifti: halka ve saat yönünün tersine dönen oklar
  const rc = St.rot[f]; if (!rc) return;
  const [px, py] = cellXY(St, f, rc[0], rc[1]), x = T.sx(px), y = T.sy(py), R = 2.2 * T.s;
  g.lineWidth = 6 * dpr; g.strokeStyle = 'rgba(255,255,255,.85)'; g.beginPath(); g.arc(x, y, R, 0, 2 * Math.PI); g.stroke();
  g.lineWidth = 2.5 * dpr; g.strokeStyle = css('--ink'); g.beginPath(); g.arc(x, y, R, 0, 2 * Math.PI); g.stroke();
  const ph = -(t / 1400) % (2 * Math.PI), Ra = R * 1.45;
  g.strokeStyle = css('--ink'); g.fillStyle = css('--ink'); g.lineWidth = 3 * dpr;
  for (let k = 0; k < 3; k++) {
    const a0 = ph + k * 2 * Math.PI / 3, a1 = a0 - 1.2;
    g.beginPath(); g.arc(x, y, Ra, a1, a0); g.stroke();      // a0 → a1: saat yönünün tersi (ok ucu a1'de)
    const hx = x + Math.cos(a1) * Ra, hy = y + Math.sin(a1) * Ra, tx = Math.sin(a1), ty = -Math.cos(a1), L2 = 11 * dpr;
    g.beginPath(); g.moveTo(hx - tx * L2 * .2, hy - ty * L2 * .2);
    g.lineTo(hx + tx * L2 + Math.cos(a1) * L2 * .55, hy + ty * L2 + Math.sin(a1) * L2 * .55);
    g.lineTo(hx + tx * L2 - Math.cos(a1) * L2 * .55, hy + ty * L2 - Math.sin(a1) * L2 * .55); g.closePath(); g.fill();
  }
}
const SLIDE_IMZA = {
  id: 'imza', name: 'Hortumun imzası', delay: 900,
  get frames() { return STORM ? STORM.F : 1; },
  build(el, d) {
    const St = stormOf(d); C3.ckey = '';
    if (!St) { el.innerHTML = '<div class="snote">Radar verisi yok</div>'; return; }
    const fr = St.frames[St.best];
    el.innerHTML = `<div class="s-imza">
      <div class="iside">
        <div class="ichip"><b>TOR</b>${fr.ef >= 0 ? `<span>EF${fr.ef}</span>` : ''}</div>
        <div class="itime mono" data-v="time"></div>
        <div class="itl">${St.frames.map((x, k) => `<i data-k="${k}" class="${x.cat.toLowerCase()}"></i>`).join('')}</div>
        <div class="imeta mono">${esc(St.site)} · ${esc(fr.date)}</div>
        ${cbar('VEL', St)}
      </div>
      <div class="icv"><canvas></canvas></div>
    </div>`;
  },
  show(el, f) {
    if (!STORM) return;
    const fr = STORM.frames[f]; el.querySelector('[data-v="time"]').textContent = fr.time;
    el.querySelectorAll('.itl i').forEach(i => i.classList.toggle('on', +i.dataset.k === f));
  },
  tick(el, t) { drawC3(el.querySelector('.icv canvas'), STORM, DECK.f, t); },
};

// ---------------------------------------------------------------- bölüm 4: modele ne veriyoruz? (01 defteri D, kayıtlı çıktılardan)
const auc = v => v == null ? '—' : trNum(v, 2);
// Yatay AUC çubukları: eksen 0,5 (yazı tura) → üst sınır; değerler çubuk sonunda
function aucAxis(lo, hi) { const t = []; for (let v = Math.ceil(lo * 10) / 10; v <= hi + 1e-9; v += .1) t.push(+v.toFixed(1)); return t; }
const SLIDE_OLCUM = {
  id: 'olcum', name: 'Hangi ölçüm?', frames: 1,
  build(el, d) {
    const G = d.girdi || {}, E = G.eda || {}, rows = (E.d3 || []).slice().sort((a, b) => b.tor - a.tor), sel = new Set(G.vars || []);
    if (!rows.length) { el.innerHTML = '<div class="snote">01 defterinde D3 çıktısı yok</div>'; return; }
    const hi = Math.max(.8, Math.ceil(Math.max(...rows.map(r => Math.max(r.tor, r.fa))) * 10) / 10), lo = .5, X = v => 100 * (v - lo) / (hi - lo);
    const ticks = aucAxis(lo, hi);
    const bars = rows.map((r, k) => `<div class="arow${sel.has(r.v) ? ' on' : ''}" style="--k:${k}">
        <b>${esc(r.v)}</b>
        <div class="atrack">${ticks.map(t => `<s style="left:${X(t)}%"></s>`).join('')}
          <div class="abar t" style="--w:${X(r.tor)}%"><span class="mono">${auc(r.tor)}</span></div>
          <div class="abar f" style="--w:${X(r.fa)}%"><span class="mono">${auc(r.fa)}</span></div>
        </div></div>`).join('');
    const B = E.d3b || [], s3 = B.find(r => /seçilen/i.test(r.name)), s6 = B.find(r => r.k === (G.all_vars || []).length);
    el.innerHTML = `<div class="s-olcum">
      <div class="achart">
        <div class="aleg"><span><i class="t"></i>hortum / hortumsuz</span><span><i class="f"></i>hortum / yanlış alarm</span></div>
        ${bars}
        <div class="arow axis"><b></b><div class="atrack">${ticks.map(t => `<em style="left:${X(t)}%">${trNum(t, 1)}</em>`).join('')}</div></div>
      </div>
      <div class="aside">
        <div class="skick">${(G.all_vars || []).length} → ${(G.vars || []).length}</div>
        <div class="acomb">
          <div><b>${auc(s3 && s3.lr)}</b><span>seçilen ${(G.vars || []).length} ölçüm</span></div>
          <div class="dim"><b>${auc(s6 && s6.lr)}</b><span>${(G.all_vars || []).length} ölçümün hepsi</span></div>
        </div>
        <div class="ameta">AUC · birlikte, lojistik regresyon</div>
      </div>
    </div>`;
  },
  show() {},
};

// --- 4b · açı ve zaman + veri küpünün daralması
const SLIDE_KUP = {
  id: 'kup', name: 'Hangi açı, hangi an?', frames: 1,
  build(el, d) {
    const G = d.girdi || {}, E = G.eda || {}, vars = G.all_vars || [], sel = new Set(G.vars || []);
    const nS = G.n_sweeps || 2, nF = G.n_frames || 4, sw = G.sweep || 0, fr = ((G.frame ?? -1) + nF) % nF;
    const sname = ((E.d4 && E.d4.cols) || ['0,5°', '0,9°']).map(c => c.replace(/\s*\(.*\)\s*/, ''));
    // açı: seçilen kanallarda iki taramanın AUC'si
    const d4 = ((E.d4 && E.d4.rows) || []).filter(r => sel.has(r.v)), aMax = Math.max(.8, ...d4.flatMap(r => [r.a, r.b]));
    const Y = v => 100 * (v - .5) / (aMax - .5);
    const aci = d4.map((r, k) => `<div class="gcol" style="--k:${k}"><div class="gbars">
        <i class="${sw === 0 ? 'on' : ''}" style="--h:${Y(r.a)}%"><span class="mono">${auc(r.a)}</span></i>
        <i class="${sw === 1 ? 'on' : ''}" style="--h:${Y(r.b)}%"><span class="mono">${auc(r.b)}</span></i></div><b>${esc(r.v)}</b></div>`).join('');
    // zaman: hortumun başladığı kare
    const d5 = E.d5 || [], nMax = Math.max(1, ...d5.map(r => r.n));
    const zaman = d5.map((r, k) => `<div class="gcol" style="--k:${k}"><div class="gbars one"><i class="${k === fr ? 'on' : ''}" style="--h:${100 * r.n / nMax}%"><span class="mono">${r.n}</span></i></div><b>${k + 1}</b></div>`).join('');
    // veri küpü: satır = ölçüm, sütun = tarama × kare; seçilmeyenler sırayla söner
    // taramalar arasında boş bir sütun (u); üstte tarama açısı, altında kare numaraları
    const sp = s => s < nS - 1 ? '<u></u>' : '';
    const head = Array.from({length: nS}, (_, s) => `<span style="grid-column:span ${nF}">${esc(sname[s] || '')}</span>${sp(s)}`).join('');
    const nums = Array.from({length: nS}, (_, s) => Array.from({length: nF}, (_, f) => `<em>${f + 1}</em>`).join('') + sp(s)).join('');
    const cube = vars.map(v => `<b class="${sel.has(v) ? '' : 'x0'}">${esc(v)}</b>` +
      Array.from({length: nS}, (_, s) => Array.from({length: nF}, (_, f) =>
        `<i class="${!sel.has(v) ? 'x0' : s !== sw ? 'x1' : f !== fr ? 'x2' : 'keep'}"></i>`).join('') + sp(s)).join('')).join('');
    const cols = 'auto ' + Array.from({length: nS}, (_, s) => `repeat(${nF},1fr)` + (s < nS - 1 ? ' 18px' : '')).join(' ');
    const NA = G.na || 120, NR = G.nr || 240, big = vars.length * nS * nF * NA * NR, small = (G.vars || []).length * NA * NR;
    el.innerHTML = `<div class="s-kup">
      <div class="kleft">
        <div class="gblk"><div class="dk">tarama açısı · AUC</div><div class="gleg"><span><i class="on"></i>${esc(sname[sw] || '')}</span><span><i></i>${esc(sname[1 - sw] || '')}</span></div><div class="gchart">${aci}</div></div>
        <div class="gblk"><div class="dk">hortumun başladığı kare</div><div class="gchart">${zaman}</div></div>
      </div>
      <div class="kright">
        <div class="cube" style="grid-template-columns:${cols}"><span></span>${head}<span></span>${nums}${cube}</div>
        <div class="csize">
          <div class="c0"><span class="mono">${vars.length} × ${nS} × ${nF} × ${NA} × ${NR}</span><b>${trNum(big)}</b></div>
          <div class="c1"><span class="mono">${(G.vars || []).length} × ${NA} × ${NR}</span><b>${trNum(small)}</b></div>
          <div class="cx">${trNum(Math.round(big / small))}× küçük</div>
        </div>
      </div>
    </div>`;
  },
  show() {},
};

// ---------------------------------------------------------------- bölüm 5: ön işleme ve bölme
// 5a · Bölüm 3'teki fırtınanın en iyi karesi: ham (boş pikseller damalı) → ağın girdisi ([-1, 1], boşluklar doldurulmuş)
function drawRawQ(cv, St, f, chName) {
  const [W, H] = fitCanvas(cv), g = cv.getContext('2d'), ch = St.ch[chName], q = ch.frames[f];
  const raw = document.createElement('canvas'); raw.width = St.NR; raw.height = St.NA;
  const rx = raw.getContext('2d'), img = rx.createImageData(St.NR, St.NA), D_ = img.data;
  for (let i = 0; i < St.NA; i++) for (let j = 0; j < St.NR; j++) {
    const k = i * St.NR + j, v = q[k];
    if (v === 0) { const c = ((i >> 2) + (j >> 2)) & 1 ? 255 : 214; D_[k * 4] = c; D_[k * 4 + 1] = c + 4; D_[k * 4 + 2] = c + 10; }
    else { D_[k * 4] = ch.rgb[v * 3]; D_[k * 4 + 1] = ch.rgb[v * 3 + 1]; D_[k * 4 + 2] = ch.rgb[v * 3 + 2]; }
    D_[k * 4 + 3] = 255;
  }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(raw, 0, 0, W, H);
}
const GRAY = x => { const t = (clamp(x, -1, 1) + 1) / 2; return [24 + t * 226, 30 + t * 222, 40 + t * 214]; };
function drawInputQ(cv, St, f, chName, fill) {
  const [W, H] = fitCanvas(cv), g = cv.getContext('2d'), q = St.ch[chName].frames[f];
  const raw = document.createElement('canvas'); raw.width = St.NR; raw.height = St.NA;
  const rx = raw.getContext('2d'), img = rx.createImageData(St.NR, St.NA), D_ = img.data;
  for (let k = 0; k < q.length; k++) { const c = GRAY(q[k] === 0 ? fill : (q[k] - 1) / 254 * 2 - 1); D_[k * 4] = c[0]; D_[k * 4 + 1] = c[1]; D_[k * 4 + 2] = c[2]; D_[k * 4 + 3] = 255; }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(raw, 0, 0, W, H);
}
const UNIT = {DBZ: 'dBZ', VEL: 'm/s', WIDTH: 'm/s'};
const gb = b => b == null ? '—' : trNum(b / 1e9, 1) + ' GB';
const SLIDE_ONISLEME = {
  id: 'onisleme', name: 'Ön işleme', frames: 1,
  build(el, d) {
    const St = stormOf(d), O = d.onisleme || {};
    if (!St) { el.innerHTML = '<div class="snote">Radar verisi yok</div>'; return; }
    const chans = Object.keys(O.ranges || {}).filter(c => St.ch[c]);
    const gstops = Array.from({length: 11}, (_, k) => { const c = GRAY(-1 + k / 5); return `rgb(${c.map(x => x | 0)}) ${k * 10}%`; }).join(',');
    const rows = chans.map((c, k) => {
      const q = St.ch[c].frames[St.best]; let e = 0; for (let i = 0; i < q.length; i++) if (q[i] === 0) e++;
      const [lo, hi] = O.ranges[c], fl = O.fill[c];
      return `<div class="prow" style="--k:${k}">
        <b>${c}</b>
        <div class="pcv"><canvas data-raw="${c}"></canvas></div>
        <div class="pmid">
          <div class="pmap"><span class="mono">${trNum(lo)} … ${trNum(hi)} ${UNIT[c] || ''}</span><i></i><span class="mono">−1 … 1</span></div>
          <div class="pfill"><span class="chk"></span><span class="mono">%${trNum(100 * e / q.length)} boş → ${fl === 0 ? '0' : trNum(fl)}</span></div>
        </div>
        <div class="pcv"><canvas data-in="${c}" data-fill="${fl}"></canvas></div>
      </div>`;
    }).join('');
    const err = O.err || {};
    el.innerHTML = `<div class="s-onis">
      <div class="phead"><span></span><span>ham</span><span></span><span>ağın girdisi</span></div>
      ${rows}
      <div class="pfoot">
        <div class="psize"><span class="mono">float32</span><b class="old">${gb(O.bytes32)}</b><span class="arr">→</span><span class="mono">uint8</span><b>${gb(O.bytes8)}</b></div>
        <div class="perr mono">${chans.map(c => `±${trNum(err[c] ?? NaN, 2)} ${UNIT[c] || ''}`).join(' · ')}</div>
        <div class="pgray"><span class="mono">−1</span><i style="background:linear-gradient(90deg,${gstops})"></i><span class="mono">1</span></div>
      </div>
    </div>`;
  },
  show(el) { this.resize(el); },
  resize(el) {
    if (!STORM) return;
    el.querySelectorAll('canvas[data-raw]').forEach(cv => drawRawQ(cv, STORM, STORM.best, cv.dataset.raw));
    el.querySelectorAll('canvas[data-in]').forEach(cv => drawInputQ(cv, STORM, STORM.best, cv.dataset.in, +cv.dataset.fill));
  },
};

// 5b · bölme: kümeler, fırtına başına renk şeritleri, kontroller (02 defterindeki check_splits, canlı)
const SPLITS = {train: ['Eğitim', '#131A24', 't'], val: ['Doğrulama', '#0072B2', 'v'], test: ['Test', '#E8950E', 's'], dropped: ['Çıkarılan', '#C5CDD8', 'd']};
const SPCODE = {t: 'train', v: 'val', s: 'test', d: 'dropped'};
const SLIDE_BOLME = {
  id: 'bolme', name: 'Bölme', frames: 1,
  build(el, d) {
    const B = d.bolme;
    if (!B) { el.innerHTML = '<div class="snote">Radar verisi yok</div>'; return; }
    const tot = B.sets.reduce((a, s) => a + s.n, 0);
    const bar = B.sets.map(s => `<i style="flex:${s.n};background:${SPLITS[s.k][1]}"></i>`).join('');
    const rows = B.sets.map(s => `<div class="brow"><i style="background:${SPLITS[s.k][1]}"></i><b>${SPLITS[s.k][0]}</b>
        <span class="mono">${trNum(s.n)}</span><span class="mono dim">${trNum(s.eps)} fırtına</span><span class="mono dim">%${trNum(100 * s.tor / s.n, 1)} hortum</span></div>`).join('');
    const strips = B.strips.map((x, r) => `<div class="strip" style="--k:${r}">${[...x].map(c => `<i style="background:${SPLITS[SPCODE[c]][1]}"></i>`).join('')}</div>`).join('');
    const C = B.checks;
    el.innerHTML = `<div class="s-bolme">
      <div class="bleft">
        <div class="bbar">${bar}</div>
        <div class="brows">${rows}</div>
        ${C ? `<div class="bchk"><div><b>${C.shared}</b><span>ortak fırtına</span></div><div><b>${C.ok}/${C.n}</b><span>kontrol ✓</span></div></div>` : ''}
      </div>
      <div class="bright"><div class="dk">fırtına → küme</div><div class="strips" style="grid-template-rows:repeat(${Math.ceil(B.strips.length / 2)},auto)">${strips}</div></div>
    </div>`;
  },
  show() {},
};

// ---------------------------------------------------------------- bölüm 6: TornadoCNN (aktivasyonlar Python'da torch ile, ana.py _cnn)
const ACT_LUT = (() => { const c = new Uint8ClampedArray(256 * 3); for (let q = 0; q < 256; q++) { const v = CH.CAM.cmap(q / 255); c[q * 3] = v[0]; c[q * 3 + 1] = v[1]; c[q * 3 + 2] = v[2]; } return c; })();
let ACTS = null;
function actsOf(d) {
  const c = d && d.cnn; if (!c) return (ACTS = null);
  if (ACTS && ACTS.src === c) return ACTS;
  const dec = a => ({score: a.score, pooled: b64(a.pooled), layers: a.layers.map(L => { const u = b64(L.x), N = L.h * L.w; return {...L, maps: Array.from({length: L.n}, (_, k) => u.subarray(k * N, (k + 1) * N))}; })});
  return (ACTS = {src: c, before: dec(c.before), after: dec(c.after), params: c.params, thr: c.threshold, label: c.label});
}
function drawMap(cv, m, h, w, ring) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d');
  const raw = document.createElement('canvas'); raw.width = w; raw.height = h;
  const rx = raw.getContext('2d'), img = rx.createImageData(w, h), D_ = img.data;
  for (let k = 0; k < m.length; k++) { const o = m[k] * 3; D_[k * 4] = ACT_LUT[o]; D_[k * 4 + 1] = ACT_LUT[o + 1]; D_[k * 4 + 2] = ACT_LUT[o + 2]; D_[k * 4 + 3] = 255; }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.clearRect(0, 0, W, H); g.drawImage(raw, 0, 0, W, H);
  if (ring) {   // ring: [i, j] 120×240 girdisinde → bu haritada
    const x = (ring[1] + .5) / 240 * W, y = (ring[0] + .5) / 120 * H;
    g.lineWidth = 2 * dpr; g.strokeStyle = css('--ink'); g.beginPath(); g.arc(x, y, Math.max(7 * dpr, W / 16), 0, 2 * Math.PI); g.stroke();
  }
}
const actStops = () => Array.from({length: 9}, (_, k) => { const c = CH.CAM.cmap(k / 8); return `rgb(${c.map(x => x | 0)}) ${k * 12.5}%`; }).join(',');
const scoreHTML = (s, thr, tag) => `<div class="sgauge"><div class="sval">${s == null ? '—' : trNum(s, 2)}</div>
  <div class="strack"><i style="width:${100 * (s ?? 0)}%"></i>${thr != null ? `<s style="left:${100 * thr}%"><em>eşik ${trNum(thr, 2)}</em></s>` : ''}</div>
  <div class="stag">${tag}</div></div>`;

// --- 6a · ileri geçiş, eğitilmemiş ağ: her kare tek bir işlem (Conv 3×3 → ReLU → MaxPool ×4, Global Max, Dense …)
const DIV = (() => { const c = new Uint8ClampedArray(256 * 3), neg = hex('#2F6DB5'), mid = hex('#F4F6F9'), pos = hex('#D4483B');
  for (let q = 0; q < 256; q++) { const t = (q - 128) / 127, a = t < 0 ? neg : pos, k = Math.min(1, Math.abs(t)); for (let i = 0; i < 3; i++) c[q * 3 + i] = mid[i] + (a[i] - mid[i]) * k; } return c; })();
function drawSigned(cv, m, h, w, relu) {   // m: uint8, 128 = 0. relu: yalnız pozitif kısım, kendi en büyüğüne göre
  const [W, H] = fitCanvas(cv), g = cv.getContext('2d'), raw = document.createElement('canvas'); raw.width = w; raw.height = h;
  const rx = raw.getContext('2d'), img = rx.createImageData(w, h), D_ = img.data;
  let mx = 1; if (relu) for (const v of m) mx = Math.max(mx, v - 128);
  for (let k = 0; k < m.length; k++) {
    const lut = relu ? ACT_LUT : DIV, q = relu ? Math.round(Math.max(0, m[k] - 128) / mx * 255) : m[k], o = q * 3;
    D_[k * 4] = lut[o]; D_[k * 4 + 1] = lut[o + 1]; D_[k * 4 + 2] = lut[o + 2]; D_[k * 4 + 3] = 255;
  }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.clearRect(0, 0, W, H); g.drawImage(raw, 0, 0, W, H);
}
const divStops = () => [0, .25, .5, .75, 1].map(k => { const q = Math.round(k * 255), o = q * 3; return `rgb(${DIV[o]},${DIV[o + 1]},${DIV[o + 2]}) ${k * 100}%`; }).join(',');
function drawCoord(cv, row) {   // CoordConv kanalı: mesafe boyunca değişir, azimut boyunca sabit (uint8, 0 = −1)
  const [W, H] = fitCanvas(cv), g = cv.getContext('2d'), raw = document.createElement('canvas'); raw.width = row.length; raw.height = 1;
  const rx = raw.getContext('2d'), img = rx.createImageData(row.length, 1);
  for (let j = 0; j < row.length; j++) { const c = GRAY(row[j] / 255 * 2 - 1); img.data.set([c[0], c[1], c[2], 255], j * 4); }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(raw, 0, 0, W, H);
}
const OPS = {in: 'Girdi', conv: 'Conv 3×3', relu: 'ReLU', pool: 'MaxPool 2×2', gmp: 'Global Max', dense: 'Dense', drelu: 'ReLU', drop: 'Dropout', out: 'Dense', sig: 'Sigmoid'};
function cnnSteps(A) {
  const L = A.before.layers, st = [{op: 'in', shape: [3 + (A.src.coord ? 2 : 0), 120, 240]}];
  L.forEach((l, b) => { st.push({op: 'conv', b, shape: [l.c, l.ch, l.cw]}, {op: 'relu', b, shape: [l.c, l.ch, l.cw]}, {op: 'pool', b, shape: [l.c, l.h, l.w]}); });
  const nd = A.dense ? A.dense.length : 64, c = L.length ? L[L.length - 1].c : 128;
  st.push({op: 'gmp', shape: [c]}, {op: 'dense', shape: [nd], from: c}, {op: 'drelu', shape: [nd]}, {op: 'drop', shape: [nd]}, {op: 'out', shape: [1], from: nd}, {op: 'sig', shape: [1]});
  return st;
}
const opNote = (s, A) => ({in: 'DBZ · VEL · WIDTH' + (A.src.coord ? ' + radara uzaklık r · 1/r' : ''), conv: `${s.shape[0]} filtre · her biri 3×3 komşuluk`, relu: 'max(0, x)', pool: '2×2 pencereden en büyük değer',
  gmp: 'her haritadan en büyük değer', dense: `${s.from} → ${s.shape[0]}`, drelu: 'max(0, x)', drop: `%${trNum(100 * (A.before.dropout ?? .3))} · yalnız eğitimde`,
  out: `${s.from} → 1 · logit ${trNum(A.before.logit ?? 0, 2)}`, sig: `1 / (1 + e⁻ˣ) = ${trNum(A.before.score, 2)}`})[s.op];
const LEVEL_W = [100, 84, 68, 54, 42];      // uzamsal seviye 0 (120 × 240) … 4 (7 × 15): yığının genişliği (%)

// Alttaki şemada etkin işlemin küçük çizimi (canvas, sürekli)
function drawOp(cv, s, t, A) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  const ink = css('--ink'), acc = css('--accent'), faint = css('--line'), op = ({conv7: 'conv', bnrelu: 'relu', pool3: 'pool', fcsig: 'sig'})[s.op] || s.op, cyc = (t % 4000) / 4000;
  const cell = Math.min(W / 16, H / 8.5), rnd = (i, j) => Math.abs(Math.sin(i * 12.99 + j * 78.23) * 43758.5) % 1;
  const shade = v => `rgba(19,26,36,${.08 + .6 * Math.abs(v)})`;
  g.lineWidth = 1.2 * dpr; g.font = `600 ${12 * dpr}px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'middle';
  const grid = (x0, y0, n, m, f) => { for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { g.fillStyle = f(i, j); g.fillRect(x0 + j * cell, y0 + i * cell, cell - 2, cell - 2); } };
  if (op === 'in') {
    ['#E8950E', '#11906A', '#7B5BE0'].forEach((c, k) => { g.fillStyle = c; g.globalAlpha = .85; g.fillRect(W / 2 - 3 * cell + k * cell * .7, H / 2 - 2 * cell + k * cell * .7, 5 * cell, 2.5 * cell); });
    g.globalAlpha = 1;
  } else if (op === 'conv') {                    // 3×3 pencere kayar, çıktı hücre hücre dolar
    const n = 7, o = n - 2, x0 = W * .12, y0 = (H - n * cell) / 2, x1 = W * .62, y1 = (H - o * cell) / 2, step = Math.floor(cyc * o * o), si = Math.floor(step / o), sj = step % o;
    grid(x0, y0, n, n, (i, j) => shade(rnd(i, j)));
    grid(x1, y1, o, o, (i, j) => i * o + j <= step ? shade(rnd(i + 9, j + 3) * 1.2) : 'rgba(19,26,36,.04)');
    g.strokeStyle = acc; g.lineWidth = 3 * dpr; g.strokeRect(x0 + sj * cell - 1, y0 + si * cell - 1, 3 * cell, 3 * cell); g.strokeRect(x1 + sj * cell - 1, y1 + si * cell - 1, cell, cell);
    g.lineWidth = 1.5 * dpr; g.setLineDash([4 * dpr, 4 * dpr]); g.beginPath(); g.moveTo(x0 + (sj + 3) * cell, y0 + (si + 1.5) * cell); g.lineTo(x1 + sj * cell, y1 + (si + .5) * cell); g.stroke(); g.setLineDash([]);
  } else if (op === 'relu' || op === 'drelu') {  // y = max(0, x): negatifler sıfıra iner
    const x0 = W * .2, x1 = W * .8, yb = H * .82, yt = H * .12, X = v => x0 + (x1 - x0) * (v + 1) / 2, Y = v => yb - (yb - yt) * v;
    g.strokeStyle = faint; g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x1, Y(0)); g.moveTo(X(0), yt); g.lineTo(X(0), yb); g.stroke();
    g.strokeStyle = ink; g.lineWidth = 3 * dpr; g.beginPath(); g.moveTo(X(-1), Y(0)); g.lineTo(X(0), Y(0)); g.lineTo(X(1), Y(1)); g.stroke();
    for (let k = 0; k < 9; k++) { const v = (k - 4) / 4.5, p = Math.min(1, cyc * 2); g.fillStyle = v < 0 ? '#2F6DB5' : '#D4483B';
      g.beginPath(); g.arc(X(v), lerp(Y(v * .9 + .05), Y(Math.max(0, v)), p), 5 * dpr, 0, 2 * Math.PI); g.fill(); }
  } else if (op === 'pool') {                    // 2×2 pencereden en büyük
    const n = 4, x0 = W * .16, y0 = (H - n * cell) / 2, x1 = W * .64, y1 = (H - 2 * cell) / 2, k = Math.floor(cyc * 4), bi = Math.floor(k / 2), bj = k % 2;
    const v = (i, j) => Math.round(rnd(i + 1, j + 2) * 9);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { g.fillStyle = shade(v(i, j) / 9); g.fillRect(x0 + j * cell, y0 + i * cell, cell - 2, cell - 2); g.fillStyle = ink; g.fillText(v(i, j), x0 + (j + .5) * cell - 1, y0 + (i + .5) * cell - 1); }
    for (let q = 0; q <= k; q++) { const a = Math.floor(q / 2), b = q % 2, m = Math.max(v(2 * a, 2 * b), v(2 * a + 1, 2 * b), v(2 * a, 2 * b + 1), v(2 * a + 1, 2 * b + 1));
      g.fillStyle = shade(m / 9); g.fillRect(x1 + b * cell, y1 + a * cell, cell - 2, cell - 2); g.fillStyle = ink; g.fillText(m, x1 + (b + .5) * cell - 1, y1 + (a + .5) * cell - 1); }
    g.strokeStyle = acc; g.lineWidth = 3 * dpr; g.strokeRect(x0 + 2 * bj * cell - 1, y0 + 2 * bi * cell - 1, 2 * cell, 2 * cell); g.strokeRect(x1 + bj * cell - 1, y1 + bi * cell - 1, cell, cell);
  } else if (op === 'gmp') {                     // her haritadan tek sayı
    const n = 3, m = 6, x0 = W * .12, x1 = W * .72;
    for (let r = 0; r < 3; r++) { const y0 = H * .14 + r * (n * cell * .62 + 10 * dpr), c2 = cell * .6;
      let bi = 0, bj = 0, bv = -1; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = rnd(i + r * 5, j); if (v > bv) { bv = v; bi = i; bj = j; } g.fillStyle = shade(v); g.fillRect(x0 + j * c2, y0 + i * c2, c2 - 2, c2 - 2); }
      g.strokeStyle = acc; g.lineWidth = 2.5 * dpr; g.strokeRect(x0 + bj * c2 - 1, y0 + bi * c2 - 1, c2, c2);
      g.fillStyle = shade(bv); g.fillRect(x1, y0 + n * c2 / 2 - c2 / 2, c2, c2);
      g.strokeStyle = faint; g.beginPath(); g.moveTo(x0 + m * c2 + 8 * dpr, y0 + n * c2 / 2); g.lineTo(x1 - 8 * dpr, y0 + n * c2 / 2); g.stroke(); }
  } else if (['dense', 'drop', 'out'].includes(op)) {   // tam bağlantı; dropout'ta bazı nöronlar kapanır
    const nL = 7, nR = op === 'out' ? 1 : 5, xL = W * .25, xR = W * .72, yL = i => H * (.1 + .8 * (i + .5) / nL), yR = i => H * (.1 + .8 * (i + .5) / nR);
    const off = i => op === 'drop' && rnd(i, Math.floor(t / 600)) < .3;
    g.strokeStyle = 'rgba(19,26,36,.18)'; g.lineWidth = dpr;
    for (let i = 0; i < nL; i++) for (let j = 0; j < nR; j++) if (!off(i)) { g.beginPath(); g.moveTo(xL, yL(i)); g.lineTo(xR, yR(j)); g.stroke(); }
    for (let i = 0; i < nL; i++) { g.fillStyle = off(i) ? faint : ink; g.beginPath(); g.arc(xL, yL(i), 6 * dpr, 0, 2 * Math.PI); g.fill(); }
    for (let j = 0; j < nR; j++) { g.fillStyle = acc; g.beginPath(); g.arc(xR, yR(j), 7 * dpr, 0, 2 * Math.PI); g.fill(); }
  } else if (op === 'res') {                     // residual blok: F(x) + x
    const y = H * .62, yS = H * .18, xA = W * .08, xB = W * .3, xC = W * .55, xP = W * .78, xO = W * .94, bw = W * .2, bh = H * .3;
    g.strokeStyle = ink; g.lineWidth = 2 * dpr;
    g.beginPath(); g.moveTo(xA, y); g.lineTo(xP - 12 * dpr, y); g.moveTo(xP + 12 * dpr, y); g.lineTo(xO, y); g.stroke();
    g.strokeStyle = acc; g.lineWidth = 3 * dpr; g.beginPath(); g.moveTo(xA + W * .05, y); g.lineTo(xA + W * .05, yS); g.lineTo(xP, yS); g.lineTo(xP, y - 12 * dpr); g.stroke();
    [[xB, 'Conv · BN · ReLU'], [xC, 'Conv · BN']].forEach(([x, nm]) => { g.fillStyle = '#fff'; g.strokeStyle = ink; g.lineWidth = 1.5 * dpr; g.fillRect(x - bw / 2, y - bh / 2, bw, bh); g.strokeRect(x - bw / 2, y - bh / 2, bw, bh);
      g.fillStyle = ink; g.font = `600 ${10 * dpr}px ${MONO}`; g.fillText(nm, x, y); });
    g.strokeStyle = ink; g.lineWidth = 2 * dpr; g.beginPath(); g.arc(xP, y, 12 * dpr, 0, 2 * Math.PI); g.stroke(); g.beginPath(); g.moveTo(xP - 6 * dpr, y); g.lineTo(xP + 6 * dpr, y); g.moveTo(xP, y - 6 * dpr); g.lineTo(xP, y + 6 * dpr); g.stroke();
    g.font = `600 ${13 * dpr}px ${MONO}`; g.fillStyle = acc; g.fillText('x', (xA + xP) / 2, yS - 12 * dpr); g.fillStyle = ink; g.fillText('F(x)', (xC + xP) / 2 + bw * .15, y + bh / 2 + 14 * dpr); g.fillText('F(x) + x', xO - 10 * dpr, y - 22 * dpr);
    const p = cyc, dot = (x, yy, c) => { g.fillStyle = c; g.beginPath(); g.arc(x, yy, 6 * dpr, 0, 2 * Math.PI); g.fill(); };
    if (p < .85) { const k = p / .85; dot(lerp(xA + W * .05, xP, k), yS, acc); dot(lerp(xA, xP, k), y, ink); } else dot(lerp(xP, xO, (p - .85) / .15), y, css('--tor'));
  } else if (op === 'gap') {                     // her haritadan ortalama
    const n = 3, m = 6, x0 = W * .12, x1 = W * .72;
    for (let r = 0; r < 3; r++) { const y0 = H * .14 + r * (n * cell * .62 + 10 * dpr), c2 = cell * .6; let sum = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = rnd(i + r * 5, j); sum += v; g.fillStyle = shade(v); g.fillRect(x0 + j * c2, y0 + i * c2, c2 - 2, c2 - 2); }
      g.strokeStyle = acc; g.lineWidth = 2 * dpr; g.strokeRect(x0 - 2, y0 - 2, m * c2 + 2, n * c2 + 2);
      g.fillStyle = shade(sum / (n * m)); g.fillRect(x1, y0 + n * c2 / 2 - c2 / 2, c2, c2);
      g.strokeStyle = faint; g.lineWidth = 1.2 * dpr; g.beginPath(); g.moveTo(x0 + m * c2 + 8 * dpr, y0 + n * c2 / 2); g.lineTo(x1 - 8 * dpr, y0 + n * c2 / 2); g.stroke(); }
  } else if (op === 'sig') {                     // gerçek logit → skor
    const x0 = W * .12, x1 = W * .88, yb = H * .85, yt = H * .12, X = v => x0 + (x1 - x0) * (v + 6) / 12, Y = v => yb - (yb - yt) * v;
    g.strokeStyle = faint; g.beginPath(); g.moveTo(x0, Y(.5)); g.lineTo(x1, Y(.5)); g.moveTo(X(0), yt); g.lineTo(X(0), yb); g.stroke();
    g.strokeStyle = ink; g.lineWidth = 3 * dpr; g.beginPath(); for (let v = -6; v <= 6.01; v += .1) g.lineTo(X(v), Y(1 / (1 + Math.exp(-v)))); g.stroke();
    const lg = clamp(A.before.logit ?? 0, -6, 6), sc = 1 / (1 + Math.exp(-lg)); g.fillStyle = css('--tor'); g.beginPath(); g.arc(X(lg), Y(sc), 7 * dpr, 0, 2 * Math.PI); g.fill();
  }
}

const SLIDE_CNN = {
  id: 'cnn', name: 'TornadoCNN', delay: 1600,
  get frames() { return ACTS ? cnnSteps(ACTS).length : 1; },
  build(el, d) {
    const A = actsOf(d), St = stormOf(d);
    if (!A || !St) { el.innerHTML = '<div class="snote">Model ya da radar verisi yok</div>'; return; }
    A.dense = A.src.before.dense ? b64(A.src.before.dense) : null; delete el.dataset.drawn;
    const L = A.before.layers, steps = cnnSteps(A);
    const co = A.src.before.coord || [];
    const inp = ['DBZ', 'VEL', 'WIDTH'].filter(c => St.ch[c]).map(c => `<div class="nmap in"><canvas data-in="${c}"></canvas></div>`).join('')
      + co.map((_, k) => `<div class="nmap in co"><canvas data-coord="${k}"></canvas><span>${k ? '1/r' : 'r'}</span></div>`).join('');
    const blocks = L.map((l, b) => `<div class="ncol" data-col="b${b}">
        <div class="nhead"><b>${l.c}</b><span class="mono" data-v="sz"></span></div>
        <div class="nstack">${l.maps.map((_, k) => `<div class="nmap" style="--o:${k}"><canvas data-b="${b}" data-k="${k}"></canvas></div>`).join('')}</div>
      </div>`).join('<div class="narr" aria-hidden="true">→</div>');
    // şema: işlemler bloklara göre gruplanmış çipler
    const chip = (s, i) => `<span class="ochip" data-i="${i}">${OPS[s.op]}${s.op === 'conv' || s.op === 'dense' || s.op === 'out' ? ` <em>${s.shape[0]}</em>` : ''}</span>`;
    const groups = [[0, 1, 'girdi']].concat(L.map((_, b) => [1 + 3 * b, 4 + 3 * b, `blok ${b + 1}`]), [[1 + 3 * L.length, steps.length, 'sınıflandırıcı']]);
    const scheme = groups.map(([a, z, nm]) => `<div class="ogrp"><div class="ochips">${steps.slice(a, z).map((s, k) => chip(s, a + k)).join('<i>›</i>')}</div><span class="olab">${nm}</span></div>`).join('');
    el.innerHTML = `<div class="s-cnn2"><div class="nlabel">${esc(A.label || '')}</div>
      <div class="nflow">
        <div class="ncol" data-col="in"><div class="nhead"><b>${3 + co.length}</b><span class="mono">120 × 240</span></div><div class="ninput${co.length ? ' five' : ''}">${inp}</div></div>
        <div class="narr">→</div>${blocks}<div class="narr" aria-hidden="true">→</div>
        <div class="ncol" data-col="gmp"><div class="nhead"><b>${L.length ? L[L.length - 1].c : ''}</b></div><div class="npool"><canvas data-v="gmp"></canvas></div></div>
        <div class="narr" aria-hidden="true">→</div>
        <div class="ncol" data-col="dense"><div class="nhead"><b>${A.dense ? A.dense.length : 64}</b></div><div class="npool sm"><canvas data-v="dense"></canvas></div></div>
        <div class="narr" aria-hidden="true">→</div>
        <div class="ncol nscore" data-col="out"><div class="nhead"><b>1</b></div><div data-v="score"></div></div>
      </div>
      <div class="nscheme">
        <div class="oflow">${scheme}</div>
        <div class="odetail">
          <div class="oname" data-v="op"></div>
          <div class="ocv"><canvas data-v="opcv"></canvas></div>
          <div class="oinfo"><div class="oshape mono" data-v="shape"></div><div class="onote" data-v="note"></div>
            <div class="oleg" data-v="leg"></div></div>
        </div>
      </div>
    </div>`;
  },
  show(el, f) {
    const A = ACTS, St = STORM; if (!A || !St || !el.querySelector('.nflow')) return;
    if (el.dataset.drawn !== '1') {          // girdi ve global max şeridi bir kez (slayt görünürken boyut bilinir)
      el.dataset.drawn = '1';
      el.querySelectorAll('canvas[data-in]').forEach(cv => drawRect(cv, St, St.best, cv.dataset.in, false));
      el.querySelectorAll('canvas[data-coord]').forEach(cv => drawCoord(cv, b64(A.src.before.coord[+cv.dataset.coord])));
      const pv = el.querySelector('canvas[data-v="gmp"]'); if (pv) drawMap(pv, A.before.pooled, A.before.pooled.length, 1);
    }
    const steps = cnnSteps(A), s = steps[Math.min(f, steps.length - 1)], L = A.before.layers, conv = A.src.before.layers;
    const vis = (k, on) => { const c = el.querySelector(`[data-col="${k}"]`); if (c) c.classList.toggle('on', on); };
    vis('in', true);
    L.forEach((l, b) => {
      const c0 = 1 + 3 * b, col = el.querySelector(`[data-col="b${b}"]`); vis('b' + b, f >= c0);
      if (f < c0) return;
      const stage = f === c0 ? 'conv' : f === c0 + 1 ? 'relu' : 'pool', small = stage === 'pool';
      col.querySelector('.nstack').style.setProperty('--w', LEVEL_W[small ? b + 1 : b] + '%');
      col.querySelector('[data-v="sz"]').textContent = small ? `${l.h} × ${l.w}` : `${l.ch} × ${l.cw}`;
      if (col.dataset.stage !== stage) {
        col.dataset.stage = stage;
        const cm = conv[b].conv ? b64(conv[b].conv) : null, N = (l.ch || 0) * (l.cw || 0);
        col.querySelectorAll('canvas[data-b]').forEach(cv => { const k = +cv.dataset.k;
          if (small || !cm) drawMap(cv, l.maps[k], l.h, l.w); else drawSigned(cv, cm.subarray(k * N, (k + 1) * N), l.ch, l.cw, stage === 'relu'); });
      }
    });
    const g0 = 1 + 3 * L.length; vis('gmp', f >= g0); vis('dense', f >= g0 + 1); vis('out', f >= g0 + 4);
    const dv = el.querySelector('[data-v="dense"]');
    if (A.dense && f >= g0 + 1) { const relu = f >= g0 + 2; if (dv.dataset.stage !== String(relu)) { dv.dataset.stage = String(relu); drawSigned(dv, A.dense, A.dense.length, 1, relu); } }
    el.querySelector('[data-v="score"]').innerHTML = f >= g0 + 5 ? scoreHTML(A.before.score, null, 'eğitilmemiş')
      : f >= g0 + 4 ? `<div class="sgauge"><div class="sval sm">${trNum(A.before.logit ?? 0, 2)}</div><div class="stag">logit</div></div>` : '';
    el.querySelectorAll('.ochip').forEach(c => { const i = +c.dataset.i; c.classList.toggle('done', i < f); c.classList.toggle('cur', i === f); });
    el.querySelector('[data-v="op"]').textContent = OPS[s.op];
    el.querySelector('[data-v="shape"]').textContent = s.shape.join(' × ');
    el.querySelector('[data-v="note"]').textContent = opNote(s, A);
    const leg = s.op === 'conv' || s.op === 'dense' ? `<span class="mono">−</span><i style="background:linear-gradient(90deg,${divStops()})"></i><span class="mono">+</span>`
      : ['relu', 'drelu', 'pool', 'gmp', 'drop'].includes(s.op) ? `<span class="mono">0</span><i style="background:linear-gradient(90deg,${actStops()})"></i><span class="mono">en büyük</span>` : '';
    el.querySelector('[data-v="leg"]').innerHTML = leg;
  },
  resize(el) {
    if (!ACTS || !STORM) return;
    delete el.dataset.drawn; el.querySelectorAll('[data-stage]').forEach(x => { delete x.dataset.stage; });
    this.show(el, DECK.i === SLIDES.indexOf(this) ? DECK.f : 0);
  },
  tick(el, t) { const A = ACTS; if (!A) return; const steps = cnnSteps(A), cv = el.querySelector('[data-v="opcv"]'); if (cv) drawOp(cv, steps[Math.min(DECK.f, steps.length - 1)], t, A); },
};

// --- 6b · eğitim eğrileri (epoch sayısı kadar kare)
function lineChart(xs, series, f, opt) {
  const W = 520, H = 300, L = 46, R = 14, T = 16, B = 34, n = xs.length;
  const all = series.flatMap(s => s.v.filter(v => v != null)); let lo = opt.lo ?? Math.min(...all), hi = opt.hi ?? Math.max(...all);
  const pad = (hi - lo) * .08; lo -= pad; hi += pad;
  const X = i => L + (W - L - R) * (n > 1 ? i / (n - 1) : .5), Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let h = `<svg viewBox="0 0 ${W} ${H}" class="lchart">`;
  for (let k = 0; k <= 4; k++) { const v = lo + (hi - lo) * k / 4; h += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="gl"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${trNum(v, 2)}</text>`; }
  xs.forEach((x, i) => { h += `<text x="${X(i)}" y="${H - 12}" text-anchor="middle">${x}</text>`; });
  if (opt.best && opt.best <= f + 1) { const i = xs.indexOf(opt.best); h += `<line x1="${X(i)}" x2="${X(i)}" y1="${T}" y2="${H - B}" class="bestl"/>`; }
  if (opt.stop && f === n - 1 && opt.best) { const i = xs.indexOf(opt.best); h += `<rect x="${X(i)}" y="${T}" width="${X(n - 1) - X(i)}" height="${H - T - B}" class="stopr"/>`; }
  for (const s of series) {
    const pts = s.v.slice(0, f + 1).map((v, i) => v == null ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(' ');
    h += `<polyline points="${pts}" fill="none" stroke="${s.c}" stroke-width="${s.w || 3}" ${s.dash ? 'stroke-dasharray="6 6"' : ''} stroke-linejoin="round" stroke-linecap="round"/>`;
    const v = s.v[f]; if (v != null) h += `<circle cx="${X(f)}" cy="${Y(v)}" r="5" fill="${s.c}"/>`;
    if (opt.best && opt.best <= f + 1 && s.mark) { const i = xs.indexOf(opt.best); h += `<circle cx="${X(i)}" cy="${Y(s.v[i])}" r="8" fill="none" stroke="${s.c}" stroke-width="2.5"/>`; }
  }
  return h + '</svg>';
}
const SLIDE_EGITIM = {
  id: 'egitim', name: 'Eğitim', delay: 900,
  get frames() { const e = (A && A.ana && A.ana.egitim); return e ? e.epoch.length : 1; },
  build(el, d) {
    const e = d.egitim;
    if (!e) { el.innerHTML = '<div class="snote">Eğitim geçmişi yok</div>'; return; }
    el.innerHTML = `<div class="s-egit">
      <div class="eside">
        <div class="etur"><b data-v="tur"></b><span>/ ${e.max_epochs} tur</span></div>
        <div class="estop mono" data-v="stop"></div>
        ${e.pos_weight ? `<div class="epw"><div class="dk">kayıp ağırlığı</div><div class="pwrow"><i class="tor"></i><b>× ${trNum(e.pos_weight, 1)}</b></div><div class="pwrow"><i></i><b class="dim">× 1</b></div></div>` : ''}
      </div>
      <div class="eplot"><div class="dk">kayıp</div><div data-v="loss"></div><div class="eleg"><span><i class="tr"></i>eğitim</span><span><i class="va"></i>doğrulama</span></div></div>
      <div class="eplot"><div class="dk">doğrulama PR-AUC</div><div data-v="pr"></div></div>
    </div>`;
  },
  show(el, f, d) {
    const e = d.egitim; if (!e) return;
    el.querySelector('[data-v="tur"]').textContent = e.epoch[f];
    el.querySelector('[data-v="loss"]').innerHTML = lineChart(e.epoch, [{v: e.train_loss, c: '#8791A1', dash: true}, {v: e.val_loss, c: css('--ink')}], f, {best: e.best, stop: true});
    el.querySelector('[data-v="pr"]').innerHTML = lineChart(e.epoch, [{v: e.val_pr, c: css('--accent'), mark: true}], f, {best: e.best, stop: true});
    const last = f === e.epoch.length - 1, sinceBest = e.epoch[f] - e.best;
    el.querySelector('[data-v="stop"]').textContent = e.epoch[f] >= e.best ? (last ? `en iyi: ${e.best}. tur · ${e.patience} tur iyileşme yok → dur` : `en iyi: ${e.best}. tur${sinceBest > 0 ? ` · ${sinceBest}/${e.patience}` : ''}`) : '';
  },
};

// --- 6c · eğitimden önce / sonra: aynı fırtına, 3. bloğun en güçlü haritaları ve skor
const SLIDE_ONCESONRA = {
  id: 'oncesonra', name: 'Eğitimden önce / sonra', frames: 1,
  build(el, d) {
    const A = actsOf(d), St = stormOf(d);
    if (!A || !St) { el.innerHTML = '<div class="snote">Model ya da radar verisi yok</div>'; return; }
    const li = Math.min(2, A.after.layers.length - 1), l = A.after.layers[li], nm = 4;
    const row = (key, tag, thr) => `<div class="oprow ${key}">
        <div class="optag">${tag}</div>
        <div class="opin"><canvas data-vel></canvas></div>
        <div class="opmaps">${Array.from({length: nm}, (_, k) => `<div class="nmap"><canvas data-s="${key}" data-k="${k}"></canvas></div>`).join('')}</div>
        ${scoreHTML(A[key].score, thr, '')}
      </div>`;
    el.innerHTML = `<div class="s-once">
      <div class="ophead"><span></span><span>VEL</span><span>${l.c} haritadan en güçlü ${nm} · ${l.h} × ${l.w}</span><span>skor</span></div>
      ${row('before', 'önce', null)}${row('after', 'sonra', A.thr)}
      <div class="opleg"><span class="mono">0</span><i style="background:linear-gradient(90deg,${actStops()})"></i><span class="mono">en güçlü</span></div>
    </div>`;
  },
  show(el) { this.resize(el); },
  resize(el) {
    const A = ACTS, St = STORM; if (!A || !St) return;
    const li = Math.min(2, A.after.layers.length - 1), ring = St.rot[St.best];
    el.querySelectorAll('canvas[data-vel]').forEach(cv => drawRect(cv, St, St.best, 'VEL', true));
    el.querySelectorAll('canvas[data-s]').forEach(cv => { const l = A[cv.dataset.s].layers[li]; drawMap(cv, l.maps[+cv.dataset.k], l.h, l.w, ring); });
  },
};

// --- 6d · hiperparametre denemeleri (04): tek grafik, baseline'a göre fark ve seed bandı
function hpPlot(H) {
  const b = H.base, rows = H.rows, groups = [];
  rows.forEach(r => { const g = groups[groups.length - 1]; if (!g || g.name !== r.group) groups.push({name: r.group, rows: [r]}); else g.rows.push(r); });
  const W = 860, rh = 24, gh = 26, T = 34, L = 230, R = 90;
  const H_ = T + groups.length * gh + rows.length * rh + 40;
  const ds = rows.map(r => r.d), CLIP = -.08;
  const lo = Math.max(CLIP, Math.min(b.min - b.mean, ...ds) - .006), hi = Math.max(b.max - b.mean, ...ds) + .012;
  const X = v => L + (W - L - R) * (clamp(v, lo, hi) - lo) / (hi - lo);
  let h = `<svg viewBox="0 0 ${W} ${H_}" class="hplot">`;
  h += `<rect x="${X(b.min - b.mean)}" y="${T - 10}" width="${X(b.max - b.mean) - X(b.min - b.mean)}" height="${H_ - T - 30}" class="band"/>`;
  h += `<line x1="${X(0)}" x2="${X(0)}" y1="${T - 10}" y2="${H_ - 40}" class="zero"/><text x="${X(0)}" y="${T - 16}" text-anchor="middle" class="ax">baseline · ${b.n} seed</text>`;
  for (let t = Math.ceil(lo * 50) / 50; t <= hi + 1e-9; t += .02) { const v = Math.round(t * 100) / 100; if (v) h += `<text x="${X(v)}" y="${H_ - 22}" text-anchor="middle" class="ax">${v > 0 ? '+' : '−'}${trNum(Math.abs(v), 2)}</text>`; }
  h += `<text x="${(X(b.min - b.mean) + X(b.max - b.mean)) / 2}" y="${H_ - 6}" text-anchor="middle" class="ax">seed aralığı</text>`;
  let y = T;
  for (const g of groups) {
    h += `<text x="12" y="${y + 17}" class="grp">${esc(g.name)}</text>`; y += gh;
    for (const r of g.rows) {
      const cy = y + rh / 2, out = r.v > b.max ? 'up' : r.v < b.min ? 'down' : 'in', clipped = r.d < lo;
      h += `<g class="hr ${out}"><text x="${L - 14}" y="${cy + 4}" text-anchor="end" class="lb">${esc(r.label)}</text>`;
      h += `<line x1="${X(0)}" x2="${X(r.d)}" y1="${cy}" y2="${cy}"/>`;
      h += clipped ? `<path d="M${X(lo) + 10},${cy - 7} L${X(lo)},${cy} L${X(lo) + 10},${cy + 7} Z"/>` : `<circle cx="${X(r.d)}" cy="${cy}" r="6"/>`;
      const tx = r.d >= 0 ? X(r.d) + 12 : clipped ? X(lo) + 16 : X(0) + 8;
      h += `<text x="${tx}" y="${clipped ? cy - 8 : cy + 4}" class="val">${r.d >= 0 ? '+' : '−'}${trNum(Math.abs(r.d), 3)}</text></g>`;
      y += rh;
    }
  }
  return h + '</svg>';
}
const SLIDE_HP = {
  id: 'hp', name: 'Hiperparametreler', frames: 1,
  build(el, d) {
    const H = d.hp;
    if (!H || !H.rows.length) { el.innerHTML = '<div class="snote">Hiperparametre denemesi yok</div>'; return; }
    const up = H.rows.filter(r => r.v > H.base.max), down = H.rows.filter(r => r.v < H.base.min), mid = H.rows.length - up.length - down.length;
    el.innerHTML = `<div class="s-hp">
      <div class="hleft"><div class="dk">doğrulama PR-AUC · baseline'a göre fark · her satırda tek bir ayar değişti</div>${hpPlot(H)}</div>
      <div class="hside">
        <div class="hnum up"><b>${up.length}</b><span>iyileşme</span></div>
        <div class="hnum"><b>${mid}</b><span>fark yok</span></div>
        <div class="hnum down"><b>${down.length}</b><span>kötüleşme</span></div>
        <div class="hverd">${up.length ? `bandı aşan: ${up.map(r => esc(r.label)).join(', ')}` : 'varsayılan ayarlar korundu'}</div>
      </div>
    </div>`;
  },
  show() {},
};

// --- 6d · varyantlar: her kare bir varyant (mini animasyon) + doğrulama PR-AUC farkı (seed aralığıyla)
const VORDER = ['avg', 'bn', 'coord', 'aug', 'ls', 'mit'];
const VNAME = {avg: 'Global Avg', bn: 'BatchNorm', coord: 'CoordConv', aug: 'Augmentation', ls: 'Label smoothing', mit: 'TorNet ağırlıkları'};
const vrows = d => ((d.varyant && d.varyant.rows) || []).slice().sort((a, b) => VORDER.indexOf(a.kind) - VORDER.indexOf(b.kind));
function drawVariant(cv, kind, t, d) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  const V = d.varyant || {}, St = STORM, A = ACTS, k = (Math.sin(t / 900) + 1) / 2, ink = css('--ink'), acc = css('--accent');
  g.font = `600 ${14 * dpr}px ${MONO}`; g.textBaseline = 'middle';
  if (kind === 'avg' && A) {               // gerçek bir harita (son blok): en büyük değer ve ortalama
    const m = A.after.layers[A.after.layers.length - 1].maps[0], n = m.length, bw = (W - 40 * dpr) / n, base = H * .85, top = H * .12;
    let mx = 0, sum = 0; for (const v of m) { mx = Math.max(mx, v); sum += v; }
    for (let i = 0; i < n; i++) { g.fillStyle = 'rgba(85,96,111,.35)'; const h = (base - top) * m[i] / 255; g.fillRect(20 * dpr + i * bw, base - h, Math.max(bw - 1, 1), h); }
    const yMax = base - (base - top) * mx / 255, yAvg = base - (base - top) * sum / n / 255;
    g.lineWidth = 3 * dpr; g.strokeStyle = ink; g.beginPath(); g.moveTo(20 * dpr, yMax); g.lineTo(W - 20 * dpr, yMax); g.stroke();
    g.strokeStyle = acc; g.beginPath(); g.moveTo(20 * dpr, yAvg); g.lineTo(W - 20 * dpr, yAvg); g.stroke();
    g.fillStyle = ink; g.textAlign = 'right'; g.fillText('max', W - 20 * dpr, yMax - 12 * dpr); g.fillStyle = acc; g.fillText('ort.', W - 20 * dpr, yAvg - 12 * dpr);
  } else if (kind === 'bn') {              // aktivasyon dağılımı ortalanıp ölçeklenir
    const mu = lerp(1.6, 0, k), sd = lerp(2.2, 1, k), x0 = -6, x1 = 6, base = H * .85;
    g.beginPath(); for (let i = 0; i <= 200; i++) { const x = x0 + (x1 - x0) * i / 200, y = Math.exp(-((x - mu) ** 2) / (2 * sd * sd)) / sd; g.lineTo(W * i / 200, base - y * H * .6); }
    g.lineTo(W, base); g.lineTo(0, base); g.closePath(); g.fillStyle = 'rgba(19,26,36,.18)'; g.fill(); g.strokeStyle = ink; g.lineWidth = 2.5 * dpr; g.stroke();
    g.strokeStyle = acc; g.setLineDash([5 * dpr, 5 * dpr]); g.beginPath(); g.moveTo(W / 2, H * .1); g.lineTo(W / 2, base); g.stroke(); g.setLineDash([]);
    g.fillStyle = acc; g.textAlign = 'center'; g.fillText('0', W / 2, base + 14 * dpr);
  } else if (kind === 'coord' && St) {     // girdiye radara uzaklık kanalları eklenir
    const rw = W * .44, rh = rw / 2, gx = (W - rw) / 2;
    const cvs = document.createElement('canvas'); cvs.width = St.NR; cvs.height = St.NA;
    const cx = cvs.getContext('2d'), img = cx.createImageData(St.NR, St.NA), q = St.ch.DBZ.frames[St.best];
    for (let p = 0; p < q.length; p++) { const o = q[p] * 3; img.data.set([St.ch.DBZ.rgb[o], St.ch.DBZ.rgb[o + 1], St.ch.DBZ.rgb[o + 2], 255], p * 4); }
    cx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(cvs, gx, H * .06, rw, rh);
    const fr = St.frames[St.best], a = .2 + .8 * Math.min(1, k * 1.5), rA = Math.max(fr.r0, 0), rB = fr.r0 + St.NR * St.DR;
    [['r', r => clamp(r, 0, 250) / 250], ['1/r', r => 10 / Math.max(r, 10)]].forEach(([nm, fn], i) => {
      const y = H * .06 + rh + 14 * dpr + i * (rh * .45 + 10 * dpr), v0 = fn(rA), v1 = fn(rB);   // pencere içinde gösterim için gerilir
      for (let x = 0; x < rw; x++) { const r = fr.r0 + x / rw * St.NR * St.DR, v = (fn(r) - Math.min(v0, v1)) / (Math.abs(v1 - v0) || 1); g.fillStyle = `rgba(${lerp(240, 19, v)},${lerp(242, 26, v)},${lerp(246, 36, v)},${a})`; g.fillRect(gx + x, y, 1.5, rh * .45); }
      g.globalAlpha = a; g.fillStyle = ink; g.textAlign = 'right'; g.fillText(nm, gx - 10 * dpr, y + rh * .22); g.globalAlpha = 1;
    });
  } else if (kind === 'aug' && St) {       // kaydırma ve gürültü
    const rw = W * .7, rh = rw / 2, gx = (W - rw) / 2, gy = (H - rh) / 2, aug = V.aug || {az: 4, rng: 5};
    const ph = Math.floor(t / 700), rnd = s => (Math.sin(ph * 12.9898 + s * 78.233) * 43758.5453) % 1;
    const dx = Math.round(rnd(1) * aug.rng), dy = Math.round(rnd(2) * aug.az);
    const cvs = document.createElement('canvas'); cvs.width = St.NR; cvs.height = St.NA;
    const cx = cvs.getContext('2d'), img = cx.createImageData(St.NR, St.NA), q = St.ch.DBZ.frames[St.best];
    for (let i = 0; i < St.NA; i++) for (let j = 0; j < St.NR; j++) {
      const si = i - dy, sj = j - dx, p = (i * St.NR + j) * 4, v = si >= 0 && si < St.NA && sj >= 0 && sj < St.NR ? q[si * St.NR + sj] : 0;
      const n = v ? clamp(v + Math.round(rnd(i * 7 + j) * 6), 1, 255) : 0, o = n * 3; img.data[p] = St.ch.DBZ.rgb[o]; img.data[p + 1] = St.ch.DBZ.rgb[o + 1]; img.data[p + 2] = St.ch.DBZ.rgb[o + 2]; img.data[p + 3] = 255;
    }
    cx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(cvs, gx, gy, rw, rh);
    g.strokeStyle = 'rgba(40,55,80,.4)'; g.lineWidth = dpr; g.strokeRect(gx, gy, rw, rh);
    g.fillStyle = css('--muted'); g.textAlign = 'center'; g.fillText(`±${aug.az} açı · ±${aug.rng} mesafe hücresi`, W / 2, gy + rh + 18 * dpr);
  } else if (kind === 'ls') {              // hortum hedefi 1 → 1 − ls
    const row = (V.rows || []).find(r => r.kind === 'ls'), ls = row ? row.ls : .1, tgt = lerp(1, 1 - ls, Math.min(1, k * 1.4));
    const bx = W * .3, bw = W * .4, base = H * .85, top = H * .12;
    g.fillStyle = 'rgba(19,26,36,.08)'; g.fillRect(bx, top, bw, base - top);
    g.fillStyle = css('--tor'); g.fillRect(bx, base - (base - top) * tgt, bw, (base - top) * tgt);
    g.fillStyle = '#fff'; g.textAlign = 'center'; g.font = `800 ${30 * dpr}px "Big Shoulders Display", sans-serif`; g.fillText(trNum(tgt, 2), W / 2, top + (base - top) * (1 - tgt) + 26 * dpr);
    g.font = `600 ${13 * dpr}px ${MONO}`; g.fillStyle = css('--muted'); g.fillText('TOR hedefi', W / 2, base + 16 * dpr);
  } else if (kind === 'mit') {             // kategori ağırlıkları (train.class_weights)
    const w = V.mit || {}, ks = Object.keys(w), mx = Math.max(...Object.values(w), 1), base = H * .8, top = H * .14, bw = W / (ks.length * 1.6);
    const col = {NUL: css('--nul'), WRN: css('--wrn')};
    ks.forEach((c, i) => { const x = W / 2 + (i - (ks.length - 1) / 2) * bw * 1.6 - bw / 2, h = (base - top) * w[c] / mx * Math.min(1, k * 1.5 + .05);
      g.fillStyle = col[c] || css('--tor'); g.fillRect(x, base - h, bw, h);
      g.fillStyle = ink; g.textAlign = 'center'; g.fillText(`× ${trNum(w[c], 1)}`, x + bw / 2, base - h - 12 * dpr); g.fillStyle = css('--muted'); g.fillText(c, x + bw / 2, base + 16 * dpr); });
  }
}
function deltaPlot(d, f) {
  const V = d.varyant, rows = vrows(d), b = V.base, W = 560, rh = 46, T = 30, H = T + rows.length * rh + 48, L = 150, R = 70;
  const ds = rows.map(r => r.d).concat([b.min - b.mean, b.max - b.mean]), m = Math.max(.03, ...ds.map(Math.abs)) * 1.15;
  const X = v => L + (W - L - R) * (v + m) / (2 * m);
  let h = `<svg viewBox="0 0 ${W} ${H}" class="dplot"><rect x="${X(b.min - b.mean)}" y="${T - 8}" width="${X(b.max - b.mean) - X(b.min - b.mean)}" height="${rows.length * rh + 8}" class="band"/>`;
  h += `<line x1="${X(0)}" x2="${X(0)}" y1="${T - 8}" y2="${T + rows.length * rh}" class="zero"/><text x="${X(0)}" y="${T - 14}" text-anchor="middle" class="ax">baseline · ${b.n} seed</text>`;
  const stp = m > .03 ? .02 : .01;
  for (let r = -Math.floor(m / stp) * stp; r <= m + 1e-9; r += stp) { const v = Math.round(r * 100) / 100; if (v) h += `<text x="${X(v)}" y="${H - 26}" text-anchor="middle" class="ax">${v > 0 ? '+' : '−'}${trNum(Math.abs(v), 2)}</text>`; }
  rows.forEach((r, i) => {
    if (i > f) return;
    const y = T + i * rh + rh / 2, out = r.v > b.max ? 'up' : r.v < b.min ? 'down' : 'in', cur = i === f;
    h += `<g class="dr ${out}${cur ? ' cur' : ''}${r.chosen ? ' chosen' : ''}"><text x="${L - 14}" y="${y + 5}" text-anchor="end" class="nm">${r.chosen ? '★ ' : ''}${esc(VNAME[r.kind] || r.label)}</text>
      <line x1="${X(0)}" x2="${X(r.d)}" y1="${y}" y2="${y}"/><circle cx="${X(r.d)}" cy="${y}" r="${cur ? 9 : 7}"/>
      <text x="${r.d >= 0 ? X(r.d) + 16 : X(0) + 10}" y="${y + 5}" text-anchor="start" class="val">${r.d >= 0 ? '+' : '−'}${trNum(Math.abs(r.d), 3)}</text></g>`;
  });
  return h + `<text x="${(X(b.min - b.mean) + X(b.max - b.mean)) / 2}" y="${H - 8}" text-anchor="middle" class="ax">seed aralığı</text></svg>`;
}
const SLIDE_VARYANT = {
  id: 'varyant', name: 'Varyantlar', delay: 2200,
  get frames() { return A && A.ana ? Math.max(1, vrows(A.ana).length) : 1; },
  build(el, d) {
    if (!d.varyant || !vrows(d).length) { el.innerHTML = '<div class="snote">Varyant sonucu yok</div>'; return; }
    el.innerHTML = `<div class="s-var">
      <div class="vleft"><div class="vname"><span data-v="name"></span><em class="vchosen" data-v="chosen" hidden>★ seçilen</em></div><div class="vanim"><canvas></canvas></div></div>
      <div class="vright"><div class="dk">doğrulama PR-AUC · baseline'a göre fark</div><div data-v="plot"></div></div>
    </div>`;
  },
  show(el, f, d) {
    const rows = vrows(d), r = rows[Math.min(f, rows.length - 1)]; if (!r) return;
    el.querySelector('[data-v="name"]').textContent = VNAME[r.kind] || r.label;
    el.querySelector('[data-v="chosen"]').hidden = !r.chosen;
    el.querySelector('[data-v="plot"]').innerHTML = deltaPlot(d, f);
  },
  tick(el, t) { const rows = vrows(A.ana || {}), r = rows[Math.min(DECK.f, rows.length - 1)]; if (r) drawVariant(el.querySelector('.vanim canvas'), r.kind, t, A.ana || {}); },
};

// Model adı + rol: ResNet ailesinin baseline'ı (sıfırdan eğitilen) her slaytta "(baseline)" ile anılır
function roleName(key, label) {
  const d = (A && A.ana) || {}, rb = (d.test && d.test.bases && d.test.bases.resnet18 && d.test.bases.resnet18.key) || (d.resnet && d.resnet.base_key);
  return key && rb && key === rb ? `${label} (baseline)` : label;
}
// ---------------------------------------------------------------- bölüm 7: ResNet-18 (ana.py _resnet; başlangıç: yalnız ImageNet ağırlıkları)
let RES = null;
function resOf(d) {
  const c = d && d.resnet; if (!c) return (RES = null);
  if (RES && RES.src === c) return RES;
  const mp = L => { const u = b64(L.x), N = L.h * L.w; return {...L, raw: u, maps: Array.from({length: L.n}, (_, k) => u.subarray(k * N, (k + 1) * N))}; };
  const dec = a => ({score: a.score, logit: a.logit, avg: b64(a.avg), n_avg: a.n_avg, pool: mp(a.pool), stages: a.stages.map(mp),
                     conv1: a.conv1 ? mp(a.conv1) : null, bn1: a.bn1 ? mp(a.bn1) : null});
  return (RES = {src: c, before: dec(c.before), after: dec(c.after), rnd: c.rnd ? dec(c.rnd) : null, other: c.other_last ? mp(c.other_last) : null,
                 params: c.params, thr: c.threshold, label: c.label, filters: c.filters});
}
const ROPS = {in: 'Girdi', conv7: 'Conv 7×7 /2', bnrelu: 'BN → ReLU', pool3: 'MaxPool 3×3 /2', res: 'Aşama', gap: 'Global Avg', fcsig: 'Dense 1 → Sigmoid'};
function resSteps(R) {
  const b = R.rnd || R.before, st = [{op: 'in', shape: [3, 120, 240]}];
  st.push({op: 'conv7', shape: [b.conv1.c, b.conv1.h, b.conv1.w]}, {op: 'bnrelu', shape: [b.bn1.c, b.bn1.h, b.bn1.w]}, {op: 'pool3', shape: [b.pool.c, b.pool.h, b.pool.w]});
  b.stages.forEach((l, k) => st.push({op: 'res', k, shape: [l.c, l.h, l.w]}));
  st.push({op: 'gap', shape: [b.n_avg]}, {op: 'fcsig', shape: [1], from: b.n_avg});
  return st;
}
const resNote = (s, R) => ({in: R.rnd ? 'DBZ · VEL · WIDTH · ağırlıklar rastgele' : R.src.pre ? 'DBZ · VEL · WIDTH (ImageNet\'te R · G · B)' : 'DBZ · VEL · WIDTH', conv7: `${s.shape[0]} filtre · 7×7 · adım 2: boyut yarıya`, bnrelu: 'normalleştir, sonra max(0, x)',
  pool3: '3×3 pencereden en büyük · adım 2', res: `${s.k === 0 ? '' : 'boyut yarıya, '}filtre ${s.shape[0]} · 2 residual blok: F(x) + x`, gap: 'her haritanın ortalaması',
  fcsig: `logit ${trNum((R.rnd || R.before).logit, 2)} → ${trNum((R.rnd || R.before).score, 2)}`})[s.op];
const RES_W = {in: 100, c1: 84, pool: 68, s0: 68, s1: 54, s2: 44, s3: 34};

const SLIDE_RES = {
  id: 'resnet', name: 'ResNet-18', delay: 1700,
  get frames() { return RES ? resSteps(RES).length : 1; },
  build(el, d) {
    const R = resOf(d), St = stormOf(d); delete el.dataset.drawn;
    if (!R || !St) { el.innerHTML = '<div class="snote">ResNet modeli ya da radar verisi yok</div>'; return; }
    const steps = resSteps(R), B = R.rnd || R.before;
    const inp = ['DBZ', 'VEL', 'WIDTH'].filter(c => St.ch[c]).map(c => `<div class="nmap in"><canvas data-in="${c}"></canvas></div>`).join('');
    const stack = (key, n) => `<div class="nstack">${Array.from({length: n}, (_, k) => `<div class="nmap" style="--o:${k}"><canvas data-m="${key}" data-k="${k}"></canvas></div>`).join('')}</div>`;
    const col = (key, b, body) => `<div class="ncol" data-col="${key}"><div class="nhead"><b data-v="c">${b}</b><span class="mono" data-v="sz"></span></div>${body}</div>`;
    const arr = '<div class="narr" aria-hidden="true">→</div>';
    const chip = (s, i) => `<span class="ochip" data-i="${i}">${s.op === 'res' ? `Aşama ${s.k + 1} <em>${s.shape[0]}</em>` : ROPS[s.op]}</span>`;
    const groups = [[0, 1, 'girdi'], [1, 4, 'kök'], [4, 4 + B.stages.length, 'her aşama 2 residual blok'], [4 + B.stages.length, steps.length, 'sınıflandırıcı']];
    const scheme = groups.map(([a, z, nm]) => `<div class="ogrp"><div class="ochips">${steps.slice(a, z).map((s, k) => chip(s, a + k)).join('<i>›</i>')}</div><span class="olab">${nm}</span></div>`).join('');
    el.innerHTML = `<div class="s-cnn2"><div class="nlabel">ResNet-18 · ${R.rnd ? 'rastgele ağırlıklar' : esc(R.label || '')}</div>
      <div class="nflow res">
        ${col('in', 3, `<div class="ninput">${inp}</div>`)}${arr}
        ${col('stem', B.conv1.c, stack('stem', B.conv1.n))}${arr}
        ${B.stages.map((l, k) => col('s' + k, l.c, stack('s' + k, l.n))).join(arr)}${arr}
        ${col('gap', B.n_avg, '<div class="npool"><canvas data-v="gap"></canvas></div>')}${arr}
        <div class="ncol nscore" data-col="out"><div class="nhead"><b>1</b></div><div data-v="score"></div></div>
      </div>
      <div class="nscheme">
        <div class="oflow">${scheme}</div>
        <div class="odetail"><div class="oname" data-v="op"></div><div class="ocv"><canvas data-v="opcv"></canvas></div>
          <div class="oinfo"><div class="oshape mono" data-v="shape"></div><div class="onote" data-v="note"></div><div class="oleg" data-v="leg"></div></div></div>
      </div>
    </div>`;
  },
  show(el, f) {
    const R = RES, St = STORM; if (!R || !St || !el.querySelector('.nflow')) return;
    const B = R.rnd || R.before, steps = resSteps(R), s = steps[Math.min(f, steps.length - 1)];
    // yığın genişlikleri çizimden önce (tuval boyutu bunlara bağlı)
    B.stages.forEach((l, k) => { const c = el.querySelector(`[data-col="s${k}"]`); c.querySelector('.nstack').style.setProperty('--w', RES_W['s' + k] + '%'); c.querySelector('[data-v="sz"]').textContent = `${l.h} × ${l.w}`; });
    el.querySelector('[data-col="in"] [data-v="sz"]').textContent = '120 × 240';
    if (el.dataset.drawn !== '1') {
      el.dataset.drawn = '1';
      el.querySelectorAll('canvas[data-in]').forEach(cv => drawRect(cv, St, St.best, cv.dataset.in, false));
      const gv = el.querySelector('canvas[data-v="gap"]'); if (gv) drawMap(gv, B.avg, B.n_avg, 1);
      B.stages.forEach((l, k) => el.querySelectorAll(`canvas[data-m="s${k}"]`).forEach(cv => drawMap(cv, l.maps[+cv.dataset.k], l.h, l.w)));
    }
    const vis = (k, on) => { const c = el.querySelector(`[data-col="${k}"]`); if (c) c.classList.toggle('on', on); };
    vis('in', true); vis('stem', f >= 1); B.stages.forEach((_, k) => vis('s' + k, f >= 4 + k)); vis('gap', f >= 4 + B.stages.length); vis('out', f >= 5 + B.stages.length);
    // kök sütunu: conv1 (işaretli) → bn1 + ReLU → maxpool
    const stem = el.querySelector('[data-col="stem"]'), stage = f <= 1 ? 'conv' : f === 2 ? 'relu' : 'pool';
    if (f >= 1) {
      const L = stage === 'pool' ? B.pool : stage === 'conv' ? B.conv1 : B.bn1;
      stem.querySelector('.nstack').style.setProperty('--w', (stage === 'pool' ? RES_W.pool : RES_W.c1) + '%');
      stem.querySelector('[data-v="sz"]').textContent = `${L.h} × ${L.w}`;
      if (stem.dataset.stage !== stage) {
        stem.dataset.stage = stage;
        stem.querySelectorAll('canvas[data-m]').forEach(cv => { const k = +cv.dataset.k;
          if (stage === 'pool') drawMap(cv, L.maps[k], L.h, L.w); else drawSigned(cv, L.maps[k], L.h, L.w, stage === 'relu'); });
      }
    }
    el.querySelector('[data-v="score"]').innerHTML = f >= 5 + B.stages.length ? scoreHTML(B.score, null, R.rnd || !R.src.pre ? 'eğitilmemiş' : 'yalnız ImageNet') : '';
    el.querySelectorAll('.ochip').forEach(c => { const i = +c.dataset.i; c.classList.toggle('done', i < f); c.classList.toggle('cur', i === f); });
    el.querySelector('[data-v="op"]').textContent = s.op === 'res' ? `Aşama ${s.k + 1}` : ROPS[s.op];
    el.querySelector('[data-v="shape"]').textContent = s.shape.join(' × ');
    el.querySelector('[data-v="note"]').textContent = resNote(s, R);
    el.querySelector('[data-v="leg"]').innerHTML = s.op === 'conv7' ? `<span class="mono">−</span><i style="background:linear-gradient(90deg,${divStops()})"></i><span class="mono">+</span>`
      : s.op === 'in' || s.op === 'fcsig' ? '' : `<span class="mono">0</span><i style="background:linear-gradient(90deg,${actStops()})"></i><span class="mono">en büyük</span>`;
  },
  resize(el) { if (!RES || !STORM) return; delete el.dataset.drawn; el.querySelectorAll('[data-stage]').forEach(x => { delete x.dataset.stage; }); this.show(el, DECK.i === SLIDES.indexOf(this) ? DECK.f : 0); },
  tick(el, t) { const R = RES; if (!R) return; const st = resSteps(R), cv = el.querySelector('[data-v="opcv"]'); if (cv) drawOp(cv, st[Math.min(DECK.f, st.length - 1)], t, {before: R.rnd || R.before}); },
};

// --- 7b · eğitim: ResNet (ön-eğitimli) tur tur; CNN karşılaştırma için soluk
const SLIDE_RES_EGITIM = {
  id: 'res_egitim', name: 'ResNet eğitimi', delay: 900,
  get frames() { const e = A && A.ana && A.ana.res_egitim; return e ? e.epoch.length : 1; },
  build(el, d) {
    const e = d.res_egitim;
    if (!e) { el.innerHTML = '<div class="snote">Eğitim geçmişi yok</div>'; return; }
    el.innerHTML = `<div class="s-egit">
      <div class="eside">
        <div class="etur"><b data-v="tur"></b><span>/ ${e.max_epochs} tur</span></div>
        <div class="estop mono" data-v="stop"></div>
        <div class="eleg col"><span><i class="ac"></i>${esc(e.label || 'ResNet-18')}</span>${e.base ? `<span><i class="bs"></i>${esc(e.base_label)} (baseline)</span>` : ''}${e.cnn ? `<span><i class="cn"></i>${esc(e.cnn_label || 'TornadoCNN')}</span>` : ''}</div>
      </div>
      <div class="eplot"><div class="dk">kayıp · ResNet</div><div data-v="loss"></div><div class="eleg"><span><i class="tr"></i>eğitim</span><span><i class="va"></i>doğrulama</span></div></div>
      <div class="eplot"><div class="dk">doğrulama PR-AUC</div><div data-v="pr"></div></div>
    </div>`;
  },
  show(el, f, d) {
    const e = d.res_egitim; if (!e) return;
    el.querySelector('[data-v="tur"]').textContent = e.epoch[f];
    el.querySelector('[data-v="loss"]').innerHTML = lineChart(e.epoch, [{v: e.train_loss, c: '#8791A1', dash: true}, {v: e.val_loss, c: css('--ink')}], f, {best: e.best, stop: true});
    const cn = e.cnn ? e.epoch.map((_, i) => e.cnn.val_pr[i] ?? null) : null, bs = e.base ? e.epoch.map((_, i) => e.base.val_pr[i] ?? null) : null;
    el.querySelector('[data-v="pr"]').innerHTML = lineChart(e.epoch, [...(cn ? [{v: cn, c: '#B9C1CC', w: 2.5}] : []), ...(bs ? [{v: bs, c: '#0072B2', w: 2.5, dash: true}] : []), {v: e.val_pr, c: css('--accent'), mark: true}], f, {best: e.best, stop: true});
    const last = f === e.epoch.length - 1, since = e.epoch[f] - e.best;
    el.querySelector('[data-v="stop"]').textContent = e.epoch[f] >= e.best ? (last ? `en iyi: ${e.best}. tur · ${e.patience} tur iyileşme yok → dur` : `en iyi: ${e.best}. tur${since > 0 ? ` · ${since}/${e.patience}` : ''}`) : '';
  },
};

// --- 7c · yalnız ImageNet / radarla eğitilmiş: aşama 3'ün haritaları, skor ve ilk katmanın filtreleri
function drawFilters(cv, b64s) {
  const [W, H] = fitCanvas(cv), g = cv.getContext('2d'), u = b64(b64s), n = u.length / (7 * 7 * 3), cols = 16, rows = Math.ceil(n / cols);
  const raw = document.createElement('canvas'); raw.width = cols * 8 - 1; raw.height = rows * 8 - 1;
  const rx = raw.getContext('2d'), img = rx.createImageData(raw.width, raw.height), D_ = img.data; D_.fill(255);
  for (let k = 0; k < n; k++) { const ox = (k % cols) * 8, oy = Math.floor(k / cols) * 8;
    for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) { const s = ((k * 7 + i) * 7 + j) * 3, p = ((oy + i) * raw.width + ox + j) * 4; D_[p] = u[s]; D_[p + 1] = u[s + 1]; D_[p + 2] = u[s + 2]; D_[p + 3] = 255; } }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.clearRect(0, 0, W, H);
  const sc = Math.min(W / raw.width, H / raw.height); g.drawImage(raw, (W - raw.width * sc) / 2, (H - raw.height * sc) / 2, raw.width * sc, raw.height * sc);
}
const SLIDE_RES_ONCE = {
  id: 'res_once', name: 'ImageNet / radar', frames: 1,
  build(el, d) {
    const R = resOf(d), St = stormOf(d);
    if (!R || !St) { el.innerHTML = '<div class="snote">ResNet modeli ya da radar verisi yok</div>'; return; }
    const li = Math.min(2, R.after.stages.length - 1), l = R.after.stages[li], nm = 4;
    const row = (key, tag, thr, fk) => `<div class="oprow ${key}">
        <div class="optag">${tag}</div>
        <div class="opin"><canvas data-vel></canvas></div>
        <div class="opmaps">${Array.from({length: nm}, (_, k) => `<div class="nmap"><canvas data-s="${key}" data-k="${k}"></canvas></div>`).join('')}</div>
        ${scoreHTML(R[key].score, thr, '')}
        <div class="opfil"><canvas data-f="${fk}"></canvas></div>
      </div>`;
    el.innerHTML = `<div class="s-once res"><div class="nlabel">${esc(R.label || '')}</div>
      <div class="ophead"><span></span><span>VEL</span><span>aşama ${li + 1} · ${l.c} haritadan en güçlü ${nm} · ${l.h} × ${l.w}</span><span>skor</span><span>ilk katman · ${R.filters ? 64 : ''} filtre 7×7</span></div>
      ${row('before', R.src.pre ? 'yalnız<br>ImageNet' : 'eğitilmemiş', null, 'start')}${row('after', R.src.pre ? 'ImageNet<br>+ radar' : 'radarla<br>eğitilmiş', R.thr, 'tuned')}
      <div class="opleg"><span class="mono">0</span><i style="background:linear-gradient(90deg,${actStops()})"></i><span class="mono">en büyük</span></div>
    </div>`;
  },
  show(el) { this.resize(el); },
  resize(el) {
    const R = RES, St = STORM; if (!R || !St) return;
    const li = Math.min(2, R.after.stages.length - 1), ring = St.rot[St.best];
    el.querySelectorAll('canvas[data-vel]').forEach(cv => drawRect(cv, St, St.best, 'VEL', true));
    el.querySelectorAll('canvas[data-s]').forEach(cv => { const l = R[cv.dataset.s].stages[li]; drawMap(cv, l.maps[+cv.dataset.k], l.h, l.w, ring); });
    el.querySelectorAll('canvas[data-f]').forEach(cv => { const f = R.filters && R.filters[cv.dataset.f]; if (f) drawFilters(cv, f); });
  },
};

// --- 7d · varyant: sıfırdan eğitim (başlangıç filtreleri + ön-eğitimli seed bandına göre fark)
function deltaRows(b, rows, f, title) {
  const W = 560, rh = 50, T = 30, H = T + rows.length * rh + 48, L = 170, R = 70;
  const ds = rows.map(r => r.d).concat([b.min - b.mean, b.max - b.mean]), m = Math.max(.03, ...ds.map(Math.abs)) * 1.15, X = v => L + (W - L - R) * (v + m) / (2 * m);
  let h = `<svg viewBox="0 0 ${W} ${H}" class="dplot"><rect x="${X(b.min - b.mean)}" y="${T - 8}" width="${X(b.max - b.mean) - X(b.min - b.mean)}" height="${rows.length * rh + 8}" class="band"/>`;
  h += `<line x1="${X(0)}" x2="${X(0)}" y1="${T - 8}" y2="${T + rows.length * rh}" class="zero"/><text x="${X(0)}" y="${T - 14}" text-anchor="middle" class="ax">${esc(title)} · ${b.n} seed</text>`;
  const stp = m > .03 ? .02 : .01;
  for (let r = -Math.floor(m / stp) * stp; r <= m + 1e-9; r += stp) { const v = Math.round(r * 100) / 100; if (v) h += `<text x="${X(v)}" y="${H - 26}" text-anchor="middle" class="ax">${v > 0 ? '+' : '−'}${trNum(Math.abs(v), 2)}</text>`; }
  rows.forEach((r, i) => { if (i > f) return; const y = T + i * rh + rh / 2, out = r.v > b.max ? 'up' : r.v < b.min ? 'down' : 'in';
    h += `<g class="dr ${out} cur"><text x="${L - 14}" y="${y + 5}" text-anchor="end" class="nm">${esc(r.name)}</text><line x1="${X(0)}" x2="${X(r.d)}" y1="${y}" y2="${y}"/><circle cx="${X(r.d)}" cy="${y}" r="9"/>
      <text x="${r.d >= 0 ? X(r.d) + 16 : X(0) + 10}" y="${y + 5}" text-anchor="start" class="val">${r.d >= 0 ? '+' : '−'}${trNum(Math.abs(r.d), 3)}</text></g>`; });
  return h + `<text x="${(X(b.min - b.mean) + X(b.max - b.mean)) / 2}" y="${H - 8}" text-anchor="middle" class="ax">seed aralığı</text></svg>`;
}
const SLIDE_RES_VAR = {
  id: 'res_var', name: 'Ön-eğitim işe yarıyor mu?', frames: 1,
  build(el, d) {
    const K = d.res_karsi || {}, G = K.groups || {}, o = K.order || [], R = resOf(d);
    const rs = o.filter(k => G[k] && G[k].arch === 'resnet18'), P = G[rs[0]], S = G[rs[1]];
    if (!P || !S || !R || !R.filters) { el.innerHTML = '<div class="snote">İkinci ResNet sonucu yok</div>'; return; }
    // kart: aynı ağ, iki başlangıç. pre: ImageNet mi rastgele mi; filtreler: başlangıç → radarla eğitimden sonra
    const card = (g, pre, fStart, fEnd, best) => `<div class="rcard${best ? ' best' : ''}">
        <div class="rrole">${pre ? 'varyant · ön-eğitim' : 'ResNet baseline'}${best ? ' · seçilen' : ''}</div>
        <div class="rtitle">${pre ? 'ImageNet ağırlıklarıyla başla' : 'Rastgele ağırlıklarla başla'}</div>
        <div class="rsub mono">${esc(g.label)}</div>
        <div class="rfil">
          <div><div class="dk">ilk katman · başlangıç</div><div class="opfil big"><canvas data-f="${fStart}"></canvas></div></div>
          <div class="rarr">→</div>
          <div><div class="dk">radarla eğitimden sonra</div><div class="opfil big"><canvas data-f="${fEnd}"></canvas></div></div>
        </div>
        <div class="rres"><b>${trNum(g.mean, 3)}</b><span>doğrulama PR-AUC${g.vals.length > 1 ? ` · ${g.vals.length} seed ort.` : ''}</span></div>
      </div>`;
    const preG = P.pre ? P : S, rndG = P.pre ? S : P, preIsBest = P.pre;
    const fPre = preIsBest ? ['imagenet', 'tuned'] : ['imagenet', 'other_tuned'], fRnd = preIsBest ? ['random', 'other_tuned'] : ['random', 'tuned'];
    const gain = preG.mean - rndG.mean;
    const band = preG.vals.length > 1 ? Math.max(...preG.vals) - Math.min(...preG.vals) : null;
    el.innerHTML = `<div class="s-rvar"><div class="nlabel">ResNet-18 · aynı ağ, iki başlangıç · doğrulamada iyi olanı seçtik</div>
      ${card(rndG, false, fRnd[0], fRnd[1], gain < 0)}
      <div class="rmid"><div class="rgain${gain >= 0 ? '' : ' neg'}">${gain >= 0 ? '+' : '−'}${trNum(Math.abs(gain), 3)}</div>
        <div class="rgl">ön-eğitimin katkısı →</div>${band != null ? `<div class="rband mono">seed'ler arası oynama: ${trNum(band, 3)}</div>` : ''}</div>
      ${card(preG, true, fPre[0], fPre[1], gain >= 0)}
    </div>`;
  },
  show(el) { this.resize(el); },
  resize(el) { const R = RES; if (!R || !R.filters) return; el.querySelectorAll('canvas[data-f]').forEach(cv => { const f = R.filters[cv.dataset.f]; if (f) drawFilters(cv, f); }); },
};

// --- 7e · TornadoCNN mi ResNet mi? parametre, tur süresi, seed'ler, son katmanın ızgarası
function drawLastMap(cv, St, m, h, w) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d'); g.clearRect(0, 0, W, H);
  const ch = St.ch.DBZ, q = ch.frames[St.best], raw = document.createElement('canvas'); raw.width = St.NR; raw.height = St.NA;
  const rx = raw.getContext('2d'), img = rx.createImageData(St.NR, St.NA);
  for (let k = 0; k < q.length; k++) { const o = q[k] * 3, gr = (ch.rgb[o] + ch.rgb[o + 1] + ch.rgb[o + 2]) / 3; img.data.set([gr, gr, gr, 255], k * 4); }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(raw, 0, 0, W, H);
  const cw = W / w, chh = H / h;
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) { const v = m[i * w + j], o = v * 3; g.fillStyle = `rgba(${ACT_LUT[o]},${ACT_LUT[o + 1]},${ACT_LUT[o + 2]},${.15 + .65 * v / 255})`; g.fillRect(j * cw, i * chh, cw, chh); }
  g.strokeStyle = 'rgba(255,255,255,.7)'; g.lineWidth = dpr;
  for (let i = 1; i < h; i++) { g.beginPath(); g.moveTo(0, i * chh); g.lineTo(W, i * chh); g.stroke(); }
  for (let j = 1; j < w; j++) { g.beginPath(); g.moveTo(j * cw, 0); g.lineTo(j * cw, H); g.stroke(); }
  const rc = St.rot[St.best]; if (rc) { g.strokeStyle = css('--ink'); g.lineWidth = 2.5 * dpr; g.beginPath(); g.arc((rc[1] + .5) / St.NR * W, (rc[0] + .5) / St.NA * H, 12 * dpr, 0, 2 * Math.PI); g.stroke(); }
}
const SLIDE_RES_KARSI = {
  id: 'res_karsi', name: 'CNN mi ResNet mi?', frames: 1,
  build(el, d) {
    const KK = d.res_karsi || {}, K = KK.groups || {}, keys = (KK.order || []).filter(k => K[k]);
    if (!keys.length) { el.innerHTML = '<div class="snote">Sonuç yok</div>'; return; }
    const NAME = Object.fromEntries(keys.map(k => [k, K[k].arch === 'resnet18' && !K[k].pre ? `${K[k].label} (baseline)` : K[k].label])), PAL = [css('--ink'), css('--accent'), '#0072B2'];
    const COL = Object.fromEntries(keys.map((k, i) => [k, PAL[i] || '#8791A1']));
    const params = k => K[k].params || (K[k].arch === 'tornado_cnn' && d.cnn ? d.cnn.params : null);
    const pMax = Math.max(...keys.map(k => params(k) || 0)), sMax = Math.max(...keys.map(k => K[k].sec || 0));
    const all = keys.flatMap(k => K[k].vals), lo = Math.floor((Math.min(...all) - .02) * 50) / 50, hi = Math.ceil((Math.max(...all) + .02) * 50) / 50, X = v => 100 * (v - lo) / (hi - lo);
    const rowsP = keys.map(k => { const p = params(k), side = p ? 100 * Math.sqrt(p / pMax) : 0;
      return `<div class="kp"><div class="ksq"><i style="width:${side}%;background:${COL[k]}"></i></div><b>${p ? (p >= 1e6 ? trNum(p / 1e6, 1) + ' M' : trNum(Math.round(p / 1000)) + ' bin') : '—'}</b><span>${NAME[k]}</span></div>`; }).join('');
    const rowsT = keys.map(k => `<div class="kt"><span>${NAME[k]}</span><div class="ktr"><i style="width:${100 * (K[k].sec || 0) / sMax}%;background:${COL[k]}"></i></div><b class="mono">${K[k].sec ? trNum(K[k].sec) + ' sn' : '—'}</b></div>`).join('');
    const ticks = []; for (let v = lo; v <= hi + 1e-9; v += .02) ticks.push(+v.toFixed(2));
    const rowsV = keys.map(k => `<div class="kv"><span>${NAME[k]}</span><div class="kvt">${ticks.map(t => `<s style="left:${X(t)}%"></s>`).join('')}
        ${K[k].vals.map(v => `<i style="left:${X(v)}%;background:${COL[k]}"></i>`).join('')}<u style="left:${X(K[k].mean)}%;border-color:${COL[k]}"></u></div><b class="mono">${trNum(K[k].mean, 3)}</b></div>`).join('');
    el.innerHTML = `<div class="s-karsi">
      <div class="kleft2">
        <div class="kblk"><div class="dk">parametre (alan)</div><div class="kps">${rowsP}</div></div>
        <div class="kblk"><div class="dk">tur süresi</div>${rowsT}</div>
        <div class="kblk"><div class="dk">doğrulama PR-AUC · her nokta bir seed</div>${rowsV}
          <div class="kv ax"><span></span><div class="kvt">${ticks.map(t => `<em style="left:${X(t)}%">${trNum(t, 2)}</em>`).join('')}</div><b></b></div></div>
      </div>
      <div class="kright2">
        <div class="dk">son katmanın en güçlü haritası · aynı fırtına</div>
        <div class="klm"><span>${esc(NAME[keys[0]] || 'TornadoCNN')}</span><div class="klcv"><canvas data-lm="cnn"></canvas></div><em class="mono" data-v="cnnsz"></em></div>
        <div class="klm"><span>${esc(NAME[keys[1]] || 'ResNet-18')}</span><div class="klcv"><canvas data-lm="res"></canvas></div><em class="mono" data-v="ressz"></em></div>
        ${keys[2] ? `<div class="klm"><span>${esc(NAME[keys[2]])}</span><div class="klcv"><canvas data-lm="oth"></canvas></div><em class="mono" data-v="othsz"></em></div>` : ''}
      </div>
    </div>`;
  },
  show(el) { this.resize(el); },
  resize(el) {
    const St = STORM; if (!St) return;
    const C = ACTS, R = RES;
    if (C) { const l = C.after.layers[C.after.layers.length - 1], cv = el.querySelector('canvas[data-lm="cnn"]'); if (cv) { drawLastMap(cv, St, l.maps[0], l.h, l.w); el.querySelector('[data-v="cnnsz"]').textContent = `${l.h} × ${l.w}`; } }
    if (R) { const l = R.after.stages[R.after.stages.length - 1], cv = el.querySelector('canvas[data-lm="res"]'); if (cv) { drawLastMap(cv, St, l.maps[0], l.h, l.w); el.querySelector('[data-v="ressz"]').textContent = `${l.h} × ${l.w}`; } }
    if (R && R.other) { const l = R.other, cv = el.querySelector('canvas[data-lm="oth"]'); if (cv) { drawLastMap(cv, St, l.maps[0], l.h, l.w); el.querySelector('[data-v="othsz"]').textContent = `${l.h} × ${l.w}`; } }
  },
};

// ---------------------------------------------------------------- bölüm 8: test sonuçları (ana.py _test; temel 3 model, baseline yalnız referans)
const TPAL = () => [css('--ink'), css('--accent'), '#0072B2'];
const pct = v => '%' + trNum(100 * v, 0);
function curveSVG(T, kind) {
  const W = 420, H = 380, L = 48, R = 14, Tp = 14, B = 40, X = v => L + (W - L - R) * v, Y = v => Tp + (H - Tp - B) * (1 - v), pal = TPAL();
  let h = `<svg viewBox="0 0 ${W} ${H}" class="cplot">`;
  for (let k = 0; k <= 5; k++) { const v = k / 5; h += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}" class="gl"/><text x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${trNum(v, 1)}</text><text x="${X(v)}" y="${H - 22}" text-anchor="middle">${trNum(v, 1)}</text>`; }
  h += kind === 'roc' ? `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" class="rnd"/>` : `<line x1="${X(0)}" y1="${Y(T.rate)}" x2="${X(1)}" y2="${Y(T.rate)}" class="rnd"/><text x="${X(1)}" y="${Y(T.rate) - 6}" text-anchor="end" class="rl">rastgele · ${pct(T.rate)}</text>`;
  h += `<text x="${(X(0) + X(1)) / 2}" y="${H - 4}" text-anchor="middle" class="axl">${kind === 'roc' ? 'yanlış alarm oranı' : 'yakalama (recall)'}</text>`;
  h += `<text transform="translate(12 ${(Y(0) + Y(1)) / 2}) rotate(-90)" text-anchor="middle" class="axl">${kind === 'roc' ? 'yakalama' : 'isabet (precision)'}</text>`;
  T.models.forEach((m, i) => { const c = m[kind], pts = c.x.map((x, j) => `${X(x).toFixed(1)},${Y(c.y[j]).toFixed(1)}`).join(' ');
    h += `<polyline points="${pts}" pathLength="1" fill="none" stroke="${pal[i]}" stroke-width="${i ? 2.2 : 3}" class="cl" style="--k:${i}"/>`; });
  return h + '</svg>';
}
const tleg = T => T.models.map((m, i) => `<span><i style="background:${TPAL()[i]}"></i>${esc(roleName(m.key, m.label))}</span>`).join('');
const SLIDE_TEST_CURVES = {
  id: 'test_curves', name: 'Test · eğriler', frames: 2, delay: 2200,
  build(el, d) {
    const T = d.test; if (!T) { el.innerHTML = '<div class="snote">Test skoru yok</div>'; return; }
    el.innerHTML = `<div class="s-tc">
      <div class="tchead"><div class="skick">Test</div><div class="tcn mono">${trNum(T.n)} kare · ${trNum(T.storms)} fırtına · ${trNum(T.n_tor)} hortum</div><div class="tleg">${tleg(T)}</div></div>
      <div class="tcp" data-at="0"><div class="dk">ROC · ROC-AUC</div>${curveSVG(T, 'roc')}<div class="tvals">${T.models.map((m, i) => `<b style="color:${TPAL()[i]}">${trNum(m['ROC-AUC'], 3)}</b>`).join('')}</div></div>
      <div class="tcp" data-at="1"><div class="dk">Precision–Recall · PR-AUC</div>${curveSVG(T, 'pr')}<div class="tvals">${T.models.map((m, i) => `<b style="color:${TPAL()[i]}">${trNum(m['PR-AUC'], 3)}</b>`).join('')}</div></div>
    </div>`;
  },
  show() {},
};

// 8b · güven aralıkları (fırtına bazlı bootstrap); her model kendi ailesinin baseline'ına göre
const famBase = arch => arch === 'resnet18' ? 'ResNet baseline' : 'CNN baseline';
const SLIDE_TEST_CI = {
  id: 'test_ci', name: 'Test · güven aralıkları', frames: 1,
  build(el, d) {
    const T = d.test; if (!T) { el.innerHTML = '<div class="snote">Test skoru yok</div>'; return; }
    const pal = TPAL(), BS = T.bases || {}, all = T.models.flatMap(m => m.pr_ci).concat(...Object.values(BS).map(b => b.vals));
    const lo = Math.floor((Math.min(...all) - .03) * 20) / 20, hi = Math.ceil((Math.max(...all) + .03) * 20) / 20, X = v => 100 * (v - lo) / (hi - lo);
    const ticks = []; for (let v = lo; v <= hi + 1e-9; v += .05) ticks.push(+v.toFixed(2));
    const VERD = {fark: ['fark', 'up'], 'eğilim': ['eğilim', 'mid'], belirsiz: ['kesin değil', 'mid'], 'gürültü': ['fark yok', 'no']};
    const rows = T.models.map((m, i) => {
      const B = BS[m.arch], ref = !m.is_base && B ? (B.vals.length > 1
        ? `<div class="tband" style="left:${X(Math.min(...B.vals))}%;width:${X(Math.max(...B.vals)) - X(Math.min(...B.vals))}%"></div>` : '') + `<div class="tbm" style="left:${X(B.mean)}%"></div>` : '';
      const v = VERD[m.verdict] || ['', 'no'], neg = (m.d ?? 0) < 0;
      const tag = m.is_base ? `<em class="tv base">${famBase(m.arch)}</em>`
        : m.d != null ? `<em class="tv ${neg && m.verdict === 'fark' ? 'down' : v[1]}">${m.d >= 0 ? '+' : '−'}${trNum(Math.abs(m.d), 3)} · ${v[0]}<small>${famBase(m.arch)}'a göre</small></em>` : '<em></em>';
      return `<div class="trow" style="--k:${i}"><span class="tn">${esc(roleName(m.key, m.label))}</span>
        <div class="ttrack">${ticks.map(t => `<s style="left:${X(t)}%"></s>`).join('')}${ref}
          <i class="tci" style="left:${X(m.pr_ci[0])}%;width:${X(m.pr_ci[1]) - X(m.pr_ci[0])}%;background:${pal[i]}"></i><u style="left:${X(m['PR-AUC'])}%;background:${pal[i]}"></u></div>
        <b class="mono">${trNum(m['PR-AUC'], 3)}</b>${tag}</div>`; }).join('');
    const refs = Object.entries(BS).sort(([a], [b]) => (a === 'tornado_cnn' ? 0 : 1) - (b === 'tornado_cnn' ? 0 : 1)).map(([a, b]) => `<span><i class="${b.vals.length > 1 ? 'bd' : 'ln'}"></i>${famBase(a)}: ${esc(b.label)}${b.vals.length > 1 ? ` · ${b.vals.length} seed` : ''}</span>`).join('');
    el.innerHTML = `<div class="s-tci">
      <div class="dk">test PR-AUC · %95 güven aralığı (${T.n_boot} fırtına örneklemi) · sağda: kendi baseline'ına göre fark</div>
      <div class="trefs">${refs}</div>
      <div class="trows">${rows}
        <div class="trow ax"><span></span><div class="ttrack">${ticks.map(t => `<em style="left:${X(t)}%">${trNum(t, 2)}</em>`).join('')}</div><b></b><em></em></div></div>
    </div>`;
  },
  show() {},
};

// 8c · karar eşiğinde: üç modelin karışıklık matrisi (eşik doğrulamada seçildi) ve POD / SR / CSI
const SLIDE_TEST_CM = {
  id: 'test_cm', name: 'Test · karar eşiği', frames: 1,
  build(el, d) {
    const T = d.test; if (!T) { el.innerHTML = '<div class="snote">Test skoru yok</div>'; return; }
    const pal = TPAL();
    const cell = (i, k, cls, lab) => `<div class="cmc ${cls}"><b data-m="${i}" data-v="${k}">0</b><span>${lab}</span></div>`;
    const card = (m, i) => `<div class="cmcard" style="--k:${i}">
        <div class="cmh"><i style="background:${pal[i]}"></i><b>${esc(roleName(m.key, m.label))}</b><span class="mono">eşik ${trNum(m.thr, 2)}</span></div>
        <div class="cm"><span></span><span class="ch">tahmin: hortum</span><span class="ch">tahmin: yok</span>
          <span class="rh">gerçek: hortum</span>${cell(i, 'TP', 'tp', 'yakalandı')}${cell(i, 'FN', 'fn', 'kaçırıldı')}
          <span class="rh">gerçek: yok</span>${cell(i, 'FP', 'fp', 'yanlış alarm')}${cell(i, 'TN', 'tn', 'doğru ret')}</div>
        <div class="cmf mono"><span>POD <b>${trNum(m.POD, 2)}</b></span><span>SR <b>${trNum(m.SR, 2)}</b></span><span>CSI <b>${trNum(m.CSI, 2)}</b></span></div>
      </div>`;
    el.innerHTML = `<div class="s-tcm3">${T.models.map(card).join('')}</div>`;
  },
  show(el, f, d, prev) {
    const T = d.test; if (!T || prev != null) return;
    el.querySelectorAll('[data-v]').forEach(n => countUp(n, T.models[+n.dataset.m][n.dataset.v], v => trNum(Math.round(v)), 1000));
  },
};

// 8d · zorluğa göre: hangi fırtınalar karşılaştırılıyor (PR-AUC) ve hortum şiddetine göre yakalama (eşikte)
const SLIDE_TEST_ZOR = {
  id: 'test_zor', name: 'Test · zorluğa göre', frames: 1,
  build(el, d) {
    const T = d.test; if (!T) { el.innerHTML = '<div class="snote">Test skoru yok</div>'; return; }
    const pal = TPAL(), m0 = T.models[0];
    const V = [['NC', 'bütün fırtınalar'], ['RC', 'hortum ↔ sıradan fırtına'], ['WC', 'hortum ↔ uyarı verilen fırtına']];
    const efs = Object.keys(m0.ef || {});
    const yaxis = fmt => `<div class="zy">${[1, .75, .5, .25, 0].map(v => `<span style="bottom:${100 * v}%">${fmt(v)}</span>`).join('')}</div>`;
    const grid = () => `<div class="zgrid">${[1, .75, .5, .25].map(v => `<s style="bottom:${100 * v}%"></s>`).join('')}</div>`;
    const panel = (title, q, groups, val, fmt, axisFmt, hard) => `<div class="zblk">
        <div class="ztitle">${title}</div><div class="zq">${q}</div>
        <div class="zplot">${yaxis(axisFmt)}<div class="zchart">${grid()}${groups.map(([k, name, sub], gi) => `<div class="zg${hard === k ? ' hard' : ''}" style="--k:${gi}">
            ${hard === k ? '<em class="zhard">en zor</em>' : ''}
            <div class="zbars">${T.models.map((m, i) => { const v = val(m, k); return `<i style="--h:${100 * v}%;background:${pal[i]}"><span>${fmt(v)}</span></i>`; }).join('')}</div>
            <b>${name}</b><span>${sub}</span></div>`).join('')}</div></div></div>`;
    el.innerHTML = `<div class="s-zor">
      <div class="tleg big">${tleg(T)}</div>
      <div class="zrow">
        ${panel('PR-AUC', 'Hangi fırtınalardan ayırt ediyor?', V.map(([k, n]) => [k, n, `hortum oranı ${pct(m0.views[k].rate)}`]), (m, k) => m.views[k].pr, v => trNum(v, 2), v => trNum(v, 2), 'WC')}
        ${panel('Yakalama', 'Hortumun şiddetine göre kaçını yakalıyor?', efs.map(k => [k, k, `${trNum(m0.ef_n[k])} kare`]), (m, k) => m.ef[k], v => pct(v), v => pct(v), null)}
      </div>
    </div>`;
  },
  show() {},
};

// ---------------------------------------------------------------- bölüm 9: açıklanabilirlik ve kapanış
const CAM_LUT = (() => { const c = []; for (let q = 0; q < 256; q++) { const v = CH.CAM.cmap(q / 255); c.push(v); } return c; })();
const CHC = {DBZ: '#0072B2', VEL: '#E8950E', WIDTH: '#7B5BE0'};     // Açıklanabilirlik sekmesindeki kanal renkleri
let XCAM = null;
function xcamOf(d) {
  const x = d && d.xai_storm; if (!x) return (XCAM = null);
  if (XCAM && XCAM.src === x) return XCAM;
  return (XCAM = {src: x, models: x.models.map(m => { const u = b64(m.x), N = m.h * m.w, F = u.length / N;
    const frames = Array.from({length: F}, (_, f) => u.subarray(f * N, (f + 1) * N));
    const peaks = frames.map(fr => { let b = 0; for (let k = 1; k < fr.length; k++) if (fr[k] > fr[b]) b = k; return [Math.floor(b / m.w) * 2 + 1, (b % m.w) * 2 + 1]; });
    return {...m, frames, peaks}; })});
}
// Grad-CAM'i kutupsal görünüme yarı saydam dörtgenlerle bindirir (60 × 120 hücre, her biri 2 × 2 girdi hücresi)
function drawCamPolar(g, St, f, cam, h, w, T, dpr) {
  const G = stormGeom(St, f, 0), NR1 = St.NR + 1;
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const v = cam[i * w + j]; if (v < 25) continue;
    const a = (2 * i) * NR1 + 2 * j, b = a + 2, c = a + 2 * NR1 + 2, d = a + 2 * NR1, col = CAM_LUT[v];
    g.fillStyle = `rgba(${col[0] | 0},${col[1] | 0},${col[2] | 0},${Math.min(.8, v / 255 * .95)})`;
    g.beginPath(); g.moveTo(T.sx(G.X[a]), T.sy(G.Y[a])); g.lineTo(T.sx(G.X[b]), T.sy(G.Y[b])); g.lineTo(T.sx(G.X[c]), T.sy(G.Y[c])); g.lineTo(T.sx(G.X[d]), T.sy(G.Y[d])); g.closePath(); g.fill();
  }
}
const XC = {cache: new Map(), key: ''};
function drawXaiPanel(cv, St, M, f) {
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d');
  const rc = St.rot[f] || St.rot[St.best] || [St.NA / 2, St.NR / 2], [cx, cy] = cellXY(St, St.rot[f] ? f : St.best, rc[0], rc[1]), hw = 13;
  const cam = {cx, cy, hw: hw * Math.max(1, W / H), hh: hw * Math.max(1, H / W)}, T = camXf(cam, W, H);
  const key = `${W}x${H}|${St.key}|${M.key}|${f}`;
  let L = XC.cache.get(key);
  if (!L) {
    if (XC.cache.size > 40) XC.cache.clear();
    L = document.createElement('canvas'); L.width = W; L.height = H; const lg = L.getContext('2d');
    drawCells(lg, St, f, 'VEL', stormGeom(St, f, 0), T, W, H, dpr);
    lg.fillStyle = 'rgba(255,255,255,.35)'; lg.fillRect(0, 0, W, H);
    drawCamPolar(lg, St, f, M.frames[f], M.h, M.w, T, dpr); XC.cache.set(key, L);
  }
  g.clearRect(0, 0, W, H); g.drawImage(L, 0, 0);
  // dönüş çifti (halka) ve Grad-CAM tepesi (artı); aradaki çizgi
  const [px, py] = cellXY(St, f, M.peaks[f][0], M.peaks[f][1]), [rx, ry] = cellXY(St, f, rc[0], rc[1]);
  const R = 2 * T.s;
  g.strokeStyle = css('--ink'); g.lineWidth = 1.5 * dpr; g.setLineDash([4 * dpr, 4 * dpr]); g.beginPath(); g.moveTo(T.sx(rx), T.sy(ry)); g.lineTo(T.sx(px), T.sy(py)); g.stroke(); g.setLineDash([]);
  g.lineWidth = 5 * dpr; g.strokeStyle = 'rgba(255,255,255,.9)'; g.beginPath(); g.arc(T.sx(rx), T.sy(ry), R, 0, 2 * Math.PI); g.stroke();
  g.lineWidth = 2.5 * dpr; g.strokeStyle = css('--ink'); g.beginPath(); g.arc(T.sx(rx), T.sy(ry), R, 0, 2 * Math.PI); g.stroke();
  const s = 10 * dpr; g.lineWidth = 3.5 * dpr; g.strokeStyle = '#fff'; g.beginPath(); g.moveTo(T.sx(px) - s, T.sy(py)); g.lineTo(T.sx(px) + s, T.sy(py)); g.moveTo(T.sx(px), T.sy(py) - s); g.lineTo(T.sx(px), T.sy(py) + s); g.stroke();
  g.lineWidth = 2 * dpr; g.strokeStyle = css('--tor'); g.stroke();
  return Math.hypot(px - rx, py - ry);
}
const SLIDE_XAI_CAM = {
  id: 'xai_cam', name: 'Model nereye bakıyor?', delay: 1100,
  get frames() { return STORM ? STORM.F : 1; },
  build(el, d) {
    const X = xcamOf(d), St = stormOf(d); XC.cache.clear();
    if (!X || !St) { el.innerHTML = '<div class="snote">Grad-CAM hesaplanamadı (torch ya da radar verisi yok)</div>'; return; }
    const pal = TPAL();
    el.innerHTML = `<div class="s-xc">
      <div class="xchead"><div class="skick">Grad-CAM</div><div class="itime mono" data-v="time"></div>
        <div class="itl">${St.frames.map((x, k) => `<i data-k="${k}" class="${x.cat.toLowerCase()}"></i>`).join('')}</div>
        <div class="xleg"><span><i class="ring"></i>dönüş çifti</span><span><i class="plus">+</i>Grad-CAM tepesi</span>
          <span class="mono">0</span><i class="bar" style="background:linear-gradient(90deg,${actStops()})"></i><span class="mono">en önemli</span></div></div>
      <div class="xcp">${X.models.map((m, i) => `<div class="xcm"><div class="xcn"><i style="background:${pal[i]}"></i><b>${esc(roleName(m.key, m.label))}</b></div>
        <div class="xccv"><canvas data-i="${i}"></canvas></div><div class="xcd mono"><span>tepe → çift</span><b data-d="${i}"></b></div></div>`).join('')}</div>
    </div>`;
  },
  show(el, f) {
    const X = XCAM, St = STORM; if (!X || !St || !el.querySelector('.xcp')) return;
    el.querySelector('[data-v="time"]').textContent = St.frames[f].time;
    el.querySelectorAll('.itl i').forEach(i => i.classList.toggle('on', +i.dataset.k === f));
    el.querySelectorAll('canvas[data-i]').forEach(cv => { const i = +cv.dataset.i, km = drawXaiPanel(cv, St, X.models[i], f); el.querySelector(`[data-d="${i}"]`).textContent = `${trNum(km, 1)} km`; });
  },
  resize(el) { XC.cache.clear(); this.show(el, DECK.i === SLIDES.indexOf(this) ? DECK.f : 0); },
};

// 9b · kanal payları (kanal SHAP, bütün test kümesi): "hortum" deyip doğru çıkanlar vs "yok" deyip doğru çıkanlar
const SLIDE_XAI_SHAP = {
  id: 'xai_shap', name: 'Hangi kanal?', frames: 1,
  build(el, d) {
    const X = d.xai; if (!X) { el.innerHTML = '<div class="snote">Açıklama özeti yok</div>'; return; }
    const pal = TPAL(), V = ['DBZ', 'VEL', 'WIDTH'];
    const bar = (sh, k) => sh ? `<div class="sbar"><span>${k === 'tp' ? 'hortum dedi · doğru' : 'yok dedi · doğru'}</span><div class="sbt">${V.map(v => `<i style="flex:${sh[v]};background:${CHC[v]}"><b>${v === 'VEL' || sh[v] > .12 ? pct(sh[v]) : ''}</b></i>`).join('')}</div></div>` : '';
    el.innerHTML = `<div class="s-xs">
      <div class="xsleg">${V.map(v => `<span><i style="background:${CHC[v]}"></i>${v}</span>`).join('')}<em>kararın kanallara dağılımı · kanal SHAP · ${trNum(X.models[0].n_shap || 0)} test karesi</em></div>
      ${X.models.map((m, i) => `<div class="xsm" style="--k:${i}"><div class="xcn"><i style="background:${pal[i]}"></i><b>${esc(roleName(m.key, m.label))}</b></div>
        ${m.shap ? bar(m.shap.tp, 'tp') + bar(m.shap.tn, 'tn') : '<div class="snote">yok</div>'}
        ${m.shap && m.shap.tp && m.shap.tn ? `<div class="xsd mono">VEL payı: ${pct(m.shap.tn.VEL)} → <b>${pct(m.shap.tp.VEL)}</b></div>` : ''}</div>`).join('')}
    </div>`;
  },
  show() {},
};

// 9c · açıklamaya güvenebilir miyiz? silme testi (rastgeleye göre kaç kat hızlı) ve tepenin dönmeye uzaklığı
const SLIDE_XAI_TRUST = {
  id: 'xai_trust', name: 'Açıklamaya güven', frames: 1,
  build(el, d) {
    const X = d.xai; if (!X) { el.innerHTML = '<div class="snote">Açıklama özeti yok</div>'; return; }
    const pal = TPAL(), ms = X.models;
    const ratio = m => m.deletion ? m.deletion.rnd / Math.max(m.deletion.cam, 1e-3) : null;
    const rMax = Math.max(2.5, ...ms.map(m => ratio(m) || 0)) * 1.1, LX = v => 100 * Math.log(v / (1 / rMax)) / Math.log(rMax * rMax);
    const dMax = Math.max(...ms.map(m => (m.dist && m.dist.median) || 0)) * 1.25 || 1;
    const del = ms.map((m, i) => { const r = ratio(m); if (r == null) return ''; const good = r > 1;
      return `<div class="xtr" style="--k:${i}"><span>${esc(roleName(m.key, m.label))}</span><div class="xtt"><s style="left:${LX(1)}%"></s>
        <i class="${good ? 'g' : 'b'}" style="left:${Math.min(LX(1), LX(r))}%;width:${Math.abs(LX(r) - LX(1))}%;background:${pal[i]}"></i></div>
        <b class="mono ${good ? 'g' : 'b'}">${good ? trNum(r, 1) + '× hızlı' : trNum(1 / r, 1) + '× yavaş'}</b></div>`; }).join('');
    const dist = ms.map((m, i) => { const v = m.dist && m.dist.median; if (v == null) return '';
      return `<div class="xtr" style="--k:${i}"><span>${esc(roleName(m.key, m.label))}</span><div class="xtt"><i style="left:0;width:${100 * v / dMax}%;background:${pal[i]}"></i></div><b class="mono">${trNum(v, 1)} km</b></div>`; }).join('');
    el.innerHTML = `<div class="s-xt">
      <div class="xtb"><div class="xtq">Önemli dediği pikseller silinince skor ne kadar hızlı düşüyor?</div>
        <div class="dk">silme testi · Grad-CAM sırası, rastgele sıraya göre · ${ms[0].deletion ? ms[0].deletion.n : ''} "hortum" kararı</div>
        ${del}<div class="xtax"><span>rastgeleden yavaş</span><span>rastgele</span><span>rastgeleden hızlı</span></div></div>
      <div class="xtb"><div class="xtq">Grad-CAM tepesi dönüş çiftine ne kadar yakın?</div>
        <div class="dk">doğru yakalanan hortumlarda ortanca uzaklık</div>${dist}</div>
    </div>`;
  },
  show() {},
};

// 9d · kapanış: özet sayılar, sınırlamalar, uygulamanın sekmelerine geçiş
const SLIDE_KAPANIS = {
  id: 'kapanis', name: 'Kapanış', frames: 1,
  build(el, d) {
    const T = d.test, K = d.kapak || {}, m = T && T.models[0], St = stormOf(d);
    const nums = [
      [K.n != null ? trNum(K.n) : '—', 'örnek · ' + ((K.years || []).join(' · '))],
      [m ? trNum(m['PR-AUC'], 2) : '—', `test PR-AUC · ${m ? esc(m.label) : ''}`],
      [m ? pct(m.POD) : '—', 'hortum yakalandı (eşikte)'],
      [m && m.ef ? pct(m.ef['EF3+'] ?? m.ef.EF2) : '—', 'güçlü hortumlarda (EF3+)'],
    ];
    const lims = ['3 yıl veri', 'tek tarama açısı ve tek kare', m && m.ef ? `zayıf hortumlarda (EF0) yakalama ${pct(m.ef.EF0)}` : 'zayıf hortumlar', 'seçilen varyant tek seed'];
    const tabs = [['gezgin', 'radar', 'Fırtına Gezgini'], ['kars', 'bars', 'Karşılaştırma'], ['hata', 'bug', 'Hata Analizi'], ['xai', 'bulb', 'Açıklanabilirlik'], ['harita', 'map', 'Harita']];
    el.innerHTML = `<div class="s-kap">
      <div class="kpn">${nums.map(([v, k], i) => `<div style="--k:${i}"><b>${v}</b><span>${k}</span></div>`).join('')}</div>
      <div class="klim"><span class="dk">sınırlamalar</span>${lims.map(x => `<em>${x}</em>`).join('')}</div>
      <div class="ktabs">${tabs.map(([id, ic, nm]) => `<button class="ktab" data-tab="${id}">${ico(ic)}<span>${nm}</span></button>`).join('')}</div>
    </div>`;
    el.querySelectorAll('.ktab').forEach(b => b.onclick = e => { e.stopPropagation(); send('tab', {tab: b.dataset.tab}); });
  },
  show() {},
};

// ---------------------------------------------------------------- bölüm 9: "nasıl çalışır" slaytları (ana.py _xai_steps; seçilen CNN, bölüm 3'ün karesi)
function howHead(steps, f) {   // altta adım çipleri (6a'daki şema gibi)
  return `<div class="hsteps">${steps.map(([nm, fm], i) => `<div class="hst${i < f ? ' done' : ''}${i === f ? ' cur' : ''}"><b>${i + 1}</b><span>${nm}</span><em class="mono">${fm}</em></div>`).join('<i>›</i>')}</div>`;
}
function drawRectOverlay(cv, St, f, chName, over, oh, ow, ring) {   // radar kanalı + üstüne yarı saydam ısı haritası
  drawRect(cv, St, f, chName, false);
  const [W, H, dpr] = fitCanvas(cv), g = cv.getContext('2d');
  g.fillStyle = 'rgba(255,255,255,.35)'; g.fillRect(0, 0, W, H);
  const raw = document.createElement('canvas'); raw.width = ow; raw.height = oh;
  const rx = raw.getContext('2d'), img = rx.createImageData(ow, oh);
  for (let k = 0; k < over.length; k++) { const v = over[k], c = CAM_LUT[v]; img.data.set([c[0], c[1], c[2], Math.min(230, v * 1.1)], k * 4); }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = true; g.drawImage(raw, 0, 0, W, H);
  const rc = ring && St.rot[St.best];
  if (rc) { const x = (rc[1] + .5) / St.NR * W, y = (rc[0] + .5) / St.NA * H; g.lineWidth = 2.5 * dpr; g.strokeStyle = css('--ink'); g.beginPath(); g.arc(x, y, 14 * dpr, 0, 2 * Math.PI); g.stroke(); }
}

// --- 9a · Grad-CAM nasıl hesaplanır? (5 kare)
const CAM_STEPS = [['son katman', 'Aₖ'], ['ağırlıklar', 'wₖ = ort(∂skor / ∂Aₖ)'], ['ağırlıklı toplam', 'Σ wₖ · Aₖ'], ['ReLU', 'max(0, ·)'], ['büyüt ve bindir', '120 × 240']];
const SLIDE_XAI_HOW_CAM = {
  id: 'xai_how_cam', name: 'Grad-CAM nasıl?', frames: 5, delay: 1800,
  build(el, d) {
    const X = d.xai_steps, St = stormOf(d); delete el.dataset.drawn;
    if (!X || !St) { el.innerHTML = '<div class="snote">Açıklama adımları hesaplanamadı</div>'; return; }
    const C = X.cam, ws = C.weights, wMax = Math.max(...ws.map(Math.abs));
    const wbars = ws.map((w, i) => `<i class="${w >= 0 ? 'p' : 'n'}${i < C.n ? ' top' : ''}" style="left:${100 * i / ws.length}%;width:${100 / ws.length}%;height:${50 * Math.abs(w) / wMax}%;${w >= 0 ? 'bottom:50%' : 'top:50%'}"></i>`).join('');
    el.innerHTML = `<div class="s-how"><div class="nlabel">${esc((d.cnn && d.cnn.label) || '')} · Grad-CAM</div>
      <div class="hflow">
        <div class="hcol" data-at="0"><div class="nhead"><b>${C.k}</b><span class="mono">${C.h} × ${C.w}</span></div>
          <div class="hmaps">${Array.from({length: C.n}, (_, k) => `<div class="hm"><div class="nmap"><canvas data-m="${k}"></canvas></div><span class="mono">w ${trNum(C.map_w[k], 2)}</span></div>`).join('')}</div></div>
        <div class="narr">→</div>
        <div class="hcol" data-at="1"><div class="nhead"><b>${C.k}</b><span class="mono">ağırlık · büyükten küçüğe</span></div><div class="hw">${wbars}<s></s></div>
          <div class="hwl mono"><span>hortum lehine</span><span>aleyhine</span></div></div>
        <div class="narr">→</div>
        <div class="hcol" data-at="2"><div class="nhead"><b>1</b><span class="mono">${C.h} × ${C.w}</span></div><div class="nmap big"><canvas data-v="raw"></canvas></div>
          <div class="oleg"><span class="mono">−</span><i style="background:linear-gradient(90deg,${divStops()})"></i><span class="mono">+</span></div></div>
        <div class="narr">→</div>
        <div class="hcol" data-at="3"><div class="nhead"><b>1</b><span class="mono">yalnız +</span></div><div class="nmap big"><canvas data-v="relu"></canvas></div></div>
        <div class="narr">→</div>
        <div class="hcol wide" data-at="4"><div class="nhead"><b>Grad-CAM</b><span class="mono">VEL üzerinde</span></div><div class="nmap big"><canvas data-v="up"></canvas></div></div>
      </div>
      <div data-v="steps"></div>
    </div>`;
  },
  show(el, f) {
    const X = A.ana && A.ana.xai_steps, St = STORM; if (!X || !St || !el.querySelector('.hflow')) return;
    const C = X.cam;
    if (el.dataset.drawn !== '1') {
      el.dataset.drawn = '1';
      const maps = b64(C.maps), N = C.h * C.w;
      el.querySelectorAll('canvas[data-m]').forEach(cv => { const k = +cv.dataset.m; drawMap(cv, maps.subarray(k * N, (k + 1) * N), C.h, C.w); });
      drawSigned(el.querySelector('[data-v="raw"]'), b64(C.raw), C.h, C.w, false);
      drawMap(el.querySelector('[data-v="relu"]'), b64(C.relu), C.h, C.w);
      drawRectOverlay(el.querySelector('[data-v="up"]'), St, St.best, 'VEL', b64(C.up), 120, 240, true);
    }
    el.querySelector('[data-v="steps"]').innerHTML = howHead(CAM_STEPS, f);
  },
  resize(el) { delete el.dataset.drawn; this.show(el, DECK.i === SLIDES.indexOf(this) ? DECK.f : 0); },
};

// --- 9c · bir kanalın katkısı nasıl ölçülür? 2³ alt küme (8 kare) + Shapley hesabı (1 kare)
const SLIDE_XAI_HOW_SHAP = {
  id: 'xai_how_shap', name: 'Kanal katkısı nasıl?', delay: 1300,
  get frames() { const x = A && A.ana && A.ana.xai_steps; return x ? x.shap.subsets.length + 1 : 1; },
  build(el, d) {
    const X = d.xai_steps, St = stormOf(d); delete el.dataset.drawn;
    if (!X || !St) { el.innerHTML = '<div class="snote">Açıklama adımları hesaplanamadı</div>'; return; }
    const V = X.variables, S = X.shap;
    const rows = S.subsets.map((r, i) => `<div class="ssr" data-r="${i}"><span class="ssc">${V.map((v, c) => `<i class="${r.on[c] ? 'on' : ''}" style="--c:${CHC[v]}">${v}</i>`).join('')}</span>
        <div class="sst"><i style="width:${100 * r.score}%"></i></div><b class="mono">${trNum(r.score, 2)}</b></div>`).join('');
    el.innerHTML = `<div class="s-hshap"><div class="nlabel">${esc((d.cnn && d.cnn.label) || '')} · kanal katkısı</div>
      <div class="hsl">${V.map(v => `<div class="hch" data-ch="${v}"><span style="color:${CHC[v]}">${v}</span><div class="nmap"><canvas data-in="${v}"></canvas></div></div>`).join('')}</div>
      <div class="hsm"><div class="hsq">skor</div><div class="hss" data-v="score"></div><div class="hsn mono" data-v="note"></div></div>
      <div class="hsr"><div class="dk">${S.subsets.length} birleşim · kapalı kanal = boş</div>${rows}</div>
      <div class="hsw" data-v="wf"></div>
    </div>`;
  },
  show(el, f) {
    const X = A.ana && A.ana.xai_steps, St = STORM; if (!X || !St || !el.querySelector('.hsl')) return;
    const V = X.variables, S = X.shap, n = S.subsets.length;
    if (el.dataset.drawn !== '1') { el.dataset.drawn = '1'; el.querySelectorAll('canvas[data-in]').forEach(cv => drawRect(cv, St, St.best, cv.dataset.in, false)); }
    const wf = f >= n, r = S.subsets[Math.min(f, n - 1)];
    el.querySelector('.s-hshap').classList.toggle('final', wf);
    V.forEach((v, c) => el.querySelector(`[data-ch="${v}"]`).classList.toggle('off', !wf && !r.on[c]));
    el.querySelectorAll('.ssr').forEach(row => { const i = +row.dataset.r; row.classList.toggle('on', i <= f); row.classList.toggle('cur', i === f); });
    el.querySelector('[data-v="score"]').textContent = trNum(wf ? S.score : r.score, 2);
    el.querySelector('[data-v="note"]').textContent = wf ? 'Shapley: her kanalın ortalama katkısı' : (r.on.some(Boolean) ? r.on.map((o, c) => o ? V[c] : null).filter(Boolean).join(' + ') : 'hiç kanal yok');
    if (wf) {   // şelale: hiçbiri → +DBZ → +VEL → +WIDTH = skor
      const steps = [['hiçbiri', S.base, 0]].concat(V.map(v => [v, S.phi[v], 1])); let acc = 0;
      el.querySelector('[data-v="wf"]').innerHTML = `<div class="dk">katkılar toplanınca gerçek skor</div><div class="wfr">${steps.map(([nm, val, isPhi]) => { const lo = isPhi ? Math.min(acc, acc + val) : 0, hi = isPhi ? Math.max(acc, acc + val) : val; acc = isPhi ? acc + val : val;
        return `<div class="wfc"><div class="wft"><i style="bottom:${100 * lo}%;height:${Math.max(.5, 100 * (hi - lo))}%;background:${isPhi ? (CHC[nm] || '#8791A1') : '#B9C1CC'}"></i></div><b class="mono">${isPhi ? (val >= 0 ? '+' : '−') + trNum(Math.abs(val), 2) : trNum(val, 2)}</b><span>${nm}</span></div>`; }).join('')}
        <div class="wfc tot"><div class="wft"><i style="bottom:0;height:${100 * S.score}%;background:${css('--ink')}"></i></div><b class="mono">${trNum(S.score, 2)}</b><span>skor</span></div></div>`;
    } else el.querySelector('[data-v="wf"]').innerHTML = '';
  },
  resize(el) { delete el.dataset.drawn; this.show(el, DECK.i === SLIDES.indexOf(this) ? DECK.f : 0); },
};

// --- 9e · silme testi nasıl yapılır? (6 kare: %0, %10 … %50 silindi)
function drawDeleted(cv, St, stepB64, k) {
  drawRect(cv, St, St.best, 'VEL', false);
  const [W, H] = fitCanvas(cv), g = cv.getContext('2d'), st = b64(stepB64), raw = document.createElement('canvas'); raw.width = St.NR; raw.height = St.NA;
  const rx = raw.getContext('2d'), img = rx.createImageData(St.NR, St.NA);
  for (let p = 0; p < st.length; p++) if (st[p] <= k) { const i = Math.floor(p / St.NR), j = p % St.NR, c = ((i >> 2) + (j >> 2)) & 1 ? 255 : 226; img.data.set([c, c, c + 4, 255], p * 4); }
  rx.putImageData(img, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(raw, 0, 0, W, H);
}
function delChart(D, k) {
  const W = 520, H = 240, L = 44, R = 14, T = 12, B = 34, xs = D.fracs, X = v => L + (W - L - R) * v / xs[xs.length - 1], Y = v => T + (H - T - B) * (1 - v);
  let h = `<svg viewBox="0 0 ${W} ${H}" class="cplot">`;
  for (let v = 0; v <= 1.001; v += .25) h += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="gl"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${trNum(v, 2)}</text>`;
  xs.forEach((x, i) => { if (i % 2 === 0) h += `<text x="${X(x)}" y="${H - 16}" text-anchor="middle">%${trNum(100 * x)}</text>`; });
  h += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle" class="axl">silinen piksel</text>`;
  [['rnd', '#8791A1'], ['cam', css('--tor')]].forEach(([key, c]) => { const pts = D[key].slice(0, k + 1).map((v, i) => `${X(xs[i])},${Y(v)}`).join(' ');
    h += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="3" stroke-linejoin="round"/><circle cx="${X(xs[k])}" cy="${Y(D[key][k])}" r="5" fill="${c}"/>`; });
  return h + '</svg>';
}
const SLIDE_XAI_HOW_DEL = {
  id: 'xai_how_del', name: 'Silme testi nasıl?', frames: 6, delay: 1300,
  build(el, d) {
    const X = d.xai_steps, St = stormOf(d);
    if (!X || !St) { el.innerHTML = '<div class="snote">Açıklama adımları hesaplanamadı</div>'; return; }
    el.innerHTML = `<div class="s-hdel"><div class="nlabel">${esc((d.cnn && d.cnn.label) || '')} · silme testi</div>
      <div class="hdi"><div class="hdt" style="color:var(--tor)">Grad-CAM'in önemli dediği yerden başla</div><div class="nmap big"><canvas data-v="cam"></canvas></div><div class="hdsc"><b class="mono" data-v="scam"></b><span>skor</span></div></div>
      <div class="hdi"><div class="hdt">rastgele yerlerden sil</div><div class="nmap big"><canvas data-v="rnd"></canvas></div><div class="hdsc"><b class="mono" data-v="srnd"></b><span>skor</span></div></div>
      <div class="hdc"><div class="hdq"><b data-v="frac"></b><span>piksel silindi</span></div><div data-v="chart"></div>
        <div class="tleg"><span><i style="background:var(--tor)"></i>Grad-CAM sırası</span><span><i style="background:#8791A1"></i>rastgele sıra</span></div></div>
    </div>`;
  },
  show(el, f) {
    const X = A.ana && A.ana.xai_steps, St = STORM; if (!X || !St || !el.querySelector('.hdi')) return;
    const D = X.deletion, k = Math.min(2 * f, D.fracs.length - 1);
    drawDeleted(el.querySelector('[data-v="cam"]'), St, D.cam_step, k);
    drawDeleted(el.querySelector('[data-v="rnd"]'), St, D.rnd_step, k);
    el.querySelector('[data-v="scam"]').textContent = trNum(D.cam[k], 2);
    el.querySelector('[data-v="srnd"]').textContent = trNum(D.rnd[k], 2);
    el.querySelector('[data-v="frac"]').textContent = `%${trNum(100 * D.fracs[k])}`;
    el.querySelector('[data-v="chart"]').innerHTML = delChart(D, k);
  },
  resize(el) { this.show(el, DECK.i === SLIDES.indexOf(this) ? DECK.f : 0); },
};

const SLIDES = [SLIDE_KAPAK, SLIDE_TORNET, SLIDE_DENGE, SLIDE_POLAR, SLIDE_KANAL, SLIDE_IMZA, SLIDE_OLCUM, SLIDE_KUP, SLIDE_ONISLEME, SLIDE_BOLME,
                SLIDE_CNN, SLIDE_EGITIM, SLIDE_ONCESONRA, SLIDE_HP, SLIDE_VARYANT,
                SLIDE_RES, SLIDE_RES_VAR, SLIDE_RES_EGITIM, SLIDE_RES_ONCE, SLIDE_RES_KARSI,
                SLIDE_TEST_CURVES, SLIDE_TEST_CI, SLIDE_TEST_CM, SLIDE_TEST_ZOR,
                SLIDE_XAI_HOW_CAM, SLIDE_XAI_CAM, SLIDE_XAI_HOW_SHAP, SLIDE_XAI_SHAP, SLIDE_XAI_HOW_DEL, SLIDE_XAI_TRUST, SLIDE_KAPANIS];

function deckEls() { return [...$('slides').children]; }
function deckBuild(force) {
  const d = A.ana || {}, key = JSON.stringify(d);
  if (!force && key === DECK.key && $('slides').children.length === SLIDES.length) return;
  DECK.key = key;
  const box = $('slides');
  if (box.children.length !== SLIDES.length) box.innerHTML = SLIDES.map(s => `<section class="slide" data-id="${s.id}" aria-label="${esc(s.name)}"></section>`).join('');
  SLIDES.forEach((s, i) => s.build(box.children[i], d));
  deckShow(null);
}
function deckShow(prev) {
  const els = deckEls(), s = SLIDES[DECK.i], el = els[DECK.i], d = A.ana || {};
  els.forEach((e, i) => e.classList.toggle('cur', i === DECK.i));
  el.querySelectorAll('[data-at]').forEach(n => n.classList.toggle('on', DECK.f >= +n.dataset.at));
  s.show(el, DECK.f, d, prev);
  const first = DECK.i === 0 && DECK.f === 0, last = DECK.i === SLIDES.length - 1 && DECK.f === s.frames - 1;
  $('hit-l').disabled = first; $('hit-r').disabled = last;
  deckLoop();
}
function deckGo(i, f, prevF) {
  i = clamp(i, 0, SLIDES.length - 1); f = clamp(f, 0, SLIDES[i].frames - 1);
  const prev = i === DECK.i ? (prevF ?? DECK.f) : null;   // slayt değişince sayılar baştan sayar
  if (i === DECK.i && f === DECK.f) return;
  DECK.i = i; DECK.f = f; deckShow(prev);
}
function deckStep(dir) {
  const s = SLIDES[DECK.i];
  if (dir > 0) { if (DECK.f < s.frames - 1) deckGo(DECK.i, DECK.f + 1); else if (DECK.i < SLIDES.length - 1) deckGo(DECK.i + 1, 0); else return false; }
  else { if (DECK.f > 0) deckGo(DECK.i, DECK.f - 1); else if (DECK.i > 0) deckGo(DECK.i - 1, SLIDES[DECK.i - 1].frames - 1); else return false; }
  return true;
}
function deckStop() { if (DECK.playing) deckPlay(); }
function deckPlay() {
  DECK.playing = !DECK.playing;
  clearInterval(DECK.timer);
  if (!DECK.playing) return;
  if (DECK.f === SLIDES[DECK.i].frames - 1) deckGo(DECK.i, 0);          // sondaysa slaytı baştan oynat
  DECK.timer = setInterval(() => {                                       // GIF gibi: slaytın sonunda durur
    if (DECK.f < SLIDES[DECK.i].frames - 1) deckGo(DECK.i, DECK.f + 1); else deckStop();
  }, SLIDES[DECK.i].delay || 1300);
}
function deckLoop() {   // sürekli animasyonu olan slayt (tick) yalnız görünürken çizilir
  cancelAnimationFrame(DECK.raf);
  const s = SLIDES[DECK.i]; if (!s.tick || S.tab !== 'ana') return;
  const el = deckEls()[DECK.i], still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fr = t => { s.tick(el, still ? 900 : t); if (!still && S.tab === 'ana' && !document.hidden) DECK.raf = requestAnimationFrame(fr); };
  DECK.raf = requestAnimationFrame(fr);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden && S.tab === 'ana') deckLoop(); });
$('hit-r').onclick = () => { deckStop(); deckStep(1); };
$('hit-l').onclick = () => { deckStop(); deckStep(-1); };
function deckKey(e) {
  if (S.tab !== 'ana' || e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = (e.target && e.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) return;
  if (['ArrowRight', 'PageDown'].includes(e.key)) { e.preventDefault(); deckStop(); deckStep(1); }
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); deckStop(); deckStep(-1); }
  else if (e.key === ' ') { e.preventDefault(); deckPlay(); }
  else if (e.key === 'Home') { deckStop(); deckGo(0, 0); }
  else if (e.key === 'End') { deckStop(); deckGo(SLIDES.length - 1, SLIDES[SLIDES.length - 1].frames - 1); }
}
document.addEventListener('keydown', deckKey);
// Odak Streamlit sayfasındayken de oklar çalışsın (bileşen aynı kökten sunulur; erişilemezse yalnız iframe dinler)
try {
  const P = window.parent;
  if (P !== window) { if (P.__deckKey) P.document.removeEventListener('keydown', P.__deckKey); P.__deckKey = deckKey; P.document.addEventListener('keydown', deckKey); }
} catch (_) { /* farklı köken */ }
// Sunum alanı ekranı doldurur: Streamlit sayfasının görünür yüksekliği − üst bar
function viewportH() { try { return window.parent.innerHeight; } catch (_) { return 0; } }
function sizeDeck() {
  if (S.tab !== 'ana') return;
  const vh = viewportH() || 820, bar = document.querySelector('.bar1').getBoundingClientRect().height;
  $('deck').style.height = Math.max(380, Math.floor(vh - bar - 30 - 6 - 8 - 8)) + 'px';   // 30: sayfa kenar boşluğu, 8: aralık
  const s = SLIDES[DECK.i]; if (s && s.resize && $('slides').children.length) requestAnimationFrame(() => s.resize(deckEls()[DECK.i], A.ana || {}));
}
try { window.parent.addEventListener('resize', () => { sizeDeck(); fitHeight(); }); } catch (_) { /* farklı köken */ }
$('logo').onclick = () => {
  if (S.tab === 'ana') return;
  stopPlay();
  $('tabs').querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', false));
  send('tab', {tab: 'ana'});
};

// ---------------------------------------------------------------- üst bar
const TABS = [['ozet', 'grid', 'Özet'], ['gezgin', 'radar', 'Fırtına Gezgini'], ['kars', 'bars', 'Model Karşılaştırma'], ['hata', 'bug', 'Hata Analizi'], ['xai', 'bulb', 'Açıklanabilirlik'], ['harita', 'map', 'Harita']];
const SPLIT_ICO = {all: 'db', train: 'dumbbell', val: 'check-c', test: 'flask'};
const SPLIT_N = {all: 'Tümü', train: 'Eğitim', val: 'Doğrulama', test: 'Test'};
function pickDock(el, items, selected, onPick, cls = 'pick') {
  // items: [{id, icon|dot, name, tip}]
  el.innerHTML = items.map(it => `<button class="rb ${cls}" data-id="${esc(it.id)}" aria-pressed="${Array.isArray(selected) ? selected.includes(it.id) : it.id === selected}" data-tip="${esc(it.tip || it.name)}" aria-label="${esc(it.name)}">` +
    (it.dot ? `<span class="dot" style="background:${it.dot}"></span>` : ico(it.icon, it.iconStyle || '')) + (cls === 'tog' ? '' : `<span class="n">${esc(it.name)}</span>`) + '</button>').join('');
  el.querySelectorAll('button').forEach(b => b.onclick = () => onPick(b.dataset.id, b));
}
function setPressed(el, id) { el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.id === id)); }

function renderTabs() {
  $('tabs').innerHTML = TABS.map(([id, ic, name]) => `<button class="rb tab" role="tab" data-id="${id}" aria-selected="${id === S.tab}" data-tip="${name}" aria-label="${name}">${ico(ic)}</button>`).join('');
  $('tabs').querySelectorAll('button').forEach(b => b.onclick = () => {
    if (b.dataset.id === S.tab) return; stopPlay();
    $('tabs').querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', x === b));
    send('tab', {tab: b.dataset.id});
  });
  TABS.forEach(([id]) => $('ctx-' + id).hidden = id !== S.tab);
  // Gösterilen modeller: ana + ablasyon (varsayılan) · + ek seed'ler · hepsi (04 denemeleri, hızlı denemeler)
  const vc = A.view_counts || {};
  const MV = [['core', 'tornado', 'Temel', 'Temel 3: doğrulamada en iyi TornadoCNN, en iyi ResNet-18 ve öteki ResNet-18'], ['main', 'check-c', 'Ana', 'Ana karşılaştırma + ablasyonlar'], ['seeds', 'split', '+ Seed', 'Ek seed\'ler dahil'],
              ['all', 'db', 'Hepsi', 'Hiperparametre ve hızlı denemeler dahil bütün modeller']];
  pickDock($('mview'), MV.map(([id, icon, name, tip]) => ({id, icon, name: `${name} ${vc[id] ?? ''}`, tip: `${tip} · ${vc[id] ?? '?'} model`})),
           A.model_view || 'core', id => { if (id === A.model_view) return; setPressed($('mview'), id); send('model_view', {view: id}); });
}
const modelItems = () => A.models.map(m => ({id: m.key, dot: m.color, name: m.label, tip: m.label}));
const splitItems = ids => ids.map(id => ({id, icon: SPLIT_ICO[id], name: SPLIT_N[id], tip: SPLIT_N[id]}));

function renderToolbars() {
  if (S.tab === 'ozet') {
    const o = A.ozet;
    pickDock($('oz-models'), modelItems(), o.model, id => { setPressed($('oz-models'), id); send('ozet_model', {key: id}); });
    $('oz-kpis').innerHTML = (o.kpis || []).map(([k, v]) => `<span><i>${esc(k)}</i>${esc(v)}</span>`).join('') || '<span><i>Ölçüt</i>—</span>';
    pickDock($('oz-split'), splitItems(['val', 'test']), o.split, id => { setPressed($('oz-split'), id); send('ozet_split', {split: id}); });
  } else if (S.tab === 'gezgin') {
    const g = A.gezgin;
    const FIL = [['all', 'layers', 'Tümü'], ['tor', 'tornado', 'Hortumlu'], ['ef2', 'flame', 'EF2+'], ['fa', 'bell-off', 'Yanlış alarm'], ['nul', 'circle-dashed', 'NUL']];
    const el = $('gz-filters');
    el.innerHTML = FIL.map(([id, ic, name]) => `<button class="rb chip" data-id="${id}" aria-pressed="${id === g.filter}" data-tip="${name} · ${fmtN(g.counts[id] || 0)}" aria-label="${name}">${ico(ic)}<span class="n">${fmtN(g.counts[id] || 0)}</span></button>`).join('');
    el.querySelectorAll('button').forEach(b => b.onclick = () => { if (b.dataset.id === g.filter) return; stopPlay(); setPressed(el, b.dataset.id); send('filter', {filter: b.dataset.id}); });
    pickDock($('gz-split'), splitItems(['all', 'train', 'val', 'test']), g.split, id => { if (id === g.split) return; stopPlay(); setPressed($('gz-split'), id); send('split', {split: id}); }, 'sp');
    STRIP = {codes: g.strip.codes, ids: g.strip.ids, pos: g.pos};
    requestAnimationFrame(drawStrip);
  } else if (S.tab === 'kars') {
    const k = A.kars;
    pickDock($('ks-models'), k.items || modelItems(), k.models, (id, b) => {
      const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', on);
      const keys = [...$('ks-models').querySelectorAll('button')].filter(x => x.getAttribute('aria-pressed') === 'true').map(x => x.dataset.id);
      send('kars_models', {keys});
    }, 'tog');
    const CHARTS = [['table', 'layers', 'Tablo', 'Ölçüt tablosu: %95 güven aralığı, Δ ve NC / RC / WC'], ['roc', 'roc', 'ROC', 'ROC eğrisi'], ['pr', 'pr', 'PR', 'Precision–Recall'], ['perf', 'crosshair', 'Performans', 'Performans diyagramı (SR–POD, CSI)'], ['rel', 'check-c', 'Güvenilirlik', 'Güvenilirlik diyagramı (kalibrasyon)'], ['ef', 'flame', 'EF', 'EF\'ye göre skor dağılımı'], ['range', 'ruler', 'Uzaklık', 'Radara uzaklığa göre başarım'], ['cm', 'grid', 'Karmaşıklık', 'Karmaşıklık matrisi'], ['learn', 'trend', 'Öğrenme', 'Öğrenme eğrileri'], ['hp', 'sliders', 'Hiperparametre', 'Hiperparametre tablosu']];
    pickDock($('ks-chart'), CHARTS.map(([id, icon, name, tip]) => ({id, icon, name, tip})), k.chart, id => { setPressed($('ks-chart'), id); send('kars_chart', {chart: id}); });
    pickDock($('ks-split'), splitItems(['val', 'test']), k.split, id => { setPressed($('ks-split'), id); send('kars_split', {split: id}); });
  } else if (S.tab === 'hata') {
    const h = A.hata;
    const KINDS = [['fn', 'x', 'Kaçırdı', 'Kaçırılan hortumlar', 'color:var(--tor)'], ['fp', 'alert', 'Yanlış alarm', 'Yanlış alarmlar', 'color:var(--accent)'], ['tp', 'check', 'Yakaladı', 'Doğru yakalananlar', 'color:var(--good)']];
    pickDock($('ht-kind'), KINDS.map(([id, icon, name, tip, st]) => ({id, icon, iconStyle: st, name: `${name} ${fmtN(h.counts[id] || 0)}`, tip})), h.kind, id => { setPressed($('ht-kind'), id); send('hata_kind', {kind: id}); });
    pickDock($('ht-models'), modelItems(), h.model, id => { setPressed($('ht-models'), id); send('hata_model', {key: id}); });
    pickDock($('ht-split'), splitItems(['val', 'test']), h.split, id => { setPressed($('ht-split'), id); send('hata_split', {split: id}); });
    const thr = $('thr'); thr.value = h.thr; $('thr-v').textContent = Number(h.thr).toFixed(3);
    thr.oninput = e => $('thr-v').textContent = (+e.target.value).toFixed(3);
    thr.onchange = e => send('hata_thr', {thr: +e.target.value});
    $('ht-cam').setAttribute('aria-pressed', !!h.cam);
    $('ht-cam').onclick = () => { const on = $('ht-cam').getAttribute('aria-pressed') !== 'true'; $('ht-cam').setAttribute('aria-pressed', on); send('hata_cam', {on}); };
    $('ht-sort').setAttribute('aria-pressed', !!h.sort_conf);
    $('ht-sort').onclick = () => { const on = $('ht-sort').getAttribute('aria-pressed') !== 'true'; $('ht-sort').setAttribute('aria-pressed', on); send('hata_sort', {on}); };
    $('ht-comp').setAttribute('aria-pressed', !!h.comp);
    $('ht-comp').onclick = () => { const on = $('ht-comp').getAttribute('aria-pressed') !== 'true'; $('ht-comp').setAttribute('aria-pressed', on); send('hata_comp', {on}); };
  } else if (S.tab === 'harita') {
    const h = A.harita;
    const HV = [['genel', 'map', 'Genel', 'Bütün örnekler haritada; tıklanan kare haritaya oturtulur'], ['istasyon', 'radar', 'İstasyon', 'İstasyona göre başarım (en az 30 hortum karesi)']];
    pickDock($('hr-view'), HV.map(([id, icon, name, tip]) => ({id, icon, name, tip})), h.view, id => { setPressed($('hr-view'), id); send('harita_view', {view: id}); });
    const HC = h.view === 'genel'
      ? [['cat', 'layers', 'Kategori', 'Renk: TOR / WRN / NUL'], ['score', 'trend', 'Skor', 'Renk: seçili modelin skoru'], ['kind', 'check', 'Karar', 'Renk: isabet / ıska / yanlış alarm / doğru ret']]
      : [['prauc', 'pr', 'PR-AUC', 'Renk: istasyonun PR-AUC değeri'], ['csi', 'crosshair', 'CSI', 'Renk: istasyonun CSI değeri (seçili eşikte)']];
    pickDock($('hr-color'), HC.map(([id, icon, name, tip]) => ({id, icon, name, tip})), h.color, id => { setPressed($('hr-color'), id); send('harita_color', {color: id}); });
    pickDock($('hr-models'), modelItems(), h.model, id => { setPressed($('hr-models'), id); send('harita_model', {key: id}); });
    $('hr-split').hidden = h.view !== 'genel';
    pickDock($('hr-split'), splitItems(['all', 'train', 'val', 'test']), h.split, id => { setPressed($('hr-split'), id); send('harita_split', {split: id}); });
  } else if (S.tab === 'xai') {
    const x = A.xai;
    pickDock($('ax-models'), modelItems(), x.model, id => { setPressed($('ax-models'), id); send('xai_model', {key: id}); });
    const VIEWS = [['frame', 'crosshair', 'Tek kare', 'Seçili karenin açıklaması'], ['global', 'layers', 'Genel', 'Test setinin toplu analizi']];
    pickDock($('ax-view'), VIEWS.map(([id, icon, name, tip]) => ({id, icon, name, tip})), x.view, id => { setPressed($('ax-view'), id); send('xai_view', {view: id}); });
    const METH = [['cam', 'eye', 'Grad-CAM', 'Grad-CAM: model nereye bakıyor (kaba)'], ['ig', 'crosshair', 'Int. Gradients', 'Integrated Gradients: piksel ve kanal düzeyinde katkı'], ['occ', 'grid', 'Occlusion', 'Occlusion: bölge kapatılınca skor ne kadar düşüyor'], ['loc', 'pin', 'Konum', 'Konum haritası: modelin kendi sınıflandırıcısı her konumda (yaklaşık)']];
    pickDock($('ax-method'), METH.map(([id, icon, name, tip]) => ({id, icon, name, tip})), x.method, id => { setPressed($('ax-method'), id); send('xai_method', {method: id}); });
    const isFrame = x.view === 'frame';
    $('ax-frame-nav').hidden = !isFrame || !x.pos;
    $('ax-split').hidden = true;          // genel görünümün bölümü xai.parquet'ten gelir (make_xai.py --split)
    $('ax-pos').textContent = x.pos || '';
    $('ax-prev').onclick = () => send('xai_frame', {step: -1});
    $('ax-next').onclick = () => send('xai_frame', {step: 1});
    $('ax-open').onclick = () => send('xai_open');
  }
}

// ---------------------------------------------------------------- çizim girişi
let iconsKey = null;
function render(state) {
  if (!state) return;
  A = state;
  if (iconsKey !== state.icons_key) {
    $('sprite').innerHTML = state.icons; iconsKey = state.icons_key;
    const logo = $('logo');
    if (state.logo) { logo.innerHTML = state.logo; logo.classList.toggle('custom', !!state.logo_custom); } else logo.innerHTML = ico('tornado');
  }
  S.tab = state.tab;
  document.body.classList.toggle('only', state.tab !== 'gezgin');
  $('app').classList.toggle('home', state.tab === 'ana');
  $('ana').hidden = state.tab !== 'ana';
  if (state.tab !== 'ana') { deckStop(); cancelAnimationFrame(DECK.raf); }
  document.querySelectorAll('.rb[data-tip]').forEach(b => b.removeAttribute('title'));
  renderTabs();
  renderToolbars();
  if (state.tab === 'ana') {
    $('gezgin').hidden = true; $('empty').hidden = true; stopPlay(); addTitles();
    sizeDeck(); deckBuild(); deckLoop(); requestAnimationFrame(() => { sizeDeck(); fitHeight(); }); return;
  }
  if (state.tab !== 'gezgin') { $('gezgin').hidden = true; $('empty').hidden = true; stopPlay(); addTitles(); requestAnimationFrame(fitHeight); return; }

  const g = state.gezgin;
  if (!g.episode) {
    $('gezgin').hidden = true; $('empty').hidden = false; $('empty').innerHTML = g.message || 'Bu filtrede fırtına yok.';
    EP = null; requestAnimationFrame(fitHeight); return;
  }
  $('empty').hidden = true; $('gezgin').hidden = false;
  const e = g.episode;
  const newEp = !EP || EP.key !== e.key;
  if (newEp) {
    stopPlay();
    EP = decodeEpisode(e);
    S.frame = clamp(e.init_frame, 0, EP.F - 1); S.hover = null;
    const rc = EP.rot[S.frame];
    S.pin = rc ? cellPoint(rc[0], rc[1]) : null;
    if (!EP.cam && S.stage === 'CAM') S.stage = 'VEL';
    if (!EP.chans[S.stage] && S.stage !== 'CAM') S.stage = EP.vars.includes('VEL') ? 'VEL' : EP.vars[0];
  } else if (e.cam && EP.camModel !== e.cam_model) {           // yalnız Grad-CAM modeli değişti
    EP.cam = decodeCam(e, EP.F, EP.NA * EP.NR); EP.camModel = e.cam_model;
  } else if (!e.cam) { EP.cam = null; if (S.stage === 'CAM') S.stage = 'VEL'; }
  EP.scores = e.scores;
  S.sel = state.sel_model;
  $('clk').textContent = EP.frames[S.frame].time;
  $('frm').textContent = `${S.frame + 1}/${EP.F} · ${EP.frames[S.frame].cat}`;
  requestAnimationFrame(() => { drawStage(); drawChannels(); drawModels(); drawTimeline(); drawStrip(); fitHeight(); });
}
function addTitles() { document.querySelectorAll('.rb[data-tip]').forEach(b => b.title = b.dataset.tip); }
window.addEventListener('resize', () => { if (EP && S.tab === 'gezgin') { drawTimeline(); drawStrip(); } if (S.tab === 'ana') sizeDeck(); });
})();
