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
  const single = Math.max(...boxes.map(area));
  ep.boxes = area(u) <= 2.2 * single ? boxes.map(() => u) : boxes;
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
    for (let a = a1; a <= aB; a += 10) { const p = proj.fwd(...toXY(a, rB + 7)); ctx.fillText((((a % 360) + 360) % 360) + '°', p[0], p[1]); }
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
    box.appendChild(b);
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
  const order = A.models.filter(m => m.key !== S.sel).concat(sm ? [sm] : []);
  for (const m of order) {
    const s = EP.scores[m.key]; if (!s) continue;
    const on = m.key === S.sel;
    const pts = s.map((v, t) => v == null ? null : `${X(t).toFixed(1)},${Y(v).toFixed(1)}`).filter(Boolean).join(' ');
    h += NF > 1 ? `<polyline points="${pts}" fill="none" stroke="${m.color}" stroke-width="${on ? 2.4 : 1.3}" stroke-linejoin="round" opacity="${on ? 1 : .5}"/>` : '';
    if (on && s[S.frame] != null) h += `<circle cx="${X(S.frame)}" cy="${Y(s[S.frame])}" r="4" fill="${m.color}" stroke="var(--panel)" stroke-width="1.5"/>`;
  }
  h += `<line x1="${X(S.frame)}" x2="${X(S.frame)}" y1="${T}" y2="${bot}" stroke="var(--accent)" stroke-width="2"/>`;
  const every = Math.ceil(44 / cw);
  for (let t = 0; t < NF; t++) { if (t % every && t !== S.frame) continue;
    h += `<text x="${X(t)}" y="${H - 4}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="10" fill="${t === S.frame ? 'var(--accent)' : 'var(--muted)'}">${EP.frames[t].time}</text>`; }
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
$('epid').addEventListener('keydown', e => { if (e.key === 'Enter' && /^-?\d+$/.test(e.target.value.trim())) { stopPlay(); send('epid', {id: e.target.value.trim()}); e.target.value = ''; e.target.blur(); } });

// ---------------------------------------------------------------- üst bar
const TABS = [['ozet', 'grid', 'Özet'], ['gezgin', 'radar', 'Fırtına Gezgini'], ['kars', 'bars', 'Model Karşılaştırma'], ['hata', 'bug', 'Hata Analizi']];
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
    pickDock($('ks-models'), modelItems(), k.models, (id, b) => {
      const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', on);
      const keys = [...$('ks-models').querySelectorAll('button')].filter(x => x.getAttribute('aria-pressed') === 'true').map(x => x.dataset.id);
      send('kars_models', {keys});
    }, 'tog');
    const CHARTS = [['roc', 'roc', 'ROC', 'ROC eğrisi'], ['pr', 'pr', 'PR', 'Precision–Recall'], ['cm', 'grid', 'Karmaşıklık', 'Karmaşıklık matrisi'], ['learn', 'trend', 'Öğrenme', 'Öğrenme eğrileri'], ['hp', 'sliders', 'Hiperparametre', 'Hiperparametre tablosu']];
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
  document.querySelectorAll('.rb[data-tip]').forEach(b => b.removeAttribute('title'));
  renderTabs();
  renderToolbars();
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
window.addEventListener('resize', () => { if (EP && S.tab === 'gezgin') { drawTimeline(); drawStrip(); } });
})();
