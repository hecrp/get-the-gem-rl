// Small canvas charts: lines (with bands, markers, log scale), bars with CI
// whiskers, histograms, scatter and heat maps. Identity is never colour-only:
// every series has a marker shape and a legend entry; proportions show n.
import { fmtNum, fmtSteps, fmtPct, escapeHtml } from './format.js';

const MARKERS = ['circle', 'square', 'triangle', 'diamond', 'cross', 'plus', 'tri-down', 'star'];
export const seriesColor = (i) => getComputedStyle(document.documentElement).getPropertyValue(`--s${(i % 8) + 1}`).trim();
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function ema(ys, alpha) {
  const out = new Array(ys.length);
  let acc = null;
  for (let i = 0; i < ys.length; i++) { acc = acc == null ? ys[i] : alpha * ys[i] + (1 - alpha) * acc; out[i] = acc; }
  return out;
}

function drawMarker(ctx, shape, x, y, r) {
  ctx.beginPath();
  switch (shape) {
    case 'square': ctx.rect(x - r, y - r, 2 * r, 2 * r); break;
    case 'triangle': ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.1, y + r); ctx.lineTo(x - r * 1.1, y + r); ctx.closePath(); break;
    case 'tri-down': ctx.moveTo(x, y + r * 1.2); ctx.lineTo(x + r * 1.1, y - r); ctx.lineTo(x - r * 1.1, y - r); ctx.closePath(); break;
    case 'diamond': ctx.moveTo(x, y - r * 1.3); ctx.lineTo(x + r * 1.3, y); ctx.lineTo(x, y + r * 1.3); ctx.lineTo(x - r * 1.3, y); ctx.closePath(); break;
    case 'cross': ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); return;
    case 'plus': ctx.moveTo(x - r, y); ctx.lineTo(x + r, y); ctx.moveTo(x, y - r); ctx.lineTo(x, y + r); ctx.stroke(); return;
    case 'star': for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.5 : r * 1.3; ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } ctx.closePath(); break;
    default: ctx.arc(x, y, r, 0, 2 * Math.PI);
  }
  ctx.fill();
}

export function markerSvg(i, color) {
  const shape = MARKERS[i % MARKERS.length];
  const c = document.createElement('canvas');
  c.width = c.height = 12;
  const ctx = c.getContext('2d');
  ctx.fillStyle = ctx.strokeStyle = color; ctx.lineWidth = 2;
  drawMarker(ctx, shape, 6, 6, 3.2);
  return `<img alt="" width="12" height="12" src="${c.toDataURL()}">`;
}

function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

class Base {
  constructor(el, opts = {}) {
    this.el = el; this.opts = opts;
    el.classList.add('chart');
    el.innerHTML = '<canvas role="img"></canvas><div class="legend"></div><div class="note"></div><div class="tooltip" hidden></div>';
    this.canvas = el.querySelector('canvas');
    this.legend = el.querySelector('.legend');
    this.noteEl = el.querySelector('.note');
    this.tip = el.querySelector('.tooltip');
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.canvas);
    this.canvas.addEventListener('mousemove', (e) => this.hover(e));
    this.canvas.addEventListener('mouseleave', () => { this.tip.hidden = true; this.hoverX = null; this.draw(); });
  }
  ctx2d() {
    const r = this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
    if (this.canvas.width !== w * dpr || this.canvas.height !== h * dpr) { this.canvas.width = w * dpr; this.canvas.height = h * dpr; }
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '11px system-ui, sans-serif';
    return [ctx, w, h];
  }
  note(text) { this.noteEl.textContent = text || ''; }
  showTip(html, x, y) {
    this.tip.innerHTML = html; this.tip.hidden = false;
    const w = this.el.clientWidth, tw = this.tip.offsetWidth;
    this.tip.style.left = Math.max(2, Math.min(w - tw - 2, x + 12)) + 'px';
    this.tip.style.top = Math.max(2, y - 10) + 'px';
  }
  destroy() { this.ro.disconnect(); }
  hover() {}
}

/** Line chart. series: [{label, x, y, color?, idx, faint?, band?: {x, lo, hi}, points?}] */
export class LineChart extends Base {
  set(series, opts = {}) {
    this.series = series; Object.assign(this.opts, opts);
    const named = series.filter((s) => !s.faint && s.label);
    this.legend.innerHTML = named.length >= 2 || this.opts.legend ? named.map((s) =>
      `<span>${markerSvg(s.idx ?? 0, s.color || seriesColor(s.idx ?? 0))}${escapeHtml(s.label)}</span>`).join('') : '';
    this.canvas.setAttribute('aria-label', this.opts.aria || named.map((s) => s.label).join(', '));
    this.draw();
  }
  scales(w, h) {
    const pad = { l: 44, r: 10, t: 8, b: 22 };
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of this.series || []) {
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (y == null || !Number.isFinite(y) || (this.opts.logY && y <= 0)) continue;
        x0 = Math.min(x0, s.x[i]); x1 = Math.max(x1, s.x[i]); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
      if (s.band) for (let i = 0; i < s.band.x.length; i++) {
        if (s.band.lo[i] != null) y0 = Math.min(y0, s.band.lo[i]);
        if (s.band.hi[i] != null) y1 = Math.max(y1, s.band.hi[i]);
      }
    }
    if (this.opts.yMin != null) y0 = Math.min(y0, this.opts.yMin);
    if (this.opts.yMax != null) y1 = Math.max(y1, this.opts.yMax);
    if (!Number.isFinite(x0)) return null;
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) { y1 += 0.5; y0 -= 0.5; }
    const log = this.opts.logY;
    const ty = (v) => (log ? Math.log10(v) : v);
    const Y0 = ty(y0), Y1 = ty(y1);
    return {
      pad, x0, x1, y0, y1, log,
      sx: (v) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r),
      sy: (v) => h - pad.b - ((ty(v) - Y0) / (Y1 - Y0)) * (h - pad.t - pad.b),
      ix: (px) => x0 + ((px - pad.l) / (w - pad.l - pad.r)) * (x1 - x0),
    };
  }
  draw() {
    if (!this.series) return;
    const [ctx, w, h] = this.ctx2d();
    const S = this.scales(w, h);
    if (!S) { ctx.fillStyle = css('--muted'); ctx.fillText('sin datos', 8, 18); return; }
    this.S = S;
    const xf = this.opts.xFmt || fmtSteps, yf = this.opts.yFmt || ((v) => fmtNum(v, Math.abs(v) < 10 ? 2 : 0));
    ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.lineWidth = 1;
    const yt = S.log ? [0.001, 0.01, 0.1, 1, 10, 100, 1000].filter((v) => v >= S.y0 && v <= S.y1) : niceTicks(S.y0, S.y1, 4);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of yt) { const y = Math.round(S.sy(v)) + 0.5; ctx.beginPath(); ctx.moveTo(S.pad.l, y); ctx.lineTo(w - S.pad.r, y); ctx.stroke(); ctx.fillText(yf(v), S.pad.l - 4, y); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const v of niceTicks(S.x0, S.x1, Math.max(2, Math.floor(w / 110)))) ctx.fillText(xf(v), S.sx(v), h - S.pad.b + 5);
    ctx.strokeStyle = css('--axis'); ctx.beginPath(); ctx.moveTo(S.pad.l, h - S.pad.b + 0.5); ctx.lineTo(w - S.pad.r, h - S.pad.b + 0.5); ctx.stroke();
    const band = parseFloat(css('--band')) || 0.18;
    for (const s of this.series) {
      const color = s.color || seriesColor(s.idx ?? 0);
      if (s.band) {
        ctx.globalAlpha = band; ctx.fillStyle = color; ctx.beginPath();
        let started = false;
        for (let i = 0; i < s.band.x.length; i++) if (s.band.hi[i] != null) { const X = S.sx(s.band.x[i]), Y = S.sy(s.band.hi[i]); started ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); started = true; }
        for (let i = s.band.x.length - 1; i >= 0; i--) if (s.band.lo[i] != null) ctx.lineTo(S.sx(s.band.x[i]), S.sy(s.band.lo[i]));
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = color; ctx.lineWidth = s.faint ? 1 : 2; ctx.globalAlpha = s.faint ? 0.3 : 1;
      ctx.setLineDash(s.dash || []); ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (y == null || !Number.isFinite(y) || (S.log && y <= 0)) { pen = false; continue; }
        const X = S.sx(s.x[i]), Y = S.sy(y);
        pen ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); pen = true;
      }
      ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      if (!s.faint) {
        ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 2;
        const shape = MARKERS[(s.idx ?? 0) % MARKERS.length];
        const every = s.points ? 1 : Math.max(1, Math.ceil(s.x.length / 8));
        for (let i = 0; i < s.x.length; i += every) {
          if (s.y[i] == null || !Number.isFinite(s.y[i])) continue;
          drawMarker(ctx, shape, S.sx(s.x[i]), S.sy(s.y[i]), 3.5);
        }
      }
    }
    if (this.hoverX != null) {
      ctx.strokeStyle = css('--axis'); ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(this.hoverX + 0.5, S.pad.t); ctx.lineTo(this.hoverX + 0.5, h - S.pad.b); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  hover(e) {
    if (!this.S || !this.series) return;
    const r = this.canvas.getBoundingClientRect(), px = e.clientX - r.left, xv = this.S.ix(px);
    this.hoverX = px; this.draw();
    const xf = this.opts.xFmt || fmtSteps, yf = this.opts.tipFmt || this.opts.yFmt || ((v) => fmtNum(v, 3));
    const rows = [];
    for (const s of this.series) {
      if (s.faint || !s.x.length) continue;
      let best = -1, bd = Infinity;
      for (let i = 0; i < s.x.length; i++) { const d = Math.abs(s.x[i] - xv); if (d < bd && s.y[i] != null) { bd = d; best = i; } }
      if (best < 0) continue;
      const bandTxt = s.band && s.band.lo ? (() => { let j = 0, d2 = Infinity; s.band.x.forEach((x, k) => { const d = Math.abs(x - s.x[best]); if (d < d2) { d2 = d; j = k; } }); return s.band.lo[j] != null ? ` [${yf(s.band.lo[j])}–${yf(s.band.hi[j])}]` : ''; })() : '';
      rows.push(`${markerSvg(s.idx ?? 0, s.color || seriesColor(s.idx ?? 0))} ${escapeHtml(s.label || '')}: <b>${yf(s.y[best])}</b>${bandTxt} <span class="muted">@ ${xf(s.x[best])}</span>`);
    }
    if (rows.length) this.showTip(rows.join('<br>'), px, e.clientY - r.top); else this.tip.hidden = true;
  }
}

/** Bars with 95 % CI whiskers: [{label, p, lo, hi, n, insufficient?, idx?}] (proportions). */
export class BarChart extends Base {
  set(bars, opts = {}) { this.bars = bars; Object.assign(this.opts, opts); this.legend.innerHTML = this.opts.legendHtml || ''; this.draw(); }
  draw() {
    if (!this.bars) return;
    const [ctx, w, h] = this.ctx2d();
    const pad = { l: 38, r: 8, t: 8, b: this.opts.rotate ? 48 : 30 };
    const n = this.bars.length;
    if (!n) { ctx.fillStyle = css('--muted'); ctx.fillText('sin datos', 8, 18); return; }
    const max = this.opts.max ?? 1;
    const sy = (v) => h - pad.b - (v / max) * (h - pad.t - pad.b);
    ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of niceTicks(0, max, 4)) { const y = Math.round(sy(v)) + 0.5; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.fillText(this.opts.pct === false ? fmtNum(v, 0) : fmtPct(v), pad.l - 4, y); }
    const bw = (w - pad.l - pad.r) / n;
    this.hits = [];
    this.bars.forEach((b, i) => {
      const x = pad.l + i * bw + 2, ww = Math.max(2, bw - 4);
      const color = b.color || seriesColor(b.idx ?? 0);
      ctx.globalAlpha = b.insufficient ? 0.35 : 1;
      ctx.fillStyle = color;
      if (b.p != null) {
        const y = sy(b.p), r = Math.min(4, ww / 2, Math.max(0, sy(0) - y));
        ctx.beginPath(); ctx.moveTo(x, sy(0)); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.lineTo(x + ww - r, y); ctx.quadraticCurveTo(x + ww, y, x + ww, y + r); ctx.lineTo(x + ww, sy(0)); ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (b.lo != null && b.hi != null) {
        ctx.strokeStyle = css('--text'); ctx.lineWidth = 1.5; const cx = x + ww / 2;
        ctx.beginPath(); ctx.moveTo(cx, sy(b.lo)); ctx.lineTo(cx, sy(b.hi));
        ctx.moveTo(cx - 4, sy(b.lo)); ctx.lineTo(cx + 4, sy(b.lo)); ctx.moveTo(cx - 4, sy(b.hi)); ctx.lineTo(cx + 4, sy(b.hi)); ctx.stroke();
      }
      ctx.fillStyle = css('--text-2'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.save(); ctx.translate(x + ww / 2, h - pad.b + 3);
      if (this.opts.rotate && ww < ctx.measureText(String(b.label)).width + 4) { ctx.rotate(-Math.PI / 5); ctx.textAlign = 'right'; }
      ctx.fillText(String(b.label).slice(0, 18), 0, 0); ctx.restore();
      ctx.fillStyle = css('--muted'); ctx.textAlign = 'center';
      if (!this.opts.rotate) ctx.fillText(b.insufficient ? `n=${b.n} ⚠` : `n=${b.n}`, x + ww / 2, h - pad.b + 15);
      this.hits.push([x, x + ww, b]);
    });
  }
  hover(e) {
    const r = this.canvas.getBoundingClientRect(), px = e.clientX - r.left;
    const hit = (this.hits || []).find(([a, b]) => px >= a - 2 && px <= b + 2);
    if (!hit) { this.tip.hidden = true; return; }
    const b = hit[2];
    const v = this.opts.pct === false ? fmtNum(b.p, 0) : `${fmtPct(b.p, 1)} [${fmtPct(b.lo, 1)}–${fmtPct(b.hi, 1)}]`;
    this.showTip(`<b>${escapeHtml(b.label)}</b><br>${v} · n=${b.n}${b.insufficient ? '<br>⚠ n insuficiente' : ''}${b.extra ? '<br>' + b.extra : ''}`, px, e.clientY - r.top);
  }
}

/** Histogram {edges, counts}. */
export class Histogram extends Base {
  set(hist, opts = {}) { this.hist = hist; Object.assign(this.opts, opts); this.draw(); }
  draw() {
    if (!this.hist) return;
    const [ctx, w, h] = this.ctx2d();
    const { edges, counts } = this.hist;
    if (!counts || !counts.length) { ctx.fillStyle = css('--muted'); ctx.fillText('sin datos', 8, 18); return; }
    const pad = { l: 38, r: 8, t: 8, b: 22 }, max = Math.max(...counts);
    const x0 = edges[0], x1 = edges[edges.length - 1];
    const sx = (v) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r), sy = (v) => h - pad.b - (v / (max || 1)) * (h - pad.t - pad.b);
    ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of niceTicks(0, max, 3)) { const y = Math.round(sy(v)) + 0.5; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.fillText(fmtNum(v, 0), pad.l - 4, y); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(w / 90)))) ctx.fillText(fmtNum(v, 0), sx(v), h - pad.b + 5);
    ctx.fillStyle = this.opts.color || seriesColor(this.opts.idx ?? 0);
    this.hits = [];
    counts.forEach((c, i) => {
      const a = sx(edges[i]) + 1, b = sx(edges[i + 1]) - 1;
      if (c) ctx.fillRect(a, sy(c), Math.max(1, b - a), sy(0) - sy(c));
      this.hits.push([a, b, i]);
    });
  }
  hover(e) {
    const r = this.canvas.getBoundingClientRect(), px = e.clientX - r.left;
    const hit = (this.hits || []).find(([a, b]) => px >= a - 1 && px <= b + 1);
    if (!hit) { this.tip.hidden = true; return; }
    const i = hit[2], { edges, counts, n } = this.hist;
    this.showTip(`${fmtNum(edges[i], 0)}–${fmtNum(edges[i + 1], 0)}: <b>${counts[i]}</b> (${fmtPct(counts[i] / n, 1)} de n=${n})`, px, e.clientY - r.top);
  }
}

/** Scatter of {x, y, label}. */
export class Scatter extends Base {
  set(points, opts = {}) { this.points = points; Object.assign(this.opts, opts); this.draw(); }
  draw() {
    if (!this.points) return;
    const [ctx, w, h] = this.ctx2d();
    const pad = { l: 42, r: 8, t: 8, b: 30 };
    if (!this.points.length) return;
    const xs = this.points.map((p) => p.x), ys = this.points.map((p) => p.y);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (x1 === x0) x1 += 1; if (y1 === y0) y1 += 1;
    const sx = (v) => pad.l + ((v - x0) / (x1 - x0)) * (w - pad.l - pad.r), sy = (v) => h - pad.b - ((v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    ctx.strokeStyle = css('--grid'); ctx.fillStyle = css('--muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of niceTicks(y0, y1, 4)) { const y = sy(v); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.fillText(fmtNum(v, 2), pad.l - 4, y); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const v of niceTicks(x0, x1, 5)) ctx.fillText(fmtNum(v, 2), sx(v), h - pad.b + 4);
    ctx.fillText(this.opts.xLabel || '', (w + pad.l) / 2, h - 12);
    ctx.save(); ctx.translate(10, (h - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(this.opts.yLabel || '', 0, -4); ctx.restore();
    ctx.fillStyle = seriesColor(0); ctx.globalAlpha = 0.7;
    this.hits = this.points.map((p) => { const X = sx(p.x), Y = sy(p.y); drawMarker(ctx, 'circle', X, Y, 3); return [X, Y, p]; });
    ctx.globalAlpha = 1;
  }
  hover(e) {
    const r = this.canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = null, bd = 100;
    for (const [X, Y, p] of this.hits || []) { const d = (X - px) ** 2 + (Y - py) ** 2; if (d < bd) { bd = d; best = p; } }
    if (!best) { this.tip.hidden = true; return; }
    this.showTip(`${escapeHtml(best.label)}<br>${escapeHtml(this.opts.xLabel)}: ${fmtNum(best.x, 3)} · ${escapeHtml(this.opts.yLabel)}: ${fmtNum(best.y, 3)}`, px, py);
  }
}

/** Sequential single-hue ramp (blue), light->dark in light mode, dark->light in dark mode. */
export function sequential(t) {
  const ramp = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
  const dark = document.documentElement.dataset.theme === 'dark' ||
    (document.documentElement.dataset.theme !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
  const r = dark ? [...ramp].reverse() : ramp;
  return r[Math.max(0, Math.min(r.length - 1, Math.floor(t * r.length)))];
}
