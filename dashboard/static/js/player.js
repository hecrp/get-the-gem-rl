// Replay player: header, pixel canvas, controls, scrub bar and action strip.
import { getReplay, getTrajectory, getLevel } from './api.js';
import { drawFrame, drawGhost, drawTrail, THEME, FLAG } from './renderer.js';
import { expandRLE, ACTION_SYMBOLS, ACTIONS, envIndexOfEngineStep } from './rle.js';
import { Clock, ticker, SPEEDS } from './clock.js';
import { replayNumber, outcomeLabel, sourceLabel, levelLabel, fmtInt, escapeHtml } from './format.js';
import { seriesColor } from './charts.js';

export const MAX_ANIMATING = 16;
const animating = new Set();
let focused = null;
export const focusedPlayer = () => focused;

const HOLD_MS = 1000;
const ACTION_BG = ['#4a5068', '#3d6fb8', '#c0632f', '#2f8f6b', '#6f5bc4', '#b5487a'];

export class ReplayPlayer {
  constructor(el, opts = {}) {
    this.el = el;
    this.opts = { camera: 'auto', speed: 1, loop: true, trail: false, ghosts: [], compact: false, ...opts };
    this.clock = opts.clock || new Clock();
    this.ownClock = !opts.clock;
    if (this.ownClock) this.clock.speed = this.opts.speed;
    this.hold = 0; this.visible = true; this.loaded = false; this.ghosts = []; this.poster = false;
    el.innerHTML = `
      <div class="player${this.opts.compact ? ' compact' : ''}" tabindex="0" role="group">
        <div class="player-head"></div>
        <div class="stage"><canvas class="game"></canvas><div class="overlay" hidden></div><div class="legend-ghosts" hidden></div><div class="poster" hidden>en espera (máx. ${MAX_ANIMATING} animando)</div></div>
        <canvas class="scrub" height="18" aria-label="Línea de tiempo"></canvas>
        <canvas class="strip" height="16" aria-hidden="true"></canvas>
        <div class="controls">
          <button class="btn icon" data-a="prev" title="Anterior (p)" aria-label="Replay anterior">⏮</button>
          <button class="btn icon" data-a="back" title="Paso atrás (,)" aria-label="Paso atrás">⏪</button>
          <button class="btn icon" data-a="play" title="Reproducir / pausa (Enter)" aria-label="Reproducir o pausar">⏸</button>
          <button class="btn icon" data-a="fwd" title="Paso adelante (.)" aria-label="Paso adelante">⏩</button>
          <button class="btn icon" data-a="next" title="Siguiente (n)" aria-label="Replay siguiente">⏭</button>
          <select data-a="speed" aria-label="Velocidad">${SPEEDS.map((s) => `<option value="${s}">${String(s).replace('.', ',')}×</option>`).join('')}</select>
          <button class="btn icon opt" data-a="loop" title="Bucle (l)" aria-label="Bucle" aria-pressed="false">🔁</button>
          <select class="opt" data-a="camera" aria-label="Cámara" title="Cámara (c)"><option value="auto">auto</option><option value="full">full</option><option value="follow">follow</option></select>
          <button class="btn icon opt" data-a="trail" title="Estela" aria-label="Estela" aria-pressed="false">〰</button>
          <span class="t"></span>
        </div>
      </div>`;
    this.root = el.querySelector('.player');
    this.head = el.querySelector('.player-head');
    this.stage = el.querySelector('.stage');
    this.canvas = el.querySelector('canvas.game');
    this.ctx = this.canvas.getContext('2d');
    this.overlay = el.querySelector('.overlay');
    this.ghostLegend = el.querySelector('.legend-ghosts');
    this.posterEl = el.querySelector('.poster');
    this.scrub = el.querySelector('.scrub');
    this.strip = el.querySelector('.strip');
    this.tEl = el.querySelector('.t');
    this.speedSel = el.querySelector('[data-a=speed]');
    this.camSel = el.querySelector('[data-a=camera]');
    this.speedSel.value = String(this.clock.speed);
    this.camSel.value = this.opts.camera;
    this.syncButtons();
    el.querySelector('.controls').addEventListener('click', (e) => {
      const a = e.target.closest('[data-a]')?.dataset.a;
      if (a === 'play') this.togglePlay();
      else if (a === 'back') this.step(-1);
      else if (a === 'fwd') this.step(1);
      else if (a === 'next') this.nav(1);
      else if (a === 'prev') this.nav(-1);
      else if (a === 'loop') { this.opts.loop = !this.opts.loop; this.syncButtons(); }
      else if (a === 'trail') { this.opts.trail = !this.opts.trail; this.syncButtons(); this.render(); }
    });
    this.speedSel.addEventListener('change', () => { this.clock.speed = parseFloat(this.speedSel.value); });
    this.camSel.addEventListener('change', () => this.setCamera(this.camSel.value));
    this.root.addEventListener('keydown', (e) => this.key(e));
    this.root.addEventListener('focus', () => { focused = this; });
    const seek = (e) => {
      if (!this.traj) return;
      const r = this.scrub.getBoundingClientRect();
      this.seekTo(((e.clientX - r.left) / r.width) * this.traj.engine_steps);
    };
    this.scrub.addEventListener('pointerdown', (e) => { seek(e); this.scrub.setPointerCapture(e.pointerId); this.dragging = true; });
    this.scrub.addEventListener('pointermove', (e) => { if (this.dragging) seek(e); });
    this.scrub.addEventListener('pointerup', () => { this.dragging = false; });
    this.ro = new ResizeObserver(() => { this.fit(); this.render(); });
    this.ro.observe(this.stage);
    this.io = new IntersectionObserver((ents) => { this.visible = ents.some((x) => x.isIntersecting); });
    this.io.observe(this.root);
    this.unsub = ticker.add((dt, gs) => this.tick(dt, gs));
  }

  destroy() {
    this.unsub(); this.ro.disconnect(); this.io.disconnect(); animating.delete(this);
    if (focused === this) focused = null;
    this.el.innerHTML = '';
  }

  syncButtons() {
    const q = (a) => this.el.querySelector(`[data-a=${a}]`);
    q('loop').setAttribute('aria-pressed', String(!!this.opts.loop));
    q('trail').setAttribute('aria-pressed', String(!!this.opts.trail));
    q('play').textContent = this.clock.paused ? '▶' : '⏸';
    q('prev').hidden = q('next').hidden = !this.opts.onNav;
  }

  async load(id) {
    this.id = id; this.loaded = false; this.error = null;
    const token = (this.loadToken = Symbol());
    this.head.innerHTML = `<span class="num">…</span><span class="muted loading"></span>`;
    try {
      const meta = await getReplay(id);
      const level = await getLevel(meta.level);
      const cam = this.resolveCamera(level);
      const traj = await getTrajectory(id, cam === 'follow' ? 'follow' : 'none');
      if (token !== this.loadToken) return;
      this.meta = meta; this.level = level; this.traj = traj; this.cam = cam;
      this.actions = expandRLE(meta.actions);
      this.frameSkip = meta.env?.frame_skip || 1;
      if (this.ownClock) this.clock.t = 0;
      this.hold = 0;
      this.loaded = true;
      this.renderHead();
      await this.loadGhosts();
      this.fit(); this.render();
    } catch (e) {
      if (token !== this.loadToken) return;
      this.error = e;
      this.head.innerHTML = `<span class="error">✘ No se pudo cargar ${escapeHtml(id)}: ${escapeHtml(e.message)}</span>`;
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  resolveCamera(level) {
    const c = this.opts.camera;
    if (c === 'full' || c === 'follow') return c;
    return level.width <= 480 && level.height <= 320 ? 'full' : 'follow';
  }

  async setCamera(c) {
    this.opts.camera = c;
    if (!this.loaded) return;
    const cam = this.resolveCamera(this.level);
    if (cam !== this.cam) {
      const t = this.clock.t;
      this.traj = await getTrajectory(this.id, cam === 'follow' ? 'follow' : 'none');
      this.cam = cam;
      if (this.ownClock) this.clock.t = t;
      this.fit();
    }
    this.render();
  }

  async loadGhosts() {
    this.ghosts = []; this.ghostError = null;
    const ids = (this.opts.ghosts || []).filter((g) => g && g !== this.id);
    for (const [k, gid] of ids.entries()) {
      try {
        const gm = await getReplay(gid);
        if (gm.level.hash !== this.meta.level.hash) {
          this.ghostError = `fantasma ${gid} rechazado: es de otro nivel`;
          continue;
        }
        const gt = await getTrajectory(gid, 'none');
        this.ghosts.push({ id: gid, meta: gm, traj: gt, color: seriesColor(k + 1), label: ghostLabel(gm) });
      } catch (e) { this.ghostError = `fantasma ${gid}: ${e.message}`; }
    }
    if (this.ghosts.length || this.ghostError) {
      this.ghostLegend.hidden = false;
      this.ghostLegend.innerHTML = `<span>■ ${escapeHtml(ghostLabel(this.meta))} (principal)</span>` +
        this.ghosts.map((g) => `<span style="color:${g.color}">▢ ${escapeHtml(g.label)}</span>`).join('') +
        (this.ghostError ? `<span class="error">⚠ ${escapeHtml(this.ghostError)}</span>` : '');
    } else this.ghostLegend.hidden = true;
  }

  renderHead() {
    const m = this.meta, o = outcomeLabel(m.outcome);
    const runPill = m.run === '_reference' ? 'referencia' : m.run;
    this.head.innerHTML = `
      <span class="num">${replayNumber(m.n)}</span>
      <span class="pill" title="${escapeHtml(m.run)}">${escapeHtml(runPill)}</span>
      <span class="badge ${o.cls}">${o.label}</span>
      <span class="pill">${escapeHtml(m.policy)}</span>
      <span class="muted small">${escapeHtml(sourceLabel(m.source))}</span>
      <span class="muted small">${escapeHtml(levelLabel(m.level))}</span>
      <span class="muted small">${fmtInt(m.outcome?.engine_steps)} pasos</span>
      ${this.traj.drift ? '<span class="badge warn" title="El generador ya no produce este nivel; se usa la copia guardada">⚠ derivado</span>' : ''}
      ${m.broken ? `<span class="badge bad">✘ roto</span>` : ''}
      ${m.demo || (m.tags || []).includes('demo') ? '<span class="badge demo">DEMO</span>' : ''}`;
  }

  fit() {
    if (!this.loaded) return;
    const [cw, ch] = this.cam === 'follow' ? [240, 160] : [this.level.width, this.level.height];
    if (this.canvas.width !== cw || this.canvas.height !== ch) { this.canvas.width = cw; this.canvas.height = ch; }
    const sw = this.stage.clientWidth, sh = this.stage.clientHeight;
    if (!sw || !sh) return;
    let s = Math.min(sw / cw, sh / ch);
    if (s >= 1) s = Math.floor(s);
    this.canvas.style.width = `${Math.max(1, Math.round(cw * s))}px`;
    this.canvas.style.height = `${Math.max(1, Math.round(ch * s))}px`;
  }

  get index() {
    return this.traj ? Math.max(0, Math.min(this.traj.engine_steps, Math.floor(this.clock.t))) : 0;
  }

  tick(dt, globalSpeed) {
    if (!this.loaded || !this.visible) { animating.delete(this); return; }
    const running = this.ownClock ? !this.clock.paused : !this.clock.paused || this._lastT !== this.clock.t;
    if (!running) { animating.delete(this); return; }
    if (!animating.has(this)) {
      if (animating.size >= MAX_ANIMATING) {
        if (!this.poster) { this.poster = true; this.posterEl.hidden = false; this.seekTo(this.traj.engine_steps); }
        return;
      }
      animating.add(this);
      if (this.poster) { this.poster = false; this.posterEl.hidden = true; }
    }
    const N = this.traj.engine_steps;
    if (this.ownClock) {
      if (this.clock.t >= N) {
        this.hold += dt;
        this.showResult(true);
        if (this.hold >= HOLD_MS) {
          this.hold = 0; this.showResult(false);
          if (this.opts.loop) this.clock.t = 0;
          else if (this.opts.onEnd) { this.opts.onEnd(this); return; }
          else { this.clock.paused = true; this.syncButtons(); }
        }
      } else {
        this.clock.tick(dt, globalSpeed);
        if (this.clock.t >= N) this.clock.t = N;
      }
    } else this.showResult(this.clock.t >= N);
    this._lastT = this.clock.t;
    this.render();
  }

  showResult(on) {
    if (!on) { this.overlay.hidden = true; return; }
    this.overlay.hidden = false;
    this.overlay.textContent = outcomeLabel(this.meta.outcome).label;
  }

  render() {
    if (!this.loaded || !THEME) return;
    const i = this.index, t = this.traj;
    let cx = 0, cy = 0;
    if (this.cam === 'follow' && t.cam) { cx = t.cam[2 * i]; cy = t.cam[2 * i + 1]; }
    const ctx = this.ctx, cw = this.canvas.width, ch = this.canvas.height;
    drawFrame(ctx, this.level, { x: t.xy[2 * i], y: t.xy[2 * i + 1], flags: t.flags[i] }, cx, cy, cw, ch);
    for (const g of this.ghosts) {
      const j = Math.min(i, g.traj.engine_steps);
      if (this.opts.trail) drawTrail(ctx, g.traj.xy, j, 90, cx, cy, g.color);
      drawGhost(ctx, g.traj.xy[2 * j], g.traj.xy[2 * j + 1], cx, cy, g.color, (g.traj.flags[j] & FLAG.DEAD) !== 0);
    }
    if (this.opts.trail) drawTrail(ctx, t.xy, i, 90, cx, cy, '#ffd166');
    this.drawScrub(i); this.drawStrip(i);
    const env = envIndexOfEngineStep(Math.max(1, i), this.frameSkip);
    this.tEl.textContent = `${i}/${t.engine_steps} · env ${Math.min(env + (i > 0), t.env_steps)}/${t.env_steps}`;
    const now = performance.now();
    if (!this._ariaT || now - this._ariaT > 1000) {
      this._ariaT = now;
      this.root.setAttribute('aria-label', `Replay ${replayNumber(this.meta.n)} de ${this.meta.run}, ${outcomeLabel(this.meta.outcome).text}, ${levelLabel(this.meta.level)}, paso ${i} de ${t.engine_steps}`);
    }
  }

  drawScrub(i) {
    const c = this.scrub, w = c.clientWidth || 200, dpr = window.devicePixelRatio || 1;
    if (c.width !== w * dpr) { c.width = w * dpr; c.height = 18 * dpr; }
    const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const N = this.traj.engine_steps || 1, cs = getComputedStyle(document.documentElement);
    ctx.clearRect(0, 0, w, 18);
    ctx.fillStyle = cs.getPropertyValue('--surface-3'); ctx.fillRect(0, 7, w, 4);
    ctx.fillStyle = cs.getPropertyValue('--accent'); ctx.fillRect(0, 7, (i / N) * w, 4);
    ctx.font = '11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const ev of this.traj.events) {
      const x = Math.min(w - 5, Math.max(5, (ev.t / N) * w));
      const glyph = { gem: '💎', won: '🚪', dead: '✘', timeout: '⏱' }[ev.type];
      if (!glyph || (ev.type === 'won' && this.level.win_condition === 'gem')) continue;
      ctx.fillStyle = ev.type === 'dead' ? '#ef5b5b' : cs.getPropertyValue('--text');
      ctx.fillText(glyph, x, 9);
    }
    ctx.fillStyle = cs.getPropertyValue('--text'); ctx.fillRect(Math.round((i / N) * w) - 1, 2, 2, 14);
  }

  drawStrip(i) {
    const c = this.strip;
    if (this.opts.compact || !c.offsetParent) return;
    const w = c.clientWidth || 200, dpr = window.devicePixelRatio || 1;
    if (c.width !== w * dpr) { c.width = w * dpr; c.height = 16 * dpr; }
    const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, 16);
    const cell = 11, cells = Math.max(1, Math.floor(w / cell)), n = this.actions.length;
    const cur = Math.min(n - 1, envIndexOfEngineStep(Math.max(1, i), this.frameSkip));
    const start = Math.max(0, Math.min(n - cells, cur - Math.floor(cells / 2)));
    ctx.font = '10px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let k = 0; k < cells && start + k < n; k++) {
      const a = this.actions[start + k];
      ctx.fillStyle = ACTION_BG[a]; ctx.globalAlpha = start + k === cur ? 1 : 0.55;
      ctx.fillRect(k * cell, 1, cell - 1, 14);
      ctx.globalAlpha = 1; ctx.fillStyle = '#fff'; ctx.fillText(ACTION_SYMBOLS[a], k * cell + cell / 2, 8.5);
    }
    if (cur >= start) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.strokeRect((cur - start) * cell + 0.5, 0.75, cell - 2, 14.5); }
    c.title = `Acción en el paso de entorno ${cur + 1}: ${ACTIONS[this.actions[cur]] || '–'}`;
  }

  togglePlay() { this.clock.paused = !this.clock.paused; this.syncButtons(); }
  seekTo(t) { if (this.traj) { this.clock.t = Math.max(0, Math.min(this.traj.engine_steps, t)); this.hold = 0; this.showResult(false); this.render(); } }
  step(dir) { this.clock.paused = true; this.syncButtons(); this.seekTo(Math.floor(this.clock.t) + dir * this.frameSkip); }
  nav(dir) { if (this.opts.onNav) this.opts.onNav(dir, this); }

  key(e) {
    if (e.target.closest('select, input')) return;
    const k = e.key;
    if (k === ',') this.step(-1);
    else if (k === '.') this.step(1);
    else if (k === 'n') this.nav(1);
    else if (k === 'p') this.nav(-1);
    else if (k === 'l') { this.opts.loop = !this.opts.loop; this.syncButtons(); }
    else if (k === 'c') { const order = ['auto', 'full', 'follow']; this.camSel.value = order[(order.indexOf(this.camSel.value) + 1) % 3]; this.setCamera(this.camSel.value); }
    else if (k === 'Enter') this.togglePlay();
    else if (k === 'ArrowLeft') this.step(-1);
    else if (k === 'ArrowRight') this.step(1);
    else return;
    e.preventDefault(); e.stopPropagation();
  }
}

function ghostLabel(m) {
  if (m.run === '_reference') return 'referencia';
  const s = m.source || {};
  const at = s.kind === 'checkpoint' ? `@${s.env_steps != null ? (s.env_steps / 1e6).toFixed(2).replace('.', ',') + 'M' : 'u' + s.update}` : s.kind;
  return `${m.run} ${replayNumber(m.n)} · ${m.policy} ${at}`;
}
