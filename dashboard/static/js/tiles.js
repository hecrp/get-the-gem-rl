// Wall tile types. Each: {label, fields, create(body, props, tile) -> {destroy, title?}}.
import { getJSON, getText, send, events } from './api.js';
import { ReplayPlayer } from './player.js';
import { Clock } from './clock.js';
import { LineChart, BarChart, Histogram, seriesColor, ema, sequential, markerSvg } from './charts.js';
import { staticLayer, THEME } from './renderer.js';
import { state, on, resolveRun, resolveRuns, resolveJob, refreshJobs } from './state.js';
import { fmtSteps, fmtProp, fmtPct, fmtNum, fmtInt, fmtDuration, escapeHtml, outcomeLabel, replayNumber } from './format.js';
import { newcombe } from './stats.js';
import { ACTIONS } from './rle.js';

const enc = encodeURIComponent;
const COLS = ['success', 'return', 'gem_rate', 'entropy', 'sps', 'eval_train', 'eval_test', 'ep_len_mean',
  'time_left_frac', 'approx_kl', 'v_loss', 'pg_loss'];
const COL_LABEL = { success: 'éxito (entrenamiento)', return: 'return medio', gem_rate: 'tasa de gema', entropy: 'entropía',
  sps: 'pasos/s', eval_train: 'eval · niveles de entrenamiento', eval_test: 'eval · niveles no vistos',
  ep_len_mean: 'longitud de episodio', time_left_frac: 'fracción de tiempo restante', approx_kl: 'KL aprox.',
  v_loss: 'pérdida de valor', pg_loss: 'pérdida de política' };
const PCT_COLS = new Set(['success', 'gem_rate', 'eval_train', 'eval_test', 'time_left_frac']);

/** Distinguishing parts of run names (common prefix/suffix removed). */
export function shortNames(names) {
  if (names.length < 2) return names.map((n) => (n.length > 16 ? n.slice(0, 16) + '…' : n));
  let pre = 0, suf = 0;
  const min = Math.min(...names.map((n) => n.length));
  while (pre < min && names.every((n) => n[pre] === names[0][pre])) pre++;
  while (suf < min - pre && names.every((n) => n[n.length - 1 - suf] === names[0][names[0].length - 1 - suf])) suf++;
  return names.map((n) => n.slice(pre, n.length - suf).replace(/^[_.-]+|[_.-]+$/g, '') || n);
}

export function empty(body, html) { body.innerHTML = `<div class="empty"><div>${html}</div></div>`; }
const noRuns = (body) => empty(body, `No hay runs todavía.<br>Entrena uno desde <a href="#/operate">Operar</a> o ejecuta <code>python -m rl.ppo --levels 1-6 --total-steps 200000</code>, o arranca con <code>python -m dashboard --demo</code>.`);
const errorBox = (body, e) => empty(body, `<span class="error">✘ ${escapeHtml(e.message || e)}</span>`);

function runOptions() { return ['$latest', ...state.runs.map((r) => r.name)]; }

async function firstSweep(run) {
  const r = await getJSON(`/api/sweeps/${enc(run)}`);
  return r.sweeps[0]?.tag || null;
}
const noSweep = (body, run) => empty(body, `El run <b>${escapeHtml(run)}</b> no tiene barridos.<br>Lanza uno en <a href="#/operate">Operar → Barrer</a> o con <code>python -m rl.rollout --run ${escapeHtml(run)} --mode sweep --tag test1k --preset tiny --start-seed 1000000 --num 1000</code>.`);

// -- replay -------------------------------------------------------------------
const replay = {
  label: 'Replay',
  fields: [{ key: 'id', label: 'replay (run/n)', type: 'text' }, { key: 'camera', type: 'select', options: ['auto', 'full', 'follow'] },
    { key: 'speed', type: 'select', options: ['0.25', '0.5', '1', '2', '4', '8', '16'] }, { key: 'ghosts', label: 'fantasmas (ids, coma)', type: 'list' },
    { key: 'loop', type: 'bool' }, { key: 'trail', label: 'estela', type: 'bool' }],
  create(body, props) {
    if (!props.id) { empty(body, 'Elige un replay: escribe su id en ⚙ (p. ej. <code>_reference/level_007</code>) o usa <a href="#/replays">Replays → Enviar al muro</a>.'); return {}; }
    const p = new ReplayPlayer(body, { camera: props.camera || 'auto', speed: parseFloat(props.speed || 1), loop: props.loop !== false,
      trail: !!props.trail, ghosts: props.ghosts || [] });
    p.load(props.id);
    return { destroy: () => p.destroy(), title: props.id };
  },
};

// -- replay-feed ------------------------------------------------------------------
function feedQuery(props, extra = {}) {
  const q = new URLSearchParams();
  const run = props.run ? resolveRun(props.run) : '';
  if (run) q.set('run', run);
  for (const k of ['split', 'policy', 'outcome', 'source', 'level', 'tag']) if (props[k]) q.set(k, props[k]);
  q.set('order', props.order || 'latest');
  q.set('seed', String(props.seed ?? 0));
  for (const [k, v] of Object.entries(extra)) q.set(k, v);
  return q;
}
const FILTER_FIELDS = [{ key: 'run', type: 'run', allowEmpty: true }, { key: 'split', type: 'select', options: ['', 'train', 'test', 'builtin', 'other'] },
  { key: 'policy', type: 'select', options: ['', 'greedy', 'sampled', 'reference', 'scripted', 'random'] },
  { key: 'outcome', type: 'select', options: ['', 'won', 'hazard', 'fall', 'timeout'] },
  { key: 'source', type: 'select', options: ['', 'checkpoint', 'final', 'reference', 'scripted', 'random'] }];

const replayFeed = {
  label: 'Cola de replays',
  fields: [...FILTER_FIELDS, { key: 'order', type: 'select', options: ['latest', 'random', 'number'] },
    { key: 'count', label: 'reproductores (1–16)', type: 'number', min: 1, max: 16 },
    { key: 'follow', label: 'seguir los últimos (en vivo)', type: 'bool' }, { key: 'camera', type: 'select', options: ['auto', 'full', 'follow'] },
    { key: 'seed', type: 'number' }],
  create(body, props) {
    const count = Math.max(1, Math.min(16, props.count || 1));
    const cols = Math.ceil(Math.sqrt(count));
    body.innerHTML = `<div class="feed" style="display:grid;height:100%;gap:2px;grid-template-columns:repeat(${cols},1fr);grid-template-rows:repeat(${Math.ceil(count / cols)},1fr)"></div>`;
    const grid = body.firstChild;
    let queue = [], pos = 0, players = [], destroyed = false, total = 0;
    const history = [];
    const next = (pl, dir = 1) => {
      if (!queue.length) return;
      if (dir < 0 && pl._hist?.length > 1) { pl._hist.pop(); pl.load(pl._hist[pl._hist.length - 1]); return; }
      const id = queue[pos % queue.length]; pos++;
      (pl._hist ||= []).push(id);
      pl.load(id);
    };
    const start = async () => {
      try {
        const r = await getJSON('/api/replays?' + feedQuery(props, { limit: 200 }));
        if (destroyed) return;
        total = r.total;
        queue = r.items.map((x) => x.id);
        // Several feeds with the same filter start at different places (seed).
        pos = props.order === 'random' ? 0 : (props.seed ?? 0) * count % Math.max(1, queue.length);
        if (!queue.length) {
          empty(body, `No hay replays que coincidan.<br>Graba algunos: <code>python -m rl.rollout --run ${escapeHtml(resolveRun(props.run || '$latest') || 'RUN')} --levels 1-6</code> o desde <a href="#/operate">Operar</a>.${props.follow ? '<br>Esperando replays nuevos en vivo…' : ''}`);
          return;
        }
        for (let k = 0; k < count; k++) {
          const cell = document.createElement('div'); cell.style.minHeight = '0';
          grid.appendChild(cell);
          const pl = new ReplayPlayer(cell, { camera: props.camera || 'auto', loop: false, compact: count > 2,
            onEnd: (p) => next(p), onNav: (dir, p) => next(p, dir) });
          players.push(pl); next(pl);
        }
      } catch (e) { errorBox(body, e); }
    };
    start();
    const off = events.on('replay.added', (e) => {
      if (!props.follow || destroyed) return;
      const run = props.run ? resolveRun(props.run) : null;
      if (run && e.run !== run) return;
      if (props.split && e.split !== props.split) return;
      if (props.policy && !String(e.policy).startsWith(props.policy)) return;
      if (props.source && e.source?.kind !== props.source) return;
      if (props.outcome && ![e.outcome?.result, e.outcome?.end_reason].includes(props.outcome)) return;
      history.unshift(e.id);
      if (!players.length) { body.innerHTML = ''; body.appendChild(grid); grid.innerHTML = ''; queue = [e.id]; pos = 0; start(); return; }
      queue.splice(pos % Math.max(1, queue.length), 0, e.id);
      const idle = players.find((p) => p.clock.t >= (p.traj?.engine_steps || 0)) || players[0];
      next(idle);
    });
    return { destroy() { destroyed = true; off(); players.forEach((p) => p.destroy()); }, title: `${props.run ? resolveRun(props.run) : 'todos'} · ${props.order || 'latest'}${props.follow ? ' · en vivo' : ''}` };
  },
};

// -- replay-grid (sync) --------------------------------------------------------------
const replayGrid = {
  label: 'Rejilla de replays',
  fields: [...FILTER_FIELDS, { key: 'count', type: 'number', min: 1, max: 16 }, { key: 'sync', label: 'sincronizados', type: 'bool' },
    { key: 'order', type: 'select', options: ['number', 'latest', 'random'] }],
  create(body, props) {
    const count = Math.max(1, Math.min(16, props.count || 4));
    const cols = Math.ceil(Math.sqrt(count));
    let players = [], destroyed = false, clock = props.sync !== false ? new Clock() : null, offTick = null;
    (async () => {
      try {
        const r = await getJSON('/api/replays?' + feedQuery({ order: 'number', ...props }, { limit: count }));
        if (destroyed) return;
        if (!r.items.length) return empty(body, 'No hay replays que coincidan con los filtros (⚙).');
        body.innerHTML = `<div style="display:grid;height:100%;gap:2px;grid-template-columns:repeat(${cols},1fr)"></div>`;
        for (const it of r.items) {
          const cell = document.createElement('div'); cell.style.minHeight = '0'; body.firstChild.appendChild(cell);
          const pl = new ReplayPlayer(cell, { clock: clock || undefined, compact: true, loop: true });
          pl.load(it.id); players.push(pl);
        }
        if (clock) {
          const { ticker } = await import('./clock.js');
          let hold = 0;
          offTick = ticker.add((dt, gs) => {
            const N = Math.max(...players.map((p) => p.traj?.engine_steps || 0));
            if (!N) return;
            if (clock.t >= N) { hold += dt; if (hold > 1000) { clock.t = 0; hold = 0; } } else clock.tick(dt, gs);
          });
        }
      } catch (e) { errorBox(body, e); }
    })();
    return { destroy() { destroyed = true; offTick?.(); players.forEach((p) => p.destroy()); }, title: `${props.run ? resolveRun(props.run) : 'todos'} · ${count}${clock ? ' · sync' : ''}` };
  },
};

// -- ghosts ---------------------------------------------------------------------------
const ghosts = {
  label: 'Fantasmas superpuestos',
  fields: [{ key: 'level', label: 'nivel (builtin:N, pcg:preset:seed o hash)', type: 'text' }, { key: 'run', type: 'run', allowEmpty: true },
    { key: 'ids', label: 'ids explícitos (coma)', type: 'list' }, { key: 'reference', label: 'incluir referencia', type: 'bool' },
    { key: 'count', label: 'máx. fantasmas', type: 'number', min: 1, max: 7 }],
  create(body, props) {
    let pl = null;
    (async () => {
      try {
        let ids = props.ids || [];
        const level = props.level || 'builtin:1';
        if (!ids.length) {
          const q = new URLSearchParams({ level, limit: '200', order: 'number' });
          if (props.run) q.set('run', resolveRun(props.run));
          const r = await getJSON('/api/replays?' + q);
          const bySource = new Map();
          for (const it of r.items) {
            if (it.run === '_reference') continue;
            const key = `${it.run}|${it.source?.update ?? it.source?.kind}|${it.policy}`;
            if (!bySource.has(key)) bySource.set(key, it.id);
          }
          ids = [...bySource.values()].slice(0, props.count || 4);
          const m = /^builtin:(\d)$/.exec(level);
          if (props.reference !== false && m) ids.push(`_reference/level_${m[1].padStart(3, '0')}`);
        }
        if (!ids.length) return empty(body, `No hay replays en <b>${escapeHtml(level)}</b>. Elige otro nivel en ⚙.`);
        pl = new ReplayPlayer(body, { ghosts: ids.slice(1), trail: true, loop: true });
        pl.load(ids[0]);
      } catch (e) { errorBox(body, e); }
    })();
    return { destroy: () => pl?.destroy(), title: props.level || 'builtin:1' };
  },
};

// -- curves -----------------------------------------------------------------------------
const curves = {
  label: 'Curvas',
  fields: [{ key: 'runs', type: 'runs' }, { key: 'groups', label: 'grupos (bandas)', type: 'groups' }, { key: 'cols', label: 'métricas', type: 'multi', options: COLS },
    { key: 'x', type: 'select', options: ['env_steps', 't', 'update'] }, { key: 'log', label: 'escala log', type: 'bool' },
    { key: 'smooth', label: 'suavizado EMA (0–0,99)', type: 'number', min: 0, max: 0.99, step: 0.05 }],
  create(body, props) {
    const cols = (props.cols && props.cols.length ? props.cols : ['success']).slice(0, 4);
    body.innerHTML = `<div style="display:grid;height:100%;grid-template-rows:repeat(${cols.length},1fr)">${cols.map(() => '<div style="min-height:0"></div>').join('')}</div>`;
    const charts = cols.map((c, i) => new LineChart(body.firstChild.children[i], { yFmt: PCT_COLS.has(c) ? (v) => fmtPct(v) : undefined,
      tipFmt: PCT_COLS.has(c) ? (v) => fmtPct(v, 1) : undefined, xFmt: props.x === 't' ? fmtDuration : props.x === 'update' ? (v) => fmtInt(v) : fmtSteps,
      logY: !!props.log, legend: true }));
    let destroyed = false, timer = null;
    const load = async () => {
      const runs = resolveRuns(props.runs?.length ? props.runs : (props.groups?.length ? [] : ['$latest']));
      if (!runs.length && !props.groups?.length) return noRuns(body);
      try {
        const groups = props.groups?.length ? (await getJSON(`/api/groups?cols=${cols.join(',')}`)).groups.filter((g) => props.groups.includes(g.group)) : [];
        // Fewer than 3 seeds: no band, the group's runs are drawn individually.
        for (const g of groups) if (Object.values(g.cols).some((b) => !b.band)) for (const r of g.runs) if (!runs.includes(r)) runs.push(r);
        const data = await Promise.all(runs.map((r) => getJSON(`/api/runs/${enc(r)}/metrics?cols=${cols.join(',')}&x=${props.x || 'env_steps'}`).catch(() => null)));
        if (destroyed) return;
        const alpha = 1 - Math.min(0.99, Math.max(0, props.smooth ?? 0.3));
        cols.forEach((c, ci) => {
          const series = [];
          let warn = [], demo = false;
          data.forEach((d, i) => {
            if (!d) return;
            demo ||= d.demo;
            const s = d.series[c];
            if (!s || !s.x.length) return;
            const sm = alpha < 1 && !c.startsWith('eval_');
            if (sm) series.push({ label: '', x: s.x, y: s.y, idx: i, faint: true });
            series.push({ label: runs[i], x: s.x, y: sm ? ema(s.y, alpha) : s.y, idx: i, points: c.startsWith('eval_') });
          });
          groups.forEach((g, gi) => {
            const b = g.cols[c];
            const idx = runs.length + gi;
            if (b?.band) series.push({ label: `${g.group} · media de ${b.runs} seeds (IC 95 %)`, x: b.x, y: b.band.mean, idx, band: { x: b.x, lo: b.band.lo, hi: b.band.hi } });
            else if (b) { warn.push(`${g.group}: ${b.warning || 'sin datos'}`); }
            demo ||= g.demo;
          });
          charts[ci].set(series, { aria: COL_LABEL[c] || c });
          const n = data.filter(Boolean).map((d) => d.series[c]?.n || 0);
          charts[ci].note(`${COL_LABEL[c] || c}${n.length ? ` · puntos: ${n.join(', ')}` : ''}${warn.length ? ' · ⚠ ' + warn.join('; ') : ''}${demo ? ' · DATOS DE DEMO' : ''}`);
          if (!series.length) charts[ci].note(`${COL_LABEL[c] || c}: sin datos (los runs antiguos no tienen ${c}).`);
        });
      } catch (e) { errorBox(body, e); }
    };
    load();
    const off = events.on('run.updated', (e) => {
      if (resolveRuns(props.runs || ['$latest']).includes(e.name) || props.groups?.length) { clearTimeout(timer); timer = setTimeout(load, 500); }
    });
    return { destroy() { destroyed = true; off(); charts.forEach((c) => c.destroy()); }, title: cols.map((c) => COL_LABEL[c] || c).join(' · ') };
  },
};

// -- train-vs-test ----------------------------------------------------------------------
const trainVsTest = {
  label: 'Entrenamiento vs no vistos',
  fields: [{ key: 'runs', type: 'runs' }, { key: 'policy', type: 'select', options: ['sampled', 'greedy'] }],
  create(body, props) {
    body.innerHTML = '<div style="display:grid;height:100%;grid-template-rows:1fr 1fr"><div style="min-height:0"></div><div style="min-height:0"></div></div>';
    const bars = new BarChart(body.firstChild.children[0], { rotate: true });
    const line = new LineChart(body.firstChild.children[1], { yFmt: (v) => fmtPct(v), tipFmt: (v) => fmtPct(v, 1), yMin: 0, yMax: 1, legend: true });
    const pol = props.policy || 'sampled';
    let destroyed = false;
    (async () => {
      const runs = resolveRuns(props.runs?.length ? props.runs : ['$all']);
      if (!runs.length) return noRuns(body);
      try {
        const det = await Promise.all(runs.map((r) => getJSON(`/api/runs/${enc(r)}`)));
        const met = await Promise.all(runs.map((r) => getJSON(`/api/runs/${enc(r)}/metrics?cols=eval_train,eval_test`)));
        if (destroyed) return;
        const b = [], gaps = [], series = [];
        let demo = false;
        const short = shortNames(det.map((d) => d.name));
        det.forEach((d, i) => {
          demo ||= d.demo;
          const ev = d.eval?.[pol];
          if (ev?.train && ev?.test) {
            b.push({ label: `${short[i]} train`, ...ev.train, idx: 0, extra: d.name }, { label: `${short[i]} test`, ...ev.test, idx: 1, extra: d.name });
            const g = newcombe(ev.train.k, ev.train.n, ev.test.k, ev.test.n);
            gaps.push(`${d.name}: hueco ${fmtNum(100 * g.d, 0)} pp [${fmtNum(100 * g.lo, 0)}–${fmtNum(100 * g.hi, 0)}]`);
          }
          const m = met[i].series;
          if (m.eval_train?.x.length) series.push({ label: `${d.name} · train`, x: m.eval_train.x, y: m.eval_train.y, idx: 2 * i % 8, points: true });
          if (m.eval_test?.x.length) series.push({ label: `${d.name} · test`, x: m.eval_test.x, y: m.eval_test.y, idx: (2 * i + 1) % 8, dash: [5, 3], points: true });
        });
        if (!b.length && !series.length) return empty(body, 'Estos runs no tienen evaluación train/test (sólo los runs con <code>--preset</code> evalúan en niveles no vistos).');
        bars.set(b, { legendHtml: `<span>${markerSvg(0, seriesColor(0))}entrenamiento</span><span>${markerSvg(1, seriesColor(1))}no vistos (seeds ≥ 1 000 000)</span>` });
        bars.note(`eval.json final, política ${pol}, IC de Wilson 95 % · ${gaps.join(' · ')}${demo ? ' · DATOS DE DEMO' : ''}`);
        line.set(series, { aria: 'evaluación durante el entrenamiento' });
        line.note('evaluaciones periódicas (--eval-every), 64 episodios cada una por defecto');
      } catch (e) { errorBox(body, e); }
    })();
    return { destroy() { destroyed = true; bars.destroy(); line.destroy(); }, title: `generalización · ${pol}` };
  },
};

// -- run-card ---------------------------------------------------------------------------
export function runCardHtml(r) {
  const st = r.status || {};
  const pct = st.updates_total ? st.update / st.updates_total : r.state === 'finished' ? 1 : 0;
  const left = st.sps && st.total_steps ? (st.total_steps - st.env_steps) / st.sps : null;
  const ev = r.eval || {};
  const evTxt = ev.sampled ? Object.entries(ev.sampled).filter(([, v]) => v && v.n).slice(0, 3).map(([k, v]) => `${escapeHtml(k)}: ${fmtProp(v)}`).join('<br>') : (ev.partial ? 'eval parcial (parado)' : '–');
  return `
    <div style="display:flex;flex-direction:column;gap:6px;padding:8px">
      <div><span class="state ${r.state}">${r.state}</span> ${r.reason ? `<span class="muted small">(${escapeHtml(r.reason)})</span>` : ''} ${r.demo ? '<span class="badge demo">DEMO</span>' : ''}</div>
      <div class="progress" role="progressbar" aria-valuenow="${Math.round(100 * pct)}" aria-valuemin="0" aria-valuemax="100"><div style="width:${100 * pct}%"></div></div>
      <dl class="kv">
        <dt>pasos</dt><dd>${fmtSteps(st.env_steps ?? r.last?.env_steps)} / ${fmtSteps(st.total_steps ?? r.config?.total_steps)}</dd>
        <dt>SPS</dt><dd>${fmtInt(st.sps ?? r.last?.sps)}</dd>
        <dt>ETA</dt><dd>${r.state === 'running' ? fmtDuration(left) : '–'}</dd>
        <dt>config</dt><dd>${escapeHtml(r.config?.preset ? `${r.config.preset} n=${r.config.num_levels}` : `niveles ${r.config?.levels}`)} · shaping ${r.config?.distance_shaping ?? 0} · seed ${r.config?.seed}</dd>
        <dt>último</dt><dd>éxito ${fmtPct(r.last?.success)} · return ${fmtNum(r.last?.return)}</dd>
        <dt>eval</dt><dd class="small">${evTxt}</dd>
        <dt>replays</dt><dd>${r.replays} · barridos ${r.sweeps}${r.only_final_policy ? ' · <span class="muted">sólo política final</span>' : ''}</dd>
      </dl>
    </div>`;
}

const runCard = {
  label: 'Estado de run',
  fields: [{ key: 'run', type: 'run' }],
  create(body, props, tile) {
    let destroyed = false;
    const draw = () => {
      if (destroyed) return;
      const name = resolveRun(props.run || '$latest');
      const r = state.runs.find((x) => x.name === name);
      if (!r) return noRuns(body);
      const job = state.jobs.find((j) => j.run === name && j.kind === 'train' && j.state === 'running');
      const can = state.jobsEnabled;
      body.innerHTML = runCardHtml(r) + `
        <div style="display:flex;gap:6px;flex-wrap:wrap;padding:0 8px 8px">
          <button class="btn danger" data-a="stop" ${job && can ? '' : 'disabled'} title="${job ? 'SIGTERM: termina el update, guarda checkpoint y model.pt' : 'sin job activo para este run'}">■ Parar</button>
          <button class="btn" data-a="resume" ${can && r.has_ckpts && ['stopped', 'dead', 'failed'].includes(r.state) ? '' : 'disabled'} title="--resume desde el último checkpoint">▶ Reanudar</button>
          <button class="btn" data-a="record" ${can && r.state !== 'running' ? '' : 'disabled'} title="rl.rollout: 8 replays de la política final">● Grabar replays</button>
          <a class="btn" href="#/runs/${enc(name)}">Detalle</a>
        </div>${can ? '' : `<div class="muted small" style="padding:0 8px 8px">Operación deshabilitada: ${escapeHtml(state.jobsNote)}</div>`}`;
      tile.setTitle(name);
      body.querySelector('[data-a=stop]').onclick = async () => { try { await send('DELETE', `/api/jobs/${job.id}`); tile.toast('Parada solicitada (SIGTERM)'); } catch (e) { tile.toast(e.message); } };
      body.querySelector('[data-a=resume]').onclick = async () => { try { await send('POST', '/api/jobs', { kind: 'train', args: { name, resume: 'latest' } }); tile.toast('Reanudando ' + name); refreshJobs(); } catch (e) { tile.toast(e.message); } };
      body.querySelector('[data-a=record]').onclick = async () => {
        const args = { run: name, num: 8, policy: 'greedy' };
        if (r.config?.preset) Object.assign(args, { preset: r.config.preset, start_seed: 1000000 }); else args.levels = r.config?.levels || '1-6';
        try { await send('POST', '/api/jobs', { kind: 'rollout', args }); tile.toast('Grabación encolada'); } catch (e) { tile.toast(e.message); }
      };
    };
    draw();
    const offs = [on('runs', draw), on('jobs', draw)];
    return { destroy() { destroyed = true; offs.forEach((f) => f()); } };
  },
};

// -- runs-table -------------------------------------------------------------------------
export function runsTable(body, opts = {}) {
  let sortKey = 'name', dir = 1, filter = opts.filter || '';
  const draw = () => {
    const rows = state.runs.filter((r) => !filter || r.name.includes(filter) || r.group.includes(filter));
    if (!state.runs.length) return noRuns(body);
    const key = (r) => ({ name: r.name, group: r.group, state: r.state, steps: r.status?.env_steps ?? r.last?.env_steps ?? 0,
      success: r.last?.success ?? -1, replays: r.replays, test: r.eval?.sampled?.test?.p ?? -1 })[sortKey];
    rows.sort((a, b) => (key(a) > key(b) ? dir : key(a) < key(b) ? -dir : 0));
    const th = (k, t, cls = '') => `<th class="${cls}" data-k="${k}" aria-sort="${sortKey === k ? (dir > 0 ? 'ascending' : 'descending') : 'none'}">${t}${sortKey === k ? (dir > 0 ? ' ▲' : ' ▼') : ''}</th>`;
    body.innerHTML = `<div style="padding:4px"><input type="text" placeholder="filtrar…" value="${escapeHtml(filter)}" aria-label="Filtrar runs" style="width:100%"></div>
      <table class="data"><thead><tr>${th('name', 'run')}${th('group', 'grupo')}${th('state', 'estado')}${th('steps', 'pasos', 'num')}${th('success', 'éxito', 'num')}<th>curva</th>${th('test', 'eval test', 'num')}${th('replays', 'replays', 'num')}</tr></thead>
      <tbody>${rows.map((r) => `<tr><td><a href="#/runs/${enc(r.name)}">${escapeHtml(r.name)}</a>${r.demo ? ' <span class="badge demo">DEMO</span>' : ''}</td>
        <td class="muted small">${escapeHtml(r.group)}</td><td><span class="state ${r.state}">${r.state}</span></td>
        <td class="num">${fmtSteps(r.status?.env_steps ?? r.last?.env_steps)}</td><td class="num">${fmtPct(r.last?.success)}</td>
        <td><canvas class="spark" data-run="${escapeHtml(r.name)}" width="80" height="18" aria-label="curva de éxito de ${escapeHtml(r.name)}"></canvas></td>
        <td class="num" title="${escapeHtml(fmtProp(r.eval?.sampled?.test))}">${r.eval?.sampled?.test ? fmtPct(r.eval.sampled.test.p) + ` <span class="muted small">n=${r.eval.sampled.test.n}</span>` : '–'}</td>
        <td class="num">${r.replays}</td></tr>`).join('')}</tbody></table>`;
    const input = body.querySelector('input');
    input.oninput = () => { filter = input.value; const pos = input.selectionStart; draw(); const i2 = body.querySelector('input'); i2.focus(); i2.setSelectionRange(pos, pos); };
    body.querySelectorAll('th[data-k]').forEach((t) => { t.onclick = () => { dir = sortKey === t.dataset.k ? -dir : 1; sortKey = t.dataset.k; draw(); }; });
    body.querySelectorAll('canvas.spark').forEach(async (c) => {
      try {
        const m = await getJSON(`/api/runs/${enc(c.dataset.run)}/metrics?cols=success&max_points=60`);
        const s = m.series.success; if (!s?.y.length) return;
        const ctx = c.getContext('2d'), w = c.width, h = c.height;
        ctx.strokeStyle = seriesColor(0); ctx.lineWidth = 1.5; ctx.beginPath();
        s.y.forEach((y, i) => { const X = (i / Math.max(1, s.y.length - 1)) * (w - 2) + 1, Y = h - 2 - y * (h - 4); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        ctx.stroke();
      } catch { /* sparkline is optional */ }
    });
  };
  draw();
  return on('runs', draw);
}
const runsTableTile = { label: 'Tabla de runs', fields: [{ key: 'filter', type: 'text' }], create(body, props) { const off = runsTable(body, props); return { destroy: off }; } };

// -- level-success ----------------------------------------------------------------------
const levelSuccess = {
  label: 'Éxito por nivel',
  fields: [{ key: 'runs', type: 'runs' }, { key: 'policy', type: 'select', options: ['sampled', 'greedy'] }],
  create(body, props) {
    const chart = new BarChart(body, { rotate: true });
    (async () => {
      const runs = resolveRuns(props.runs?.length ? props.runs : ['$all']);
      if (!runs.length) return noRuns(body);
      const pol = props.policy || 'sampled';
      const det = await Promise.all(runs.map((r) => getJSON(`/api/runs/${enc(r)}`)));
      const bars = [];
      det.forEach((d, i) => {
        const ev = d.eval?.[pol];
        if (!ev) return;
        for (const [k, v] of Object.entries(ev)) if (v?.n) bars.push({ label: `${/^\d$/.test(k) ? 'nivel ' + k : k}`, ...v, idx: i, extra: d.name });
      });
      if (!bars.length) return empty(body, 'Sin evaluaciones por nivel (eval.json).');
      chart.set(bars, { legendHtml: det.map((d, i) => `<span>${markerSvg(i, seriesColor(i))}${escapeHtml(d.name)}</span>`).join('') });
      chart.note(`eval.json, política ${pol}; barras = proporción de victorias con IC de Wilson 95 %`);
    })().catch((e) => errorBox(body, e));
    return { destroy: () => chart.destroy(), title: 'éxito por nivel' };
  },
};

// -- sweep-based tiles ---------------------------------------------------------------------
function sweepTile(label, extraFields, render) {
  return {
    label,
    fields: [{ key: 'run', type: 'run' }, { key: 'tag', label: 'barrido (tag)', type: 'text' }, ...extraFields],
    create(body, props, tile) {
      let cleanup = () => {};
      (async () => {
        const run = resolveRun(props.run || '$latest');
        if (!run) return noRuns(body);
        const tag = props.tag || await firstSweep(run);
        if (!tag) return noSweep(body, run);
        tile.setTitle(`${run} · ${tag}`);
        cleanup = (await render(body, props, run, tag)) || cleanup;
      })().catch((e) => errorBox(body, e));
      return { destroy: () => cleanup() };
    },
  };
}
const getSummary = (run, tag) => getJSON(`/api/sweeps/${enc(run)}/${enc(tag)}/summary`);

const difficulty = sweepTile('Éxito vs dificultad', [{ key: 'x', type: 'select', options: ['min_window', 'area', 'leniency'] }], async (body, props, run, tag) => {
  const s = await getSummary(run, tag);
  const key = props.x || 'min_window';
  const chart = new BarChart(body, {});
  const bars = (s.difficulty[key] || []).map((b) => ({ ...b, label: b.label, idx: 0 }));
  if (!bars.length) { empty(body, 'Este barrido no tiene métricas de nivel (niveles integrados: sólo tamaño).'); return; }
  chart.set(bars);
  chart.note(`${{ min_window: 'ventana mínima de salto (cubos enteros)', area: 'área (cuantiles)', leniency: 'leniency (cuantiles)' }[key]} · n=${s.n} · ⚠ = menos de ${s.min_bucket} episodios${s.demo ? ' · DATOS DE DEMO' : ''}`);
  return () => chart.destroy();
});

const endReasons = sweepTile('Causas de fin', [], async (body, props, run, tag) => {
  const s = await getSummary(run, tag);
  const chart = new BarChart(body, {});
  const names = { won: '✔ victoria', hazard: '✘ pinchos', fall: '✘ vacío', timeout: '⏱ timeout', incomplete: '… incompleto' };
  const color = { won: 'var(--good)', hazard: 'var(--critical)', fall: 'var(--serious)', timeout: 'var(--warning)' };
  const cs = getComputedStyle(document.documentElement);
  chart.set(Object.entries(s.end_reasons).map(([k, v]) => ({ label: names[k] || k, ...v, color: cs.getPropertyValue(color[k]?.slice(4, -1) || '--muted') })));
  chart.note(`proporciones que suman 1 · n=${s.n} episodios · IC de Wilson 95 %${s.demo ? ' · DATOS DE DEMO' : ''}`);
  return () => chart.destroy();
});

const histogramTile = sweepTile('Histograma', [{ key: 'metric', type: 'select', options: ['ep_len', 'ep_len_won', 'ep_len_other', 'time_to_gem', 'gem_to_exit', 'actions'] }], async (body, props, run, tag) => {
  const s = await getSummary(run, tag);
  const metric = props.metric || 'ep_len';
  if (metric === 'actions') {
    const chart = new BarChart(body, { pct: false, max: Math.max(...s.action_hist) || 1 });
    const total = s.action_hist.reduce((a, b) => a + b, 0);
    chart.set(s.action_hist.map((c, i) => ({ label: ACTIONS[i], p: c, n: c, idx: i, extra: fmtPct(c / total, 1) })));
    chart.note(`acciones por paso de entorno · total ${fmtInt(total)}`);
    return () => chart.destroy();
  }
  const h = new Histogram(body, { idx: 0 });
  h.set(s.histograms[metric]);
  const sum = metric.startsWith('ep_len') ? s.ep_len[metric === 'ep_len_won' ? 'won' : 'other'] : s[metric];
  h.note(`${{ ep_len: 'longitud (pasos de motor)', ep_len_won: 'longitud de victorias', ep_len_other: 'longitud de fracasos', time_to_gem: 'pasos hasta la gema', gem_to_exit: 'pasos gema → puerta' }[metric]} · n=${s.histograms[metric].n}${sum?.median != null ? ` · mediana ${fmtNum(sum.median, 0)} [Q1 ${fmtNum(sum.q1, 0)}, Q3 ${fmtNum(sum.q3, 0)}]` : ''}${s.demo ? ' · DATOS DE DEMO' : ''}`);
  return () => h.destroy();
});

const heatmap = sweepTile('Mapa de calor', [{ key: 'level', label: 'nivel (hash; vacío = el más jugado)', type: 'text' },
  { key: 'layer', type: 'select', options: ['visits', 'deaths'] }], async (body, props, run, tag) => {
  const layer = props.layer || 'visits';
  const q = new URLSearchParams({ layer }); if (props.level) q.set('level', props.level);
  const hm = await getJSON(`/api/sweeps/${enc(run)}/${enc(tag)}/heatmap?${q}`);
  body.innerHTML = '<div class="stage" style="height:calc(100% - 22px)"><canvas class="game"></canvas></div><div class="note small muted" style="padding:2px 6px"></div>';
  const c = body.querySelector('canvas'), lv = hm.level;
  c.width = lv.width; c.height = lv.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(staticLayer(lv), 0, 0);
  ctx.fillStyle = 'rgba(0,0,0,.45)'; ctx.fillRect(0, 0, c.width, c.height);
  const cell = hm.cell;
  const paint = (grid, max, tint) => {
    grid.forEach((v, i) => {
      if (!v) return;
      const x = (i % hm.cols) * cell, y = Math.floor(i / hm.cols) * cell;
      ctx.fillStyle = tint || sequential(Math.sqrt(v / max));
      ctx.globalAlpha = tint ? Math.min(1, 0.35 + v / max) : 0.85;
      ctx.fillRect(x, y, cell, cell);
    });
    ctx.globalAlpha = 1;
  };
  if (layer === 'visits') paint(hm.grid, hm.max);
  else { paint(hm.grids.hazard, hm.max, '#ef5b5b'); paint(hm.grids.fall, hm.max, '#fab219'); }
  const fit = () => { const st = body.querySelector('.stage'); const s = Math.min(st.clientWidth / c.width, st.clientHeight / c.height); const k = s >= 1 ? Math.floor(s) : s; c.style.width = c.width * k + 'px'; c.style.height = c.height * k + 'px'; };
  const ro = new ResizeObserver(fit); ro.observe(body);
  body.querySelector('.note').textContent = layer === 'visits'
    ? `visitas: pasos de motor por celda de 8 px (centro de la hitbox), media por episodio · ${hm.used} episodios re-simulados de ${hm.episodes} · escala √ · ${lv.name}`
    : `muertes por celda: rojo = pinchos, amarillo = vacío · ${hm.episodes} episodios · ${lv.name}`;
  return () => ro.disconnect();
});

// -- level-viewer ---------------------------------------------------------------------------
const levelViewer = {
  label: 'Visor de nivel',
  fields: [{ key: 'ref', label: 'builtin:N o pcg:preset:seed', type: 'text' }],
  create(body, props, tile) {
    const ref = props.ref || 'builtin:1';
    let pl = null;
    const m = /^builtin:(\d)$/.exec(ref);
    if (m) {
      body.innerHTML = '<div style="height:100%"></div>';
      pl = new ReplayPlayer(body.firstChild, { loop: true, trail: true });
      pl.load(`_reference/level_${m[1].padStart(3, '0')}`);
      return { destroy: () => pl.destroy(), title: `nivel ${m[1]} · solución de referencia` };
    }
    const p = /^pcg:(\w+):(\d+)$/.exec(ref);
    if (!p) { empty(body, 'Referencia no válida: usa <code>builtin:7</code> o <code>pcg:tiny:1000007</code>.'); return {}; }
    getJSON(`/api/levels/by-ref?ref=${enc(JSON.stringify({ kind: 'pcg', preset: p[1], seed: +p[2] }))}`).then((lv) => {
      body.innerHTML = '<div class="stage" style="height:70%"><canvas class="game"></canvas></div><dl class="kv" style="padding:6px"></dl>';
      const c = body.querySelector('canvas'); c.width = lv.width; c.height = lv.height;
      c.getContext('2d').drawImage(staticLayer(lv), 0, 0);
      const st = body.querySelector('.stage'); const s = Math.min(st.clientWidth / lv.width, st.clientHeight / lv.height);
      c.style.width = lv.width * (s >= 1 ? Math.floor(s) : s) + 'px';
      body.querySelector('dl').innerHTML = Object.entries(lv.metrics || {}).map(([k, v]) => `<dt>${k}</dt><dd>${fmtNum(v, 3)}</dd>`).join('');
      tile.setTitle(`${p[1]} · seed ${fmtInt(+p[2])}`);
    }).catch((e) => errorBox(body, e));
    return {};
  },
};

// -- job-log ---------------------------------------------------------------------------------
const jobLog = {
  label: 'Log de job',
  fields: [{ key: 'job', label: 'job id o $latest', type: 'text' }, { key: 'filter', type: 'text' }],
  create(body, props, tile) {
    body.innerHTML = `<div style="display:flex;flex-direction:column;height:100%"><div class="chart-controls"><input type="text" placeholder="filtrar líneas…" value="${escapeHtml(props.filter || '')}" aria-label="Filtrar log"><button class="btn" data-a="dl">Descargar</button><span class="muted small st"></span></div><pre class="log" aria-live="polite"></pre></div>`;
    const pre = body.querySelector('pre'), input = body.querySelector('input'), st = body.querySelector('.st');
    let text = '', timer = null, destroyed = false;
    const show = () => {
      const f = input.value;
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
      pre.textContent = f ? text.split('\n').filter((l) => l.includes(f)).join('\n') : text;
      if (atBottom) pre.scrollTop = pre.scrollHeight;
    };
    const poll = async () => {
      if (destroyed) return;
      const job = resolveJob(props.job || '$latest');
      if (!job) { pre.textContent = 'No hay jobs. Lanza uno desde Operar.'; timer = setTimeout(poll, 3000); return; }
      tile.setTitle(`job ${job.id} · ${job.kind} · ${job.run}`);
      st.textContent = job.state;
      try { text = await getText(`/api/jobs/${job.id}/log?tail=2000`); show(); } catch (e) { pre.textContent = e.message; }
      timer = setTimeout(poll, job.state === 'running' || job.state === 'queued' ? 1500 : 10000);
    };
    input.oninput = show;
    body.querySelector('[data-a=dl]').onclick = () => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' })); a.download = 'job.log'; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };
    poll();
    return { destroy() { destroyed = true; clearTimeout(timer); } };
  },
};

// -- note --------------------------------------------------------------------------------------
export function miniMarkdown(src) {
  const inline = (s) => escapeHtml(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\*([^*]+)\*/g, '<i>$1</i>');
  const out = []; let list = false;
  for (const line of String(src || '').split('\n')) {
    const h = /^(#{1,3})\s+(.*)/.exec(line), li = /^\s*[-*]\s+(.*)/.exec(line);
    if (li) { if (!list) { out.push('<ul>'); list = true; } out.push(`<li>${inline(li[1])}</li>`); continue; }
    if (list) { out.push('</ul>'); list = false; }
    if (h) out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push('</ul>');
  return out.join('');
}
const note = {
  label: 'Nota', fields: [{ key: 'text', type: 'textarea' }],
  create(body, props) { body.innerHTML = `<div class="note-tile">${miniMarkdown(props.text || '*Nota vacía.* Edítala con ⚙.')}</div>`; return { title: 'nota' }; },
};

const emptyTile = {
  label: 'Vacío', fields: [],
  create(body) { empty(body, 'Tile vacío: elige un tipo con ⚙.'); return {}; },
};

export const TILES = { replay, 'replay-feed': replayFeed, 'replay-grid': replayGrid, ghosts, curves, 'train-vs-test': trainVsTest,
  'run-card': runCard, 'runs-table': runsTableTile, 'level-success': levelSuccess, difficulty, 'end-reasons': endReasons,
  heatmap, histogram: histogramTile, 'level-viewer': levelViewer, 'job-log': jobLog, note, empty: emptyTile };
export { COLS, COL_LABEL, runOptions };
