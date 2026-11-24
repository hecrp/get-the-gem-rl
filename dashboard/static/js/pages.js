// Pages: Runs (+ detail), Replays (gallery), Levels (+ generator explorer), Operate.
import { getJSON, getTrajectory, getLevel, send } from './api.js';
import { TILES, runsTable, runCardHtml, empty } from './tiles.js';
import { ReplayPlayer } from './player.js';
import { frameCanvas, staticLayer } from './renderer.js';
import { Scatter } from './charts.js';
import { state, on, refreshJobs } from './state.js';
import { sendToWall } from './wall.js';
import { toast } from './ui.js';
import { replayNumber, outcomeLabel, sourceLabel, levelLabel, fmtInt, fmtNum, fmtSteps, fmtProp, escapeHtml, fmtDuration } from './format.js';

const enc = encodeURIComponent;
const tileApi = { setTitle() {}, toast };

function card(parent, title, height) {
  const c = document.createElement('section');
  c.className = 'card';
  c.innerHTML = `<h2>${title}</h2><div class="card-body" style="${height ? `height:${height}px;` : ''}position:relative"></div>`;
  parent.appendChild(c);
  return c.querySelector('.card-body');
}
function mountTile(parent, title, type, props, height = 260) {
  const body = card(parent, title, height);
  return TILES[type].create(body, props, tileApi) || {};
}

export function openPlayerDialog(id, opts = {}) {
  const d = document.createElement('dialog');
  d.style.width = 'min(1100px, 95vw)'; d.style.height = 'min(760px, 90vh)'; d.style.padding = '8px';
  d.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px"><b>${escapeHtml(id)}</b>
    <span><button class="btn" data-a="wall">Enviar al muro</button> <button class="btn" data-a="close">Cerrar (Esc)</button></span></div><div style="height:calc(100% - 36px)"></div>`;
  document.body.appendChild(d);
  const p = new ReplayPlayer(d.lastChild, { loop: true, ...opts });
  p.load(id);
  d.querySelector('[data-a=close]').onclick = () => d.close();
  d.querySelector('[data-a=wall]').onclick = () => { d.close(); sendToWall([id]); };
  d.addEventListener('close', () => { p.destroy(); d.remove(); });
  d.showModal();
  setTimeout(() => p.root.focus(), 50);
}

// -- Runs -----------------------------------------------------------------------
export async function renderRuns(main, name) {
  if (name) return renderRunDetail(main, name);
  main.innerHTML = `<div class="page-head"><h1>Runs</h1><span class="muted small">${state.runs.length} runs · ${escapeHtml(state.health.runs_dir)}</span></div><div class="grid-cards" id="rg"></div>`;
  const g = main.querySelector('#rg');
  const t = card(g, 'Runs', null); t.parentElement.style.gridColumn = '1 / -1';
  const offs = [runsTable(t)];
  const gb = card(g, 'Grupos de experimento (mismas opciones, distinta seed)', null);
  gb.parentElement.style.gridColumn = '1 / -1';
  try {
    const groups = (await getJSON('/api/groups?cols=success')).groups;
    gb.innerHTML = `<table class="data"><thead><tr><th>grupo</th><th class="num">seeds</th><th>runs</th><th>bandas</th></tr></thead><tbody>${groups.map((g2) => `<tr>
      <td>${escapeHtml(g2.group)}${g2.demo ? ' <span class="badge demo">DEMO</span>' : ''}</td><td class="num">${g2.runs.length}</td><td class="small">${g2.runs.map((r) => `<a href="#/runs/${enc(r)}">${escapeHtml(r)}</a>`).join(', ')}</td>
      <td class="small">${g2.runs.length >= 3 ? `✔ media + IC 95 % (bootstrap) · <a href="#/wall?layout=" data-group="${escapeHtml(g2.group)}">ver en el muro</a>` : `<span class="muted">⚠ ${g2.runs.length} seed(s): sin banda (hacen falta ≥ 3)</span>`}</td></tr>`).join('')}</tbody></table>
      <p class="muted small">Grupo = <code>--group</code> o, si falta, un hash de la configuración sin <code>seed</code>/<code>name</code>. Un solo run no permite conclusiones: entrena 3 seeds con el mismo <code>--group</code>.</p>`;
    gb.querySelectorAll('[data-group]').forEach((a) => {
      a.onclick = async (e) => {
        e.preventDefault();
        const { encodeLayout } = await import('./layout.js');
        location.hash = '#/wall?l=' + await encodeLayout({ v: 1, cols: 1, rows: 1, tiles: [{ type: 'curves', span: [1, 1], props: { groups: [a.dataset.group], cols: ['success', 'return'], smooth: 0.3 } }] });
      };
    });
  } catch (e) { gb.textContent = e.message; }
  return () => offs.forEach((f) => f());
}

async function renderRunDetail(main, name) {
  let d;
  try { d = await getJSON(`/api/runs/${enc(name)}`); } catch (e) { main.innerHTML = `<div class="empty error">✘ ${escapeHtml(e.message)}</div>`; return; }
  main.innerHTML = `<div class="page-head"><a href="#/runs" class="btn ghost">← Runs</a><h1>${escapeHtml(name)}</h1><span class="state ${d.state}">${d.state}</span>
    ${d.demo ? '<span class="badge demo">DATOS DE DEMO</span>' : ''}${d.only_final_policy ? '<span class="badge warn" title="Sin checkpoints intermedios: los replays son de la política final">sólo política final</span>' : ''}
    <span class="muted small">grupo ${escapeHtml(d.group)}</span></div><div class="grid-cards" id="dg"></div>`;
  const g = main.querySelector('#dg');
  const insts = [];
  const sc = card(g, 'Estado', null);
  const drawState = () => { const r = state.runs.find((x) => x.name === name); if (r) sc.innerHTML = runCardHtml(r); };
  drawState();
  const off = on('runs', drawState);
  insts.push(mountTile(g, 'Éxito y return', 'curves', { runs: [name], cols: ['success', 'return'], smooth: 0.3 }, 300));
  insts.push(mountTile(g, 'Entropía y SPS', 'curves', { runs: [name], cols: ['entropy', 'sps'], smooth: 0.3 }, 300));
  if (d.config?.preset) insts.push(mountTile(g, 'Generalización (eval.json)', 'train-vs-test', { runs: [name] }, 300));
  else insts.push(mountTile(g, 'Éxito por nivel (eval.json)', 'level-success', { runs: [name] }, 300));
  const cfg = card(g, 'Configuración completa', null);
  cfg.innerHTML = `<dl class="kv">${Object.entries(d.config_full || {}).map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd class="mono small">${escapeHtml(JSON.stringify(v))}</dd>`).join('')}</dl>`;
  const ev = card(g, 'Evaluación final', null);
  ev.innerHTML = d.eval && !d.eval.partial ? `<table class="data"><thead><tr><th>política</th><th>conjunto</th><th>éxito [IC 95 %] n</th><th class="num">return</th></tr></thead><tbody>${['sampled', 'greedy'].flatMap((p) => Object.entries(d.eval[p] || {}).map(([k, v]) => `<tr><td>${p}</td><td>${escapeHtml(k)}</td><td>${fmtProp(v)}</td><td class="num">${fmtNum(v.mean_return)}</td></tr>`)).join('')}</tbody></table><p class="muted small">Entrenado en ${fmtDuration(d.eval.train_seconds)}.</p>`
    : d.eval?.partial ? `<p>Evaluación parcial: parado en el update ${d.eval.stopped_at_update} de ${d.eval.updates_total}.</p>` : '<p class="muted">Sin eval.json todavía.</p>';
  const ck = card(g, 'Checkpoints', null);
  ck.innerHTML = d.files.ckpts.length ? `<p class="small">${d.files.ckpts.map(escapeHtml).join(' · ')}</p>` : '<p class="muted small">Sin checkpoints (el run no usó <code>--ckpt-every</code>). Los replays de este run son de la política final.</p>';
  const rp = card(g, `Replays (${d.replays})`, 280);
  if (d.replays) insts.push(TILES['replay-feed'].create(rp, { run: name, order: 'latest', count: 1, follow: true }, tileApi));
  else empty(rp, `Sin replays. <code>python -m rl.rollout --run ${escapeHtml(name)}</code> o «Grabar replays».`);
  for (const sw of d.sweeps_list || []) {
    insts.push(mountTile(g, `Barrido ${escapeHtml(sw.tag)} · causas de fin`, 'end-reasons', { run: name, tag: sw.tag }, 240));
    insts.push(mountTile(g, `Barrido ${escapeHtml(sw.tag)} · dificultad`, 'difficulty', { run: name, tag: sw.tag }, 240));
    insts.push(mountTile(g, `Barrido ${escapeHtml(sw.tag)} · longitud`, 'histogram', { run: name, tag: sw.tag }, 240));
    insts.push(mountTile(g, `Barrido ${escapeHtml(sw.tag)} · mapa de calor`, 'heatmap', { run: name, tag: sw.tag }, 300));
  }
  const jobs = state.jobs.filter((j) => j.run === name);
  if (jobs.length) insts.push(mountTile(g, 'Log del último job', 'job-log', { job: String(jobs[0].id) }, 260));
  return () => { off(); insts.forEach((i) => i.destroy?.()); };
}

// -- Replays ---------------------------------------------------------------------
export async function renderReplays(main, params) {
  const f = Object.fromEntries(params);
  const sel = new Set();
  main.innerHTML = `<div class="page-head"><h1>Replays</h1><span class="muted small" id="tot"></span><span class="spacer"></span>
      <button class="btn primary" data-a="wall" disabled>Enviar selección al muro</button></div>
    <form class="filters" aria-label="Filtros">
      ${sel_('run', ['', '_reference', ...state.runs.map((r) => r.name)], f.run)}${sel_('split', ['', 'train', 'test', 'builtin', 'other'], f.split)}
      ${sel_('policy', ['', 'greedy', 'sampled', 'reference', 'scripted', 'random'], f.policy)}${sel_('outcome', ['', 'won', 'hazard', 'fall', 'timeout'], f.outcome)}
      ${sel_('source', ['', 'checkpoint', 'final', 'reference', 'scripted', 'random'], f.source)}
      <label class="field">nivel<input type="text" name="level" placeholder="builtin:7 · pcg:tiny:1000003 · hash" value="${escapeHtml(f.level || '')}"></label>
      <label class="field">tag<input type="text" name="tag" value="${escapeHtml(f.tag || '')}"></label>
      ${sel_('order', ['number', 'latest', 'random'], f.order)}<button class="btn">Filtrar</button></form>
    <div class="gallery" id="gal"></div><div style="text-align:center;margin:10px"><button class="btn" data-a="more" hidden>Cargar más</button></div>`;
  const gal = main.querySelector('#gal'), more = main.querySelector('[data-a=more]'), wallBtn = main.querySelector('[data-a=wall]');
  main.querySelector('form').onsubmit = (e) => {
    e.preventDefault();
    const q = new URLSearchParams([...new FormData(e.target)].filter(([, v]) => v));
    location.hash = '#/replays?' + q;
  };
  wallBtn.onclick = () => sendToWall([...sel]);
  const io = new IntersectionObserver((ents) => ents.forEach((en) => { if (en.isIntersecting) { io.unobserve(en.target); en.target._load(); } }), { rootMargin: '200px' });
  let after = '';
  const page = async () => {
    const q = new URLSearchParams(Object.entries(f).filter(([, v]) => v)); q.set('limit', '48'); if (after) q.set('after', after);
    const r = await getJSON('/api/replays?' + q);
    main.querySelector('#tot').textContent = `${fmtInt(r.total)} replays`;
    if (!r.total) gal.innerHTML = `<div class="empty card" style="grid-column:1/-1">No hay replays con estos filtros. Graba con <code>python -m rl.rollout --run RUN</code>, desde <a href="#/operate">Operar</a>, o mira <a href="#/replays?run=_reference">las 9 soluciones de referencia</a>.</div>`;
    for (const it of r.items) gal.appendChild(thumb(it));
    after = r.next; more.hidden = !after;
  };
  const thumb = (it) => {
    const el = document.createElement('article');
    el.className = 'thumb';
    const o = outcomeLabel(it.outcome);
    el.innerHTML = `<div class="pics"><canvas aria-label="primer frame"></canvas><canvas aria-label="último frame"></canvas></div>
      <div class="meta"><div><b class="mono">${replayNumber(it.n)}</b> <span class="pill">${escapeHtml(it.run === '_reference' ? 'referencia' : it.run)}</span> <span class="badge ${o.cls}">${o.label}</span></div>
      <div class="muted small">${escapeHtml(it.policy)} · ${escapeHtml(sourceLabel(it.source))}</div>
      <div class="muted small">${escapeHtml(levelLabel(it.level))} · ${fmtInt(it.outcome?.engine_steps)} pasos${it.broken ? ' · <span class="error">✘ roto</span>' : ''}${it.demo ? ' · <span class="badge demo">DEMO</span>' : ''}</div></div>
      <div class="actions"><button class="btn" data-a="open">▶ Abrir</button><button class="btn" data-a="wall">Al muro</button>
      <label class="small"><input type="checkbox" data-a="sel"> elegir</label></div>`;
    el.querySelector('[data-a=open]').onclick = () => openPlayerDialog(it.id);
    el.querySelector('[data-a=wall]').onclick = () => sendToWall([it.id]);
    el.querySelector('[data-a=sel]').onchange = (e) => {
      e.target.checked ? sel.add(it.id) : sel.delete(it.id);
      el.classList.toggle('selected', e.target.checked);
      wallBtn.disabled = !sel.size; wallBtn.textContent = `Enviar selección al muro (${sel.size})`;
    };
    el._load = async () => {
      try {
        const [lv, tr] = await Promise.all([getLevel(it.level), getTrajectory(it.id, 'none')]);
        const [a, b] = el.querySelectorAll('canvas');
        for (const [c, i] of [[a, 0], [b, tr.engine_steps]]) { const f2 = frameCanvas(lv, tr, i); c.width = f2.width; c.height = f2.height; c.getContext('2d').drawImage(f2, 0, 0); }
      } catch (e) { el.querySelector('.pics').innerHTML = `<div class="empty small error">${escapeHtml(e.message)}</div>`; }
    };
    io.observe(el);
    return el;
  };
  more.onclick = page;
  await page();
  return () => io.disconnect();
}
function sel_(name, opts, v) {
  return `<label class="field">${name}<select name="${name}">${opts.map((o) => `<option value="${escapeHtml(o)}" ${o === (v || '') ? 'selected' : ''}>${escapeHtml(o || '—')}</option>`).join('')}</select></label>`;
}

// -- Levels ----------------------------------------------------------------------
function drawLevel(c, lv, maxW = 360) {
  c.width = lv.width; c.height = lv.height;
  c.getContext('2d').drawImage(staticLayer(lv), 0, 0);
  const s = Math.min(maxW / lv.width, 220 / lv.height);
  c.style.width = Math.round(lv.width * (s >= 1 ? Math.floor(s) : s)) + 'px';
  c.style.imageRendering = 'pixelated';
}

export async function renderLevels(main) {
  main.innerHTML = `<div class="page-head"><h1>Niveles</h1></div>
    <h2 style="font-size:14px">Los 9 niveles integrados</h2><div class="grid-cards" id="bl"></div>
    <h2 style="font-size:14px;margin-top:18px">Explorador del generador</h2>
    <div class="split"><section class="card"><form id="gen" class="form-grid">
      <label class="field">preset<select name="preset">${['tiny', 'small', 'wide', 'tall', 'large', 'huge'].map((p) => `<option>${p}</option>`).join('')}</select></label>
      <label class="field">seed<input type="number" name="seed" value="1000000" min="0"></label>
      <div class="row" style="display:flex;gap:4px;align-items:flex-end"><button type="button" class="btn" data-a="prev">◀ seed</button><button type="button" class="btn" data-a="next">seed ▶</button></div>
      <details style="grid-column:1/-1"><summary class="small">config avanzada (sustituye al preset)</summary><div class="form-grid" style="margin-top:6px">
        <label class="field">cols (×30)<input type="number" name="cols" step="30" min="30" max="240"></label>
        <label class="field">rows (×20)<input type="number" name="rows" step="20" min="20" max="160"></label>
        <label class="field">difficulty<input type="number" name="difficulty" step="0.05" min="0" max="1"></label>
        <label class="field">win_condition<select name="win_condition"><option value="">(preset)</option><option>exit_with_gem</option><option>gem</option></select></label>
        <label class="small"><input type="checkbox" name="no_pits"> sin fosos</label><label class="small"><input type="checkbox" name="no_spikes"> sin pinchos</label><label class="small"><input type="checkbox" name="no_distractors"> sin distractores</label>
      </div></details>
      <button class="btn primary">Generar</button></form><div id="genout" style="margin-top:8px"></div></section>
    <section class="card"><h2>Rango expresivo (métricas de pcg.metrics sobre N seeds)</h2>
      <form id="er" class="filters"><label class="field">preset<select name="preset">${['tiny', 'small', 'wide', 'tall', 'large', 'huge'].map((p) => `<option>${p}</option>`).join('')}</select></label>
      <label class="field">desde seed<input type="number" name="start" value="0" min="0"></label><label class="field">N<input type="number" name="n" value="150" min="1" max="500"></label>
      <label class="field">x<select name="x">${['linearity', 'leniency', 'density', 'min_window', 'mean_window', 'path_ledges', 'backtrack_ratio'].map((m) => `<option>${m}</option>`).join('')}</select></label>
      <label class="field">y<select name="y">${['leniency', 'linearity', 'density', 'min_window', 'mean_window', 'path_ledges', 'backtrack_ratio'].map((m) => `<option>${m}</option>`).join('')}</select></label><button class="btn">Calcular</button></form>
      <div id="sc" style="height:320px"></div></section></div>`;
  const bl = main.querySelector('#bl');
  const { levels } = await getJSON('/api/levels/builtin');
  for (const L of levels) {
    const c = document.createElement('section'); c.className = 'card';
    c.innerHTML = `<h2>${L.n} · ${escapeHtml(L.name)}</h2><canvas></canvas><dl class="kv small" style="margin-top:6px"><dt>tamaño</dt><dd>${L.width}×${L.height}</dd><dt>victoria</dt><dd>${L.win_condition === 'gem' ? 'coger la gema' : 'gema y puerta'}</dd>
      <dt>límite</dt><dd>${fmtInt(L.max_steps ?? 1800)} pasos</dd><dt>solución</dt><dd>${fmtInt(L.reference_steps)} pasos</dd><dt>geometría</dt><dd>${L.platforms} plataformas · ${L.hazards} pinchos</dd></dl>
      <div style="display:flex;gap:6px;margin-top:6px"><button class="btn" data-a="sol">▶ Solución (fantasma)</button><a class="btn" href="#/replays?level=builtin:${L.n}">Replays</a></div>`;
    bl.appendChild(c);
    getLevel({ ref: L.ref, hash: L.hash }).then((lv) => drawLevel(c.querySelector('canvas'), lv, 300));
    c.querySelector('[data-a=sol]').onclick = () => openPlayerDialog(L.reference_id, { trail: true });
  }
  const gen = main.querySelector('#gen'), out = main.querySelector('#genout');
  const generate = async () => {
    const fd = new FormData(gen);
    const ref = { kind: 'pcg', preset: fd.get('preset'), seed: Number(fd.get('seed')) };
    const adv = ['cols', 'rows', 'difficulty'].some((k) => fd.get(k)) || fd.get('win_condition') || ['no_pits', 'no_spikes', 'no_distractors'].some((k) => fd.get(k));
    if (adv) {
      const base = { tiny: [30, 20, 0.2], small: [60, 40, 0.5], wide: [90, 20, 0.5], tall: [30, 60, 0.5], large: [120, 80, 0.5], huge: [240, 160, 0.7] }[ref.preset];
      ref.config = { cols: Number(fd.get('cols')) || base[0], rows: Number(fd.get('rows')) || base[1], difficulty: fd.get('difficulty') === '' ? base[2] : Number(fd.get('difficulty')),
        floor_pits: !fd.get('no_pits'), ledge_spikes: !fd.get('no_spikes'), distractors: !fd.get('no_distractors'), win_condition: fd.get('win_condition') || 'exit_with_gem', max_attempts: 20 };
      ref.preset = null;
    }
    out.innerHTML = '<span class="loading"></span>';
    try {
      const lv = await getJSON(`/api/levels/by-ref?ref=${enc(JSON.stringify(ref))}`);
      out.innerHTML = `<div class="stage" style="min-height:120px"><canvas class="game"></canvas></div><dl class="kv small" style="margin-top:6px"><dt>nombre</dt><dd>${escapeHtml(lv.name)}</dd><dt>tamaño</dt><dd>${lv.width}×${lv.height}</dd>
        ${Object.entries(lv.metrics || {}).map(([k, v]) => `<dt>${k}</dt><dd>${fmtNum(v, 3)}</dd>`).join('')}<dt>hash</dt><dd class="mono">${escapeHtml(lv.hash.slice(0, 23))}…</dd></dl>
        <a class="btn" href="#/replays?level=pcg:${escapeHtml(ref.preset || '')}:${ref.seed}">Replays de este nivel</a>`;
      drawLevel(out.querySelector('canvas'), lv, 560);
    } catch (e) { out.innerHTML = `<span class="error">✘ ${escapeHtml(e.message)}</span>`; }
  };
  gen.onsubmit = (e) => { e.preventDefault(); generate(); };
  gen.querySelector('[data-a=prev]').onclick = () => { gen.seed.value = Math.max(0, Number(gen.seed.value) - 1); generate(); };
  gen.querySelector('[data-a=next]').onclick = () => { gen.seed.value = Number(gen.seed.value) + 1; generate(); };
  generate();
  const er = main.querySelector('#er'), sc = new Scatter(main.querySelector('#sc'));
  const expressive = async () => {
    const fd = new FormData(er);
    sc.note('calculando en el servidor…');
    try {
      const r = await getJSON(`/api/levels/expressive?preset=${fd.get('preset')}&start=${fd.get('start')}&n=${fd.get('n')}`);
      const x = fd.get('x'), y = fd.get('y');
      sc.set(r.points.map((p) => ({ x: p[x], y: p[y], label: `seed ${fmtInt(p.seed)}` })), { xLabel: x, yLabel: y });
      sc.note(`${r.points.length} niveles ${r.preset} desde la seed ${fmtInt(r.start)} · cada punto es un nivel`);
    } catch (e) { sc.note('✘ ' + e.message); }
  };
  er.onsubmit = (e) => { e.preventDefault(); expressive(); };
  expressive();
  return () => sc.destroy();
}

// -- Operate -----------------------------------------------------------------------
const TRAIN_MAIN = ['name', 'group', 'preset', 'levels', 'num_levels', 'start_seed', 'total_steps', 'distance_shaping', 'gem_reward', 'seed',
  'ckpt_every', 'replay_every', 'replay_n', 'eval_every', 'device', 'threads', 'log_every'];
const ROLLOUT_MAIN = ['run', 'ckpt', 'preset', 'levels', 'start_seed', 'num', 'repeat', 'policy', 'tag', 'seed', 'workers'];
const DEFAULTS = {
  train: { name: '', preset: 'tiny', num_levels: 100, total_steps: 300000, distance_shaping: 1.0, gem_reward: 0.5, seed: 1, ckpt_every: 10, replay_every: 10, replay_n: 4, eval_every: 20, log_every: 2, device: 'auto', threads: 2 },
  rollout: { ckpt: 'model.pt', preset: 'tiny', start_seed: 1000000, num: 8, policy: 'greedy' },
  sweep: { ckpt: 'model.pt', preset: 'tiny', start_seed: 1000000, num: 1000, policy: 'greedy', tag: 'test1k', workers: 4 },
};

function schemaForm(kind, schema, mainKeys) {
  const field = (k, s) => {
    const v = DEFAULTS[kind][k] ?? '';
    const label = `${k}${s.min != null ? ` <span class="muted">[${s.min}–${s.max}]</span>` : ''}`;
    if (k === 'run') return `<label class="field">run<select name="run" aria-label="run">${state.runs.map((r) => `<option>${escapeHtml(r.name)}</option>`).join('')}</select></label>`;
    if (s.type === 'enum') return `<label class="field">${label}<select name="${k}" aria-label="${k}"><option value="">—</option>${s.choices.map((c) => `<option ${c === v ? 'selected' : ''}>${c}</option>`).join('')}</select></label>`;
    if (s.type === 'bool') return `<label class="small"><input type="checkbox" name="${k}" aria-label="${k}"> ${k}</label>`;
    if (s.type === 'int' || s.type === 'float') return `<label class="field">${label}<input type="number" name="${k}" aria-label="${k}" value="${v}" min="${s.min}" max="${s.max}" step="${s.type === 'int' ? 1 : 'any'}"></label>`;
    return `<label class="field">${label}<input type="text" name="${k}" aria-label="${k}" value="${escapeHtml(v)}" pattern="${escapeHtml(s.pattern || '')}"></label>`;
  };
  const keys = Object.keys(schema);
  const main = mainKeys.filter((k) => schema[k]), rest = keys.filter((k) => !main.includes(k));
  return `<div class="form-grid">${main.map((k) => field(k, schema[k])).join('')}</div>
    <details><summary class="small">avanzado (${rest.length} flags)</summary><div class="form-grid" style="margin-top:6px">${rest.map((k) => field(k, schema[k])).join('')}</div></details>`;
}

function readForm(form, schema) {
  const args = {};
  for (const [k, s] of Object.entries(schema)) {
    const el = form.elements[k];
    if (!el) continue;
    if (s.type === 'bool') { if (el.checked) args[k] = true; continue; }
    const v = el.value;
    if (v === '' || v == null) continue;
    args[k] = s.type === 'int' ? parseInt(v, 10) : s.type === 'float' ? parseFloat(v) : v;
  }
  return args;
}

export async function renderOperate(main) {
  const schema = await getJSON('/api/jobs/schema');
  const enabled = state.jobsEnabled;
  main.innerHTML = `<div class="page-head"><h1>Operar</h1>${enabled ? '' : `<span class="badge warn">⚠ deshabilitado: ${escapeHtml(state.jobsNote)}</span>`}</div>
    <div class="grid-cards" id="og"></div>`;
  const g = main.querySelector('#og');
  const forms = [['train', 'Entrenar (python -m rl.ppo)', TRAIN_MAIN], ['rollout', 'Grabar replays (rl.rollout --mode replays)', ROLLOUT_MAIN], ['sweep', 'Barrer (rl.rollout --mode sweep)', [...ROLLOUT_MAIN, 'max_episodes']]];
  for (const [kind, title, keys] of forms) {
    const c = card(g, title, null);
    c.innerHTML = `<form data-kind="${kind}">${schemaForm(kind, schema[kind], keys)}
      ${kind === 'train' ? '<label class="small" style="display:block;margin-top:6px"><input type="checkbox" name="__then"> al terminar, barrer 1 000 niveles no vistos</label>' : ''}
      <div style="margin-top:8px;display:flex;gap:6px;align-items:center"><button class="btn primary" ${enabled ? '' : 'disabled'}>Lanzar</button><span class="muted small out"></span></div></form>`;
    const form = c.querySelector('form');
    if (kind === 'train') form.elements.levels.value = '';
    form.onsubmit = async (e) => {
      e.preventDefault();
      const args = readForm(form, schema[kind]);
      if (kind === 'train' && args.preset && args.levels) delete args.levels;
      if (kind !== 'train' && args.preset && args.levels) delete args.preset;
      const body = { kind, args };
      if (kind === 'train' && form.elements.__then?.checked && args.preset) body.then = [{ kind: 'sweep', args: { preset: args.preset, start_seed: 1000000, num: 1000, tag: 'test1k', policy: 'greedy', workers: 4 } }];
      try {
        const job = await send('POST', '/api/jobs', body);
        form.querySelector('.out').textContent = `job ${job.id} · ${job.state} · run ${job.run}`;
        toast(`Job ${job.id} encolado (${kind})`); refreshJobs();
      } catch (err) { form.querySelector('.out').innerHTML = `<span class="error">✘ ${escapeHtml(err.message)}</span>`; }
    };
  }
  const jc = card(g, 'Trabajos', null);
  jc.parentElement.style.gridColumn = '1 / -1';
  const lc = card(g, 'Log', 340); lc.parentElement.style.gridColumn = '1 / -1';
  let logInst = TILES['job-log'].create(lc, { job: '$latest' }, tileApi);
  const drawJobs = () => {
    if (!state.jobs.length) { jc.innerHTML = '<p class="muted">Sin trabajos todavía. Los entrenamientos lanzados fuera de la UI aparecen aquí si siguen vivos (adopción por <code>status.json</code>).</p>'; return; }
    jc.innerHTML = `<table class="data"><thead><tr><th>id</th><th>tipo</th><th>run</th><th>estado</th><th class="num">pid</th><th>inicio</th><th>argumentos</th><th></th></tr></thead><tbody>${state.jobs.map((j) => `<tr>
      <td>${j.id}</td><td>${j.kind}${j.external ? ' <span class="pill">adoptado</span>' : ''}</td><td><a href="#/runs/${enc(j.run)}">${escapeHtml(j.run)}</a></td><td><span class="state ${j.state}">${j.state}</span>${j.error ? ` <span class="error small" title="${escapeHtml(j.error)}">✘</span>` : ''}</td>
      <td class="num">${j.pid ?? '–'}</td><td class="small">${j.started ? new Date(j.started * 1000).toLocaleTimeString() : '–'}</td>
      <td class="small mono" style="max-width:420px;overflow:hidden;text-overflow:ellipsis" title="${escapeHtml(j.argv.join(' '))}">${escapeHtml(j.argv.slice(1).join(' '))}</td>
      <td><button class="btn" data-log="${j.id}">log</button> ${['running', 'queued'].includes(j.state) ? `<button class="btn danger" data-stop="${j.id}">Parar</button>` : ''}${j.term_sent && j.state === 'running' ? ` <button class="btn danger" data-kill="${j.id}" title="SIGKILL (30 s después de Parar)">Forzar</button>` : ''}</td></tr>`).join('')}</tbody></table>`;
    jc.querySelectorAll('[data-stop]').forEach((b) => { b.onclick = async () => { try { await send('DELETE', `/api/jobs/${b.dataset.stop}`); toast('Parada solicitada'); } catch (e) { toast(e.message); } }; });
    jc.querySelectorAll('[data-kill]').forEach((b) => { b.onclick = async () => { try { await send('DELETE', `/api/jobs/${b.dataset.kill}?force=1`); } catch (e) { toast(e.message); } }; });
    jc.querySelectorAll('[data-log]').forEach((b) => { b.onclick = () => { logInst.destroy?.(); logInst = TILES['job-log'].create(lc, { job: b.dataset.log }, tileApi); }; });
  };
  drawJobs();
  const off = on('jobs', drawJobs);
  return () => { off(); logInst.destroy?.(); };
}
