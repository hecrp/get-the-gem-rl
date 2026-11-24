// The wall: a CSS grid of swappable tiles; layouts live in the URL hash and localStorage.
import { TILES, runOptions } from './tiles.js';
import { GRIDS, SPANS, TILE_TYPES, validateLayout, factoryLayouts, regrid, emptyLayout, encodeLayout, decodeLayout } from './layout.js';
import { state } from './state.js';
import { getJSON } from './api.js';
import { escapeHtml } from './format.js';
import { toast } from './ui.js';

const LS_SAVED = 'gtg.layouts', LS_LAST = 'gtg.lastLayout';
const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
export const savedLayouts = () => ls.get(LS_SAVED, {});

let current = null;          // {layout, tiles: [{el, inst}]}

export function currentLayout() {
  if (current) return current.layout;
  const last = ls.get(LS_LAST, null);
  const v = last && validateLayout(last);
  return v && v.ok ? v.layout : defaultLayout();
}

function defaultLayout() {
  const f = factoryLayouts();
  return state.runs.length ? f['Muro de replays'] : f['Muro de replays'];
}

export async function sendToWall(ids) {
  let layout = JSON.parse(JSON.stringify(currentLayout()));
  for (const id of ids) {
    let i = layout.tiles.findIndex((t) => t.type === 'empty');
    if (i < 0) {
      const order = Object.keys(GRIDS);
      const cur = order.findIndex((g) => GRIDS[g][0] === layout.cols && GRIDS[g][1] === layout.rows);
      const area = layout.tiles.reduce((s, t) => s + t.span[0] * t.span[1], 0);
      const bigger = order.slice(cur + 1).find((g) => GRIDS[g][0] * GRIDS[g][1] > area);
      if (bigger) { layout = regrid(layout, bigger); i = layout.tiles.findIndex((t) => t.type === 'empty'); }
    }
    if (i < 0) i = layout.tiles.length - 1;
    layout.tiles[i] = { type: 'replay', span: [1, 1], props: { id, camera: 'auto', loop: true } };
  }
  ls.set(LS_LAST, layout);
  location.hash = '#/wall?l=' + await encodeLayout(layout);
}

export async function renderWall(main, params) {
  let layout = null;
  if (params.get('l')) {
    try { layout = await decodeLayout(params.get('l')); } catch (e) { toast('Diseño de la URL no válido: ' + e.message); }
  }
  if (!layout && params.get('layout')) layout = validateLayout(factoryLayouts()[params.get('layout')] || savedLayouts()[params.get('layout')] || {}).layout;
  layout ||= currentLayout();
  main.innerHTML = `
    <div class="wall-toolbar" role="toolbar" aria-label="Muro">
      <label class="small muted">Diseño <select data-a="preset" aria-label="Diseño">
        <option value="">— ${escapeHtml(layout.name || 'actual')} —</option>
        <optgroup label="De fábrica">${Object.keys(factoryLayouts()).map((n) => `<option value="f:${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('')}</optgroup>
        <optgroup label="Guardados">${Object.keys(savedLayouts()).map((n) => `<option value="s:${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('')}</optgroup>
      </select></label>
      <label class="small muted">Rejilla <select data-a="grid" aria-label="Rejilla">${Object.keys(GRIDS).map((g) => `<option ${GRIDS[g][0] === layout.cols && GRIDS[g][1] === layout.rows ? 'selected' : ''}>${g}</option>`).join('')}</select></label>
      <button class="btn" data-a="save">Guardar…</button>
      <button class="btn" data-a="export">Exportar</button>
      <label class="btn">Importar<input type="file" accept="application/json,.json" data-a="import" hidden></label>
      <button class="btn" data-a="share" title="Copia un enlace con el diseño">Compartir URL</button>
      <button class="btn ghost" data-a="del" title="Borrar el diseño guardado seleccionado">Borrar guardado</button>
      <span class="spacer"></span><span class="muted small" id="fps"></span>
    </div>
    <div class="wall" id="wall"></div>`;
  const wallEl = main.querySelector('#wall');
  const set = async (l, push = true) => {
    const v = validateLayout(l);
    if (!v.ok) { toast('Diseño no válido: ' + v.errors.join('; ')); return; }
    layout = v.layout; current = { layout };
    ls.set(LS_LAST, layout);
    if (push) history.replaceState(null, '', '#/wall?l=' + await encodeLayout(layout));
    build();
  };
  const tb = main.querySelector('.wall-toolbar');
  tb.querySelector('[data-a=preset]').onchange = (e) => {
    const v = e.target.value; if (!v) return;
    set(v.startsWith('f:') ? factoryLayouts()[v.slice(2)] : savedLayouts()[v.slice(2)]);
  };
  tb.querySelector('[data-a=grid]').onchange = (e) => set(regrid(layout, e.target.value));
  tb.querySelector('[data-a=save]').onclick = () => {
    const name = prompt('Nombre del diseño', layout.name || 'Mi muro');
    if (!name) return;
    const all = savedLayouts(); all[name] = { ...layout, name }; ls.set(LS_SAVED, all);
    set({ ...layout, name }); toast(`Guardado «${name}» (en este navegador)`);
  };
  tb.querySelector('[data-a=del]').onclick = () => {
    const v = tb.querySelector('[data-a=preset]').value;
    if (!v.startsWith('s:')) return toast('Selecciona primero un diseño guardado');
    const all = savedLayouts(); delete all[v.slice(2)]; ls.set(LS_SAVED, all); renderWall(main, params);
  };
  tb.querySelector('[data-a=export]').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(layout, null, 2)], { type: 'application/json' }));
    a.download = `${(layout.name || 'muro').replace(/[^\w.-]+/g, '_')}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  tb.querySelector('[data-a=import]').onchange = async (e) => {
    try { await set(JSON.parse(await e.target.files[0].text())); toast('Diseño importado'); } catch (err) { toast('No se pudo importar: ' + err.message); }
  };
  tb.querySelector('[data-a=share]').onclick = async () => {
    const url = location.origin + '/#/wall?l=' + await encodeLayout(layout);
    try { await navigator.clipboard.writeText(url); toast('Enlace copiado'); } catch { prompt('Copia el enlace', url); }
  };

  let tiles = [];
  const destroyTiles = () => { for (const t of tiles) t.inst?.destroy?.(); tiles = []; };
  const build = () => {
    destroyTiles();
    wallEl.innerHTML = '';
    wallEl.style.gridTemplateColumns = `repeat(${layout.cols}, minmax(0, 1fr))`;
    wallEl.style.gridTemplateRows = `repeat(${layout.rows}, minmax(0, 1fr))`;
    layout.tiles.forEach((t, i) => tiles.push(mountTile(wallEl, t, i)));
  };

  const mountTile = (parent, t, i) => {
    const el = document.createElement('section');
    el.className = 'tile'; el.tabIndex = -1;
    el.style.gridColumn = `span ${t.span[0]}`; el.style.gridRow = `span ${t.span[1]}`;
    const def = TILES[t.type] || TILES.empty;
    el.setAttribute('aria-label', def.label);
    el.innerHTML = `<header class="tile-head" draggable="true"><span class="t-type">${escapeHtml(def.label)}</span><span class="t-title"></span>
      <button class="btn ghost icon" data-a="cfg" aria-label="Ajustes del tile" title="Ajustes">⚙</button>
      <button class="btn ghost icon" data-a="max" aria-label="Maximizar" title="Maximizar (F)">⤢</button>
      <button class="btn ghost icon" data-a="close" aria-label="Cerrar tile" title="Cerrar">✕</button></header><div class="tile-body"></div>`;
    parent.appendChild(el);
    const body = el.querySelector('.tile-body'), title = el.querySelector('.t-title');
    const api = { setTitle: (s) => { title.textContent = s; title.title = s; }, toast };
    let inst = {};
    try { inst = def.create(body, t.props || {}, api) || {}; } catch (e) { body.innerHTML = `<div class="empty error">✘ ${escapeHtml(e.message)}</div>`; }
    if (inst.title) api.setTitle(inst.title);
    const rec = { el, inst };
    el.querySelector('[data-a=max]').onclick = () => toggleMax(el);
    el.querySelector('.tile-head').ondblclick = (e) => { if (!e.target.closest('button')) toggleMax(el); };
    el.querySelector('[data-a=close]').onclick = () => { const l = structuredClone(layout); l.tiles[i] = { type: 'empty', span: t.span, props: {} }; set(l); };
    el.querySelector('[data-a=cfg]').onclick = () => openSettings(el, t, i);
    const head = el.querySelector('.tile-head');
    head.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/x-tile', String(i)); e.dataTransfer.effectAllowed = 'move'; });
    el.addEventListener('dragover', (e) => { if (e.dataTransfer.types.includes('text/x-tile')) { e.preventDefault(); el.classList.add('drag-over'); } });
    el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
    el.addEventListener('drop', (e) => {
      el.classList.remove('drag-over');
      const from = Number(e.dataTransfer.getData('text/x-tile'));
      if (Number.isNaN(from) || from === i) return;
      e.preventDefault();
      const l = structuredClone(layout); [l.tiles[from], l.tiles[i]] = [l.tiles[i], l.tiles[from]]; set(l);
    });
    el.addEventListener('focusin', () => { lastFocusedTile = el; });
    el.addEventListener('pointerdown', () => { lastFocusedTile = el; });
    return rec;
  };

  const openSettings = async (el, t, i) => {
    el.querySelector('.tile-menu')?.remove();
    const menu = document.createElement('form');
    menu.className = 'tile-menu';
    const def = TILES[t.type] || TILES.empty;
    let groups = [];
    if (def.fields.some((f) => f.type === 'groups')) { try { groups = (await getJSON('/api/groups?cols=success')).groups.map((g) => g.group); } catch { /* optional */ } }
    const val = (k) => (t.props || {})[k];
    const field = (f) => {
      const id = `f-${i}-${f.key}`, label = escapeHtml(f.label || f.key), v = val(f.key);
      switch (f.type) {
        case 'select': return `<label class="field">${label}<select name="${f.key}">${f.options.map((o) => `<option ${String(v ?? '') === o ? 'selected' : ''} value="${o}">${o || '—'}</option>`).join('')}</select></label>`;
        case 'bool': return `<label class="small"><input type="checkbox" name="${f.key}" ${v === true || (v === undefined && f.key === 'loop') ? 'checked' : ''}> ${label}</label>`;
        case 'number': return `<label class="field">${label}<input type="number" name="${f.key}" value="${v ?? ''}" ${f.min != null ? `min="${f.min}"` : ''} ${f.max != null ? `max="${f.max}"` : ''} step="${f.step || 1}"></label>`;
        case 'run': return `<label class="field">${label}<select name="${f.key}">${f.allowEmpty ? `<option value="">— todos —</option>` : ''}${runOptions().map((o) => `<option ${v === o ? 'selected' : ''}>${escapeHtml(o)}</option>`).join('')}</select></label>`;
        case 'runs': case 'groups': case 'multi': {
          const opts = f.type === 'runs' ? ['$latest', '$all', ...state.runs.map((r) => r.name)] : f.type === 'groups' ? groups : f.options;
          return `<fieldset class="field" style="border:1px solid var(--border);border-radius:6px"><legend>${label}</legend>${opts.map((o) => `<label class="small" style="display:block"><input type="checkbox" name="${f.key}" value="${escapeHtml(o)}" ${(v || []).includes(o) ? 'checked' : ''}> ${escapeHtml(o)}</label>`).join('') || '<span class="muted small">ninguno</span>'}</fieldset>`;
        }
        case 'list': return `<label class="field">${label}<input type="text" name="${f.key}" value="${escapeHtml((v || []).join(', '))}" id="${id}"></label>`;
        case 'textarea': return `<label class="field">${label}<textarea name="${f.key}" rows="6">${escapeHtml(v || '')}</textarea></label>`;
        default: return `<label class="field">${label}<input type="text" name="${f.key}" value="${escapeHtml(v ?? '')}"></label>`;
      }
    };
    menu.innerHTML = `
      <div class="row"><label class="field">tipo<select name="__type">${TILE_TYPES.map((ty) => `<option value="${ty}" ${ty === t.type ? 'selected' : ''}>${escapeHtml(TILES[ty].label)}</option>`).join('')}</select></label>
      <label class="field">tamaño<select name="__span">${SPANS.filter(([a, b]) => a <= layout.cols && b <= layout.rows).map(([a, b]) => `<option value="${a}x${b}" ${a === t.span[0] && b === t.span[1] ? 'selected' : ''}>${a}×${b}</option>`).join('')}</select></label></div>
      ${def.fields.map(field).join('')}
      <details><summary class="small">props (JSON)</summary><textarea name="__json" rows="5" style="width:100%">${escapeHtml(JSON.stringify(t.props || {}, null, 1))}</textarea></details>
      <div class="row"><button class="btn primary">Aplicar</button><button type="button" class="btn" data-a="cancel">Cancelar</button></div>`;
    el.appendChild(menu);
    menu.querySelector('[data-a=cancel]').onclick = () => menu.remove();
    menu.querySelector('[name=__type]').onchange = (e) => { const l = structuredClone(layout); l.tiles[i] = { type: e.target.value, span: t.span, props: {} }; set(l); };
    menu.onsubmit = (e) => {
      e.preventDefault();
      const fd = new FormData(menu);
      let props = {};
      try { props = JSON.parse(fd.get('__json') || '{}'); } catch { return toast('JSON no válido'); }
      for (const f of def.fields) {
        if (['runs', 'groups', 'multi'].includes(f.type)) props[f.key] = fd.getAll(f.key);
        else if (f.type === 'bool') props[f.key] = fd.get(f.key) === 'on';
        else if (f.type === 'number') { const n = fd.get(f.key); if (n === '' || n == null) delete props[f.key]; else props[f.key] = Number(n); }
        else if (f.type === 'list') props[f.key] = String(fd.get(f.key) || '').split(',').map((s) => s.trim()).filter(Boolean);
        else { const s = fd.get(f.key); if (s === '' || s == null) delete props[f.key]; else props[f.key] = s; }
      }
      const [a, b] = String(fd.get('__span')).split('x').map(Number);
      const l = structuredClone(layout); l.tiles[i] = { type: t.type, span: [a, b], props }; set(l);
    };
  };

  build();
  current = { layout };
  const fpsEl = main.querySelector('#fps');
  const { ticker } = await import('./clock.js');
  const fpsTimer = setInterval(() => { if (!document.body.contains(fpsEl)) { clearInterval(fpsTimer); return; } fpsEl.textContent = `${Math.round(ticker.fps)} fps · ${ticker.workMs.toFixed(1).replace('.', ',')} ms/frame · velocidad global ${String(ticker.globalSpeed).replace('.', ',')}×${ticker.globalPaused ? ' · ⏸ pausa' : ''}`; }, 500);
  return () => { destroyTiles(); clearInterval(fpsTimer); current = null; };
}

let lastFocusedTile = null;
function toggleMax(el) {
  const on = !el.classList.contains('maximized');
  document.querySelectorAll('.tile.maximized').forEach((t) => t.classList.remove('maximized'));
  if (on) { el.classList.add('maximized'); el.focus(); }
  window.dispatchEvent(new Event('resize'));
}
export function maximizeFocused() { if (lastFocusedTile && document.body.contains(lastFocusedTile)) toggleMax(lastFocusedTile); }
export function restoreMaximized() { const m = document.querySelector('.tile.maximized'); if (m) { m.classList.remove('maximized'); return true; } return false; }
