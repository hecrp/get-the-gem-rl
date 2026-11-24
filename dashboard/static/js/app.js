// Entry point: theme, state, router, global shortcuts and indicators.
import { getJSON, events } from './api.js';
import { loadTheme } from './renderer.js';
import { init, state, on } from './state.js';
import { ticker, nextSpeed } from './clock.js';
import { renderWall, maximizeFocused, restoreMaximized } from './wall.js';
import { renderRuns, renderReplays, renderLevels, renderOperate } from './pages.js';
import { toast } from './ui.js';

const main = document.getElementById('main');
let cleanup = null;

function applyTheme() {
  let t = null;
  try { t = localStorage.getItem('gtg.theme'); } catch { /* ignore */ }
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
}
function toggleTheme() {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('gtg.theme', next); } catch { /* ignore */ }
  route();
}

async function route() {
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  const hash = location.hash.slice(1) || '/wall';
  const [path, query = ''] = hash.split('?');
  const parts = path.split('/').filter(Boolean);
  const page = parts[0] || 'wall';
  const params = new URLSearchParams(query);
  document.querySelectorAll('.nav a').forEach((a) => a.removeAttribute('aria-current'));
  document.querySelectorAll('.nav a').forEach((a) => { if (a.dataset.page === page) a.setAttribute('aria-current', 'page'); });
  main.innerHTML = '<div class="empty loading"></div>';
  try {
    if (page === 'runs') cleanup = await renderRuns(main, parts[1] && decodeURIComponent(parts[1]));
    else if (page === 'replays') cleanup = await renderReplays(main, params);
    else if (page === 'levels') cleanup = await renderLevels(main);
    else if (page === 'operate') cleanup = await renderOperate(main);
    else cleanup = await renderWall(main, params);
  } catch (e) {
    console.error(e);
    main.innerHTML = `<div class="empty error">✘ ${e.message}</div>`;
  }
}

function indicators() {
  const n = state.runs.filter((r) => r.state === 'running').length;
  const el = document.getElementById('active-runs');
  el.textContent = `● ${n} run${n === 1 ? '' : 's'} activo${n === 1 ? '' : 's'}`;
  el.classList.toggle('ok', n > 0);
}

function keys(e) {
  if (e.target.closest('input, textarea, select, [contenteditable]') || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === ' ') { ticker.globalPaused = !ticker.globalPaused; toast(ticker.globalPaused ? '⏸ Pausa global' : '▶ Reanudado', 1200); }
  else if (k === '+' || k === '=') { ticker.globalSpeed = nextSpeed(ticker.globalSpeed, 1); toast(`Velocidad global ${ticker.globalSpeed}×`, 1200); }
  else if (k === '-' || k === '−') { ticker.globalSpeed = nextSpeed(ticker.globalSpeed, -1); toast(`Velocidad global ${ticker.globalSpeed}×`, 1200); }
  else if (k === '?') document.getElementById('help').showModal();
  else if (k === 't') toggleTheme();
  else if (k === 'f' || k === 'F') maximizeFocused();
  else if (k === 'Escape') { if (!restoreMaximized()) return; }
  else if ('12345'.includes(k)) location.hash = ['#/wall', '#/runs', '#/replays', '#/levels', '#/operate'][Number(k) - 1];
  else return;
  e.preventDefault();
}

async function start() {
  applyTheme();
  document.getElementById('theme-toggle').onclick = toggleTheme;
  document.getElementById('help-btn').onclick = () => document.getElementById('help').showModal();
  const sse = document.getElementById('sse-status');
  events.addEventListener('state', (e) => {
    const ok = e.detail === 'open';
    sse.textContent = ok ? 'SSE conectado' : e.detail === 'connecting' ? 'SSE conectando…' : 'SSE desconectado (reintentando)';
    sse.classList.toggle('ok', ok); sse.classList.toggle('bad', !ok);
  });
  events.connect();
  try {
    await Promise.all([loadTheme(getJSON), init()]);
  } catch (e) {
    main.innerHTML = `<div class="empty error">✘ No se pudo contactar con el servidor: ${e.message}</div>`;
    return;
  }
  document.getElementById('demo-badge').hidden = !state.health.demo;
  indicators();
  on('runs', indicators);
  window.addEventListener('hashchange', route);
  document.addEventListener('keydown', keys);
  route();
}
start();
