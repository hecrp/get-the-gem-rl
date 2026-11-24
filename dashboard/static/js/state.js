// Shared client state: health, runs and jobs, kept fresh by SSE events.
import { getJSON, events } from './api.js';

export const state = new EventTarget();
state.health = null; state.runs = []; state.jobs = []; state.jobsEnabled = false; state.jobsNote = '';

let runsTimer = null;
export async function refreshRuns() {
  const r = await getJSON('/api/runs');
  state.runs = r.runs;
  state.dispatchEvent(new Event('runs'));
  return state.runs;
}
export async function refreshJobs() {
  const r = await getJSON('/api/jobs');
  state.jobs = r.jobs; state.jobsEnabled = r.enabled; state.jobsNote = r.note;
  state.dispatchEvent(new Event('jobs'));
  return state.jobs;
}
export async function init() {
  state.health = await getJSON('/api/health');
  await Promise.all([refreshRuns(), refreshJobs()]);
  events.on('run.updated', () => { clearTimeout(runsTimer); runsTimer = setTimeout(refreshRuns, 300); });
  events.on('job.updated', (job) => {
    const i = state.jobs.findIndex((j) => j.id === job.id);
    if (i >= 0) state.jobs[i] = job; else state.jobs.unshift(job);
    state.dispatchEvent(new Event('jobs'));
  });
}
export const on = (type, fn) => { state.addEventListener(type, fn); return () => state.removeEventListener(type, fn); };

const ts = (r) => Date.parse(r.status?.updated_at || r.status?.started_at || '') || 0;
/** "$latest" = the running run updated last, else the run updated last. */
export function resolveRun(name) {
  if (name !== '$latest') return name;
  const runs = [...state.runs];
  if (!runs.length) return null;
  const running = runs.filter((r) => r.state === 'running');
  const pool = running.length ? running : runs;
  pool.sort((a, b) => ts(b) - ts(a) || (a.name < b.name ? 1 : -1));
  return pool[0].name;
}
export function resolveRuns(list) {
  if (!list || !list.length) return [];
  if (list.includes('$all')) return state.runs.map((r) => r.name);
  return [...new Set(list.map(resolveRun).filter(Boolean))];
}
export function resolveJob(id) {
  if (id !== '$latest') return state.jobs.find((j) => String(j.id) === String(id)) || null;
  return state.jobs.find((j) => j.state === 'running') || state.jobs[0] || null;
}
export const isDemo = () => !!state.health?.demo;
