// HTTP + SSE client. Writes carry the page's token (R-14).
export const TOKEN = document.querySelector('meta[name="dashboard-token"]')?.content || '';

export class ApiError extends Error {
  constructor(status, code, detail) { super(detail || code); this.status = status; this.code = code; }
}

async function check(r) {
  if (r.ok) return r;
  let body = {};
  try { body = await r.json(); } catch { /* not JSON */ }
  throw new ApiError(r.status, body.error || 'http_error', body.detail || r.statusText);
}

export async function getJSON(url, opts = {}) {
  return (await check(await fetch(url, { headers: { Accept: 'application/json' }, ...opts }))).json();
}
export async function getText(url) { return (await check(await fetch(url))).text(); }

export async function send(method, url, body) {
  const r = await fetch(url, {
    method, headers: { 'Content-Type': 'application/json', 'X-Dashboard-Token': TOKEN },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return (await check(r)).json();
}

// -- small caches ------------------------------------------------------------
class LRU {
  constructor(max) { this.max = max; this.map = new Map(); }
  get(k) { const v = this.map.get(k); if (v !== undefined) { this.map.delete(k); this.map.set(k, v); } return v; }
  set(k, v) { this.map.delete(k); this.map.set(k, v); while (this.map.size > this.max) this.map.delete(this.map.keys().next().value); }
}
const replayCache = new LRU(400), trajCache = new LRU(120), levelCache = new LRU(80);

export const splitId = (id) => { const i = id.indexOf('/'); return [id.slice(0, i), id.slice(i + 1)]; };
const enc = encodeURIComponent;

export function getReplay(id) {
  let p = replayCache.get(id);
  if (!p) {
    const [run, n] = splitId(id);
    p = getJSON(`/api/replays/${enc(run)}/${enc(n)}`);
    p.catch(() => replayCache.map.delete(id));
    replayCache.set(id, p);
  }
  return p;
}

export function getTrajectory(id, cam = 'none') {
  const key = id + '|' + cam;
  let p = trajCache.get(key);
  if (!p) {
    const [run, n] = splitId(id);
    p = getJSON(`/api/replays/${enc(run)}/${enc(n)}/trajectory?cam=${cam}`).then((t) => ({
      ...t, xy: Float32Array.from(t.xy), flags: Uint8Array.from(t.flags),
      cam: t.cam ? Int32Array.from(t.cam) : null,
    }));
    p.catch(() => trajCache.map.delete(key));
    trajCache.set(key, p);
  }
  return p;
}

export function getLevel(level) {
  const key = level.hash || JSON.stringify(level.ref);
  let p = levelCache.get(key);
  if (!p) {
    const ref = level.ref;
    p = (ref && ref.kind !== 'custom')
      ? getJSON(`/api/levels/by-ref?ref=${enc(JSON.stringify(ref))}&hash=${enc(level.hash || '')}`)
      : getJSON(`/api/levels/by-hash/${enc(level.hash)}`);
    p.catch(() => levelCache.map.delete(key));
    levelCache.set(key, p);
  }
  return p;
}

// -- server-sent events -----------------------------------------------------
class Events extends EventTarget {
  constructor() { super(); this.state = 'connecting'; this.es = null; }
  connect() {
    if (this.es) this.es.close();
    this.es = new EventSource('/api/events');
    const setState = (s) => { this.state = s; this.dispatchEvent(new CustomEvent('state', { detail: s })); };
    this.es.onopen = () => setState('open');
    this.es.onerror = () => setState(this.es.readyState === 2 ? 'closed' : 'error');
    for (const type of ['hello', 'run.updated', 'job.updated', 'replay.added', 'heartbeat', 'error']) {
      this.es.addEventListener(type, (e) => {
        let data = null;
        try { data = JSON.parse(e.data); } catch { /* ignore */ }
        if (type === 'hello' || type === 'heartbeat') setState('open');
        this.dispatchEvent(new CustomEvent(type, { detail: data }));
      });
    }
  }
  on(type, fn) { const h = (e) => fn(e.detail); this.addEventListener(type, h); return () => this.removeEventListener(type, h); }
}
export const events = new Events();
