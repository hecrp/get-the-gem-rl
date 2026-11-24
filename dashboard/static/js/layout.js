// Wall layouts: versioned JSON, validated on load, shareable in the URL hash.
export const LAYOUT_VERSION = 1;
export const GRIDS = { '1x1': [1, 1], '2x1': [2, 1], '2x2': [2, 2], '3x2': [3, 2], '3x3': [3, 3], '4x3': [4, 3], '4x4': [4, 4] };
export const SPANS = [[1, 1], [2, 1], [1, 2], [2, 2]];
export const TILE_TYPES = ['replay', 'replay-feed', 'replay-grid', 'ghosts', 'curves', 'train-vs-test',
  'run-card', 'runs-table', 'level-success', 'difficulty', 'end-reasons', 'heatmap', 'histogram',
  'level-viewer', 'job-log', 'note', 'empty'];

/** Validate (and migrate) a layout object. Returns {ok, layout, errors}. */
export function validateLayout(obj) {
  const errors = [];
  if (!obj || typeof obj !== 'object') return { ok: false, errors: ['layout: expected an object'] };
  let v = obj.v;
  let o = obj;
  if (v === undefined && Array.isArray(obj.tiles)) { v = 1; o = { ...obj, v: 1 }; }   // v0: unversioned
  if (v !== LAYOUT_VERSION) return { ok: false, errors: [`layout: unsupported version ${v}`] };
  const cols = o.cols, rows = o.rows;
  if (!Number.isInteger(cols) || cols < 1 || cols > 4) errors.push('cols: 1..4');
  if (!Number.isInteger(rows) || rows < 1 || rows > 4) errors.push('rows: 1..4');
  if (!Array.isArray(o.tiles)) errors.push('tiles: expected a list');
  if (errors.length) return { ok: false, errors };
  if (o.tiles.length > 16) errors.push('tiles: at most 16');
  const tiles = [];
  o.tiles.forEach((t, i) => {
    if (!t || typeof t !== 'object') { errors.push(`tiles[${i}]: expected an object`); return; }
    if (!TILE_TYPES.includes(t.type)) { errors.push(`tiles[${i}].type: unknown ${t.type}`); return; }
    const span = Array.isArray(t.span) ? t.span : [1, 1];
    if (!SPANS.some(([a, b]) => a === span[0] && b === span[1])) { errors.push(`tiles[${i}].span: invalid`); return; }
    if (span[0] > cols || span[1] > rows) { errors.push(`tiles[${i}].span: larger than the grid`); return; }
    const props = t.props && typeof t.props === 'object' && !Array.isArray(t.props) ? t.props : {};
    tiles.push({ type: t.type, span: [span[0], span[1]], props: JSON.parse(JSON.stringify(props)) });
  });
  if (errors.length) return { ok: false, errors };
  const name = typeof o.name === 'string' ? o.name.slice(0, 80) : undefined;
  return { ok: true, layout: { v: LAYOUT_VERSION, cols, rows, tiles, ...(name ? { name } : {}) }, errors: [] };
}

export function emptyLayout(grid = '2x2') {
  const [cols, rows] = GRIDS[grid];
  return { v: 1, cols, rows, tiles: Array.from({ length: cols * rows }, () => ({ type: 'empty', span: [1, 1], props: {} })) };
}

/** Change the grid keeping tiles (extra cells become empty, overflow dropped). */
export function regrid(layout, grid) {
  const [cols, rows] = GRIDS[grid];
  const tiles = layout.tiles.map((t) => ({ ...t, span: [Math.min(t.span[0], cols), Math.min(t.span[1], rows)] }));
  let area = tiles.reduce((s, t) => s + t.span[0] * t.span[1], 0);
  while (tiles.length && area > cols * rows) { const t = tiles.pop(); area -= t.span[0] * t.span[1]; }
  while (area < cols * rows) { tiles.push({ type: 'empty', span: [1, 1], props: {} }); area++; }
  return { ...layout, cols, rows, tiles };
}

export function factoryLayouts() {
  return {
    'Monitor de entrenamiento': { v: 1, cols: 4, rows: 3, name: 'Monitor de entrenamiento', tiles: [
      { type: 'run-card', span: [1, 1], props: { run: '$latest' } },
      { type: 'curves', span: [2, 1], props: { runs: ['$latest'], cols: ['success', 'return'], x: 'env_steps', smooth: 0.3 } },
      { type: 'train-vs-test', span: [1, 1], props: { runs: ['$latest'] } },
      { type: 'replay-feed', span: [2, 2], props: { run: '$latest', order: 'latest', follow: true, count: 4 } },
      { type: 'curves', span: [1, 1], props: { runs: ['$latest'], cols: ['entropy'], x: 'env_steps', smooth: 0.3 } },
      { type: 'job-log', span: [1, 1], props: { job: '$latest' } },
      { type: 'end-reasons', span: [1, 1], props: { run: '$latest' } },
      { type: 'runs-table', span: [1, 1], props: {} },
    ] },
    'Muro de replays': { v: 1, cols: 4, rows: 3, name: 'Muro de replays', tiles: Array.from({ length: 12 }, (_, i) => (
      { type: 'replay-feed', span: [1, 1], props: { order: i % 2 ? 'random' : 'latest', count: 1, seed: i, follow: i === 0 } })) },
    'Comparar runs': { v: 1, cols: 3, rows: 2, name: 'Comparar runs', tiles: [
      { type: 'curves', span: [2, 1], props: { runs: ['$all'], cols: ['success'], x: 'env_steps', smooth: 0.3 } },
      { type: 'runs-table', span: [1, 1], props: {} },
      { type: 'train-vs-test', span: [2, 1], props: { runs: ['$all'] } },
      { type: 'level-success', span: [1, 1], props: { runs: ['$all'] } },
    ] },
  };
}

// -- URL encoding (deflate-raw + base64url when available) ------------------

const b64url = (bytes) => {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const unb64url = (str) => {
  const s = atob(str.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice((str.length + 3) % 4));
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
};

async function pipe(bytes, stream) {
  const res = new Response(new Blob([bytes]).stream().pipeThrough(stream));
  return new Uint8Array(await res.arrayBuffer());
}

export async function encodeLayout(layout) {
  const json = new TextEncoder().encode(JSON.stringify(layout));
  if (typeof CompressionStream !== 'undefined') {
    return 'z' + b64url(await pipe(json, new CompressionStream('deflate-raw')));
  }
  return 'j' + b64url(json);
}

export async function decodeLayout(text) {
  if (!text) throw new Error('layout: empty');
  const kind = text[0];
  const bytes = unb64url(text.slice(1));
  let json;
  if (kind === 'z') json = await pipe(bytes, new DecompressionStream('deflate-raw'));
  else if (kind === 'j') json = bytes;
  else throw new Error('layout: unknown encoding');
  const res = validateLayout(JSON.parse(new TextDecoder().decode(json)));
  if (!res.ok) throw new Error(res.errors.join('; '));
  return res.layout;
}
