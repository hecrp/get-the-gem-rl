import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { polygonSpans, roundHalfUp, rectPx } from '../../dashboard/static/js/raster.js';

// Pixels rendered by pygame-ce (tests/dashboard_js/make_fixtures.py).
const cases = JSON.parse(readFileSync(new URL('./fixtures/polygons.json', import.meta.url)));

function diff(points, pixels) {
  const got = new Set();
  for (const [y, x0, x1] of polygonSpans(points)) for (let x = x0; x <= x1; x++) got.add(`${x},${y}`);
  const want = new Set(pixels.map(([x, y]) => `${x},${y}`));
  return [...got].filter((p) => !want.has(p)).length + [...want].filter((p) => !got.has(p)).length;
}
const height = (pts) => Math.max(...pts.map((p) => p[1])) - Math.min(...pts.map((p) => p[1]));

test('roundHalfUp is floor(v + 0.5), not banker rounding', () => {
  assert.equal(roundHalfUp(0.5), 1);
  assert.equal(roundHalfUp(1.5), 2);
  assert.equal(roundHalfUp(2.5), 3);
  assert.equal(roundHalfUp(-0.5), 0);
  assert.equal(roundHalfUp(110.49), 110);
  assert.deepEqual(rectPx(1.5, 2.5, 12, 16), [2, 3, 12, 16]);
});

test('polygon fill is pixel-exact vs pygame for the renderer shapes up to 32 px tall', () => {
  const small = cases.filter((c) => height(c.points) <= 32);
  assert.ok(small.length > 250);
  for (const { points, pixels } of small) assert.equal(diff(points, pixels), 0, JSON.stringify(points));
});

test('taller spikes differ from pygame by at most one pixel (documented residual)', () => {
  const tall = cases.filter((c) => height(c.points) > 32);
  let off = 0;
  for (const { points, pixels } of tall) {
    const d = diff(points, pixels);
    assert.ok(d <= 1, JSON.stringify(points));
    off += d > 0;
  }
  assert.ok(off <= 8, `${off} tall shapes differ`);
});
