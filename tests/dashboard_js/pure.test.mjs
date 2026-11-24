import { test } from 'node:test';
import assert from 'node:assert/strict';
import { expandRLE, actionAtEngineStep, envIndexOfEngineStep } from '../../dashboard/static/js/rle.js';
import { advance, nextSpeed, Clock, STEPS_PER_SECOND } from '../../dashboard/static/js/clock.js';
import { validateLayout, factoryLayouts, regrid, emptyLayout, encodeLayout, decodeLayout } from '../../dashboard/static/js/layout.js';
import { replayNumber, fmtSteps, fmtProp, outcomeLabel, sourceLabel, levelLabel, fmtInt } from '../../dashboard/static/js/format.js';

test('expandRLE and engine-step mapping', () => {
  assert.deepEqual([...expandRLE([[5, 2], [3, 1]])], [5, 5, 3]);
  assert.deepEqual([...expandRLE([])], []);
  for (const bad of [[[6, 1]], [[1, 0]], [[1.5, 1]], [[1]]]) assert.throws(() => expandRLE(bad));
  const acts = expandRLE([[2, 1], [5, 2]]);
  assert.equal(actionAtEngineStep(acts, 0, 4), -1);
  assert.equal(actionAtEngineStep(acts, 1, 4), 2);
  assert.equal(actionAtEngineStep(acts, 4, 4), 2);
  assert.equal(actionAtEngineStep(acts, 5, 4), 5);
  assert.equal(actionAtEngineStep(acts, 12, 4), 5);
  assert.equal(actionAtEngineStep(acts, 13, 4), -1);
  assert.equal(envIndexOfEngineStep(9, 4), 2);
});

test('clock: 1x is 60 engine steps per second, speeds clamp', () => {
  assert.equal(STEPS_PER_SECOND, 60);
  assert.equal(advance(0, 250, 1), 15);
  assert.equal(advance(10, 100, 4), 34);
  assert.equal(advance(0, 5000, 1), 15);              // big gaps clamped to 250 ms
  assert.equal(nextSpeed(1, 1), 2);
  assert.equal(nextSpeed(16, 1), 16);
  assert.equal(nextSpeed(0.25, -1), 0.25);
  const c = new Clock();
  c.speed = 2; c.tick(100); assert.equal(c.t, 12);
  c.paused = true; c.tick(100); assert.equal(c.t, 12);
});

test('layouts: factory layouts are valid, bad ones rejected, v0 migrated', () => {
  for (const [name, l] of Object.entries(factoryLayouts())) {
    const r = validateLayout(l);
    assert.ok(r.ok, name + ': ' + r.errors.join(','));
  }
  assert.equal(factoryLayouts()['Muro de replays'].tiles.length, 12);
  assert.ok(!validateLayout({ v: 2, cols: 1, rows: 1, tiles: [] }).ok);
  assert.ok(!validateLayout({ v: 1, cols: 5, rows: 1, tiles: [] }).ok);
  assert.ok(!validateLayout({ v: 1, cols: 2, rows: 2, tiles: [{ type: 'nope' }] }).ok);
  assert.ok(!validateLayout({ v: 1, cols: 1, rows: 1, tiles: [{ type: 'note', span: [2, 1] }] }).ok);
  const mig = validateLayout({ cols: 1, rows: 1, tiles: [{ type: 'note', props: { text: 'x' } }] });
  assert.ok(mig.ok && mig.layout.v === 1 && mig.layout.tiles[0].span[0] === 1);
  const g = regrid(emptyLayout('4x4'), '2x2');
  assert.equal(g.tiles.length, 4);
  assert.equal(regrid(emptyLayout('1x1'), '3x2').tiles.length, 6);
});

test('layouts round-trip through the URL encoding', async () => {
  const l = factoryLayouts()['Comparar runs'];
  const text = await encodeLayout(l);
  assert.match(text, /^[zj][A-Za-z0-9_-]+$/);
  assert.deepEqual(await decodeLayout(text), validateLayout(l).layout);
  await assert.rejects(decodeLayout('x123'));
});

test('formatters', () => {
  assert.equal(replayNumber(412), '#0412');
  assert.equal(fmtSteps(1230000), '1,23 M');
  assert.equal(fmtSteps(2600), '2,6 k');
  assert.equal(fmtInt(1000007), '1 000 007');
  assert.equal(fmtProp({ p: 0.83, lo: 0.75, hi: 0.89, n: 64 }), '83 % [75–89] n=64');
  assert.equal(fmtProp({ n: 0 }), 'n=0');
  assert.equal(outcomeLabel({ result: 'won', end_reason: 'won' }).label, '✔ victoria');
  assert.equal(outcomeLabel({ result: 'dead', end_reason: 'hazard' }).label, '✘ pinchos');
  assert.equal(outcomeLabel({ result: 'dead', end_reason: 'fall' }).label, '✘ vacío');
  assert.equal(outcomeLabel({ result: 'timeout', end_reason: 'timeout' }).label, '⏱ timeout');
  assert.equal(sourceLabel({ kind: 'checkpoint', env_steps: 1230000 }), 'checkpoint @ 1,23 M pasos');
  assert.equal(levelLabel({ ref: { kind: 'pcg', preset: 'tiny', seed: 1000007 } }), 'tiny · seed 1 000 007');
  assert.equal(levelLabel({ ref: { kind: 'builtin', n: 7 }, name: 'First big one' }), 'nivel 7 · First big one');
});
