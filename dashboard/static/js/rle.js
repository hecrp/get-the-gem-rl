// RLE helpers: replays store [[action, repeat], ...] over env steps.
export const ACTIONS = ['NOOP', 'LEFT', 'RIGHT', 'JUMP', 'LEFT_JUMP', 'RIGHT_JUMP'];
export const ACTION_SYMBOLS = ['·', '←', '→', '↑', '↖', '↗'];

/** Expand RLE pairs to a Uint8Array of env-step actions (validated). */
export function expandRLE(pairs) {
  let total = 0;
  for (const p of pairs) {
    if (!Array.isArray(p) || p.length !== 2) throw new Error('RLE: expected [action, repeat]');
    const [a, r] = p;
    if (!Number.isInteger(a) || a < 0 || a > 5) throw new Error(`RLE: invalid action ${a}`);
    if (!Number.isInteger(r) || r < 1) throw new Error(`RLE: invalid repeat ${r}`);
    total += r;
  }
  const out = new Uint8Array(total);
  let i = 0;
  for (const [a, r] of pairs) { out.fill(a, i, i + r); i += r; }
  return out;
}

/** Env-step index that drives engine step t (1-based), given frame_skip. */
export const envIndexOfEngineStep = (t, frameSkip) => Math.max(0, Math.floor((t - 1) / frameSkip));

/** Action applied at engine step t (1-based); -1 for t = 0 (the reset state). */
export function actionAtEngineStep(actions, t, frameSkip) {
  if (t <= 0) return -1;
  const i = envIndexOfEngineStep(t, frameSkip);
  return i < actions.length ? actions[i] : -1;
}
