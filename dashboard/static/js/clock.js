// One global requestAnimationFrame ticker; time is measured in engine steps.
export const STEPS_PER_SECOND = 60;          // 1x = 60 engine steps/s
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16];

/** Advance a time t (engine steps) by dtMs at speed; pure. */
export function advance(t, dtMs, speed) {
  return t + (Math.min(dtMs, 250) / 1000) * STEPS_PER_SECOND * speed;
}

export function nextSpeed(speed, dir) {
  const i = SPEEDS.indexOf(speed);
  const j = Math.max(0, Math.min(SPEEDS.length - 1, (i < 0 ? 2 : i) + dir));
  return SPEEDS[j];
}

/** A clock shared by a sync group (or owned by one player). */
export class Clock {
  constructor() { this.t = 0; this.speed = 1; this.paused = false; }
  tick(dtMs, globalSpeed = 1) {
    if (!this.paused) this.t = advance(this.t, dtMs, this.speed * globalSpeed);
    return this.t;
  }
}

class Ticker {
  constructor() {
    this.subs = new Set(); this.last = null; this.running = false;
    this.globalSpeed = 1; this.globalPaused = false; this.fps = 0; this._frames = 0; this._fpsT = 0; this.workMs = 0;
  }
  add(fn) { this.subs.add(fn); this.start(); return () => this.subs.delete(fn); }
  start() {
    if (this.running || typeof requestAnimationFrame === 'undefined') return;
    this.running = true;
    const loop = (now) => {
      const dt = this.last == null ? 0 : now - this.last;
      this.last = now;
      this._frames++; this._fpsT += dt;
      if (this._fpsT >= 1000) { this.fps = (this._frames * 1000) / this._fpsT; this._frames = 0; this._fpsT = 0; }
      const eff = this.globalPaused ? 0 : dt;
      const w0 = performance.now();
      for (const fn of this.subs) { try { fn(eff, this.globalSpeed, dt); } catch (e) { console.error(e); } }
      // Main-thread work per frame (EMA): the frame budget the wall needs.
      this.workMs = 0.95 * this.workMs + 0.05 * (performance.now() - w0);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}

export const ticker = new Ticker();
