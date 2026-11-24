// Canvas port of rendering/pygame_renderer.py using the served palette (R-08).
import { roundHalfUp, rectPx, fillPolygon, strokeRect1, spikePolygon } from './raster.js';

export let THEME = null;
export const FLAG = { GROUNDED: 1, GEM: 2, DEAD: 4, WON: 8, EXIT_OPEN: 16 };
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

export async function loadTheme(getJSON) {
  THEME = await getJSON('/api/theme');
  THEME.css = Object.fromEntries(Object.entries(THEME.colors).map(([k, v]) => [k, rgb(v)]));
  return THEME;
}

const statics = new Map();      // level hash -> canvas (the _static_layer equivalent)

export function staticLayer(level) {
  let c = statics.get(level.hash);
  if (c) return c;
  const { css, shapes } = THEME;
  c = document.createElement('canvas');
  c.width = level.width; c.height = level.height;
  const ctx = c.getContext('2d');
  ctx.fillStyle = css.background;
  ctx.fillRect(0, 0, c.width, c.height);
  for (const p of level.platforms) {
    const [x, y, w, h] = rectPx(...p);
    ctx.fillStyle = css.platform; ctx.fillRect(x, y, w, h);
    ctx.fillStyle = css.platform_top; ctx.fillRect(x, y, w, Math.min(shapes.platform_top_h, h));
  }
  ctx.fillStyle = css.hazard;
  for (const hz of level.hazards) {
    const [x0, top, w, h] = rectPx(...hz);
    const right = x0 + w, bottom = top + h;
    for (let x = x0; x < right; x += shapes.spike_period) fillPolygon(ctx, spikePolygon(x, top, bottom, right, shapes));
  }
  if (statics.size > 64) statics.delete(statics.keys().next().value);
  statics.set(level.hash, c);
  return c;
}

/** Draw one frame exactly like PygameRenderer.draw. ``s`` = {x, y, flags}. */
export function drawFrame(ctx, level, s, camX, camY, cw, ch) {
  const { css, shapes } = THEME;
  ctx.fillStyle = css.void;
  ctx.fillRect(0, 0, cw, ch);
  ctx.drawImage(staticLayer(level), -camX, -camY);
  const collectedFlag = (s.flags & FLAG.GEM) !== 0;
  if (level.exit) {
    let [dx, dy, dw, dh] = rectPx(...level.exit);
    dx -= camX; dy -= camY;
    const decorative = level.win_condition === 'gem';
    const collected = collectedFlag && !decorative;
    const opened = (s.flags & FLAG.EXIT_OPEN) !== 0 && !decorative;
    const edge = collected ? css.gem : css.exit_edge;
    ctx.fillStyle = opened ? css.exit_interior : css.exit;
    ctx.fillRect(dx, dy, dw, dh);
    ctx.fillStyle = edge;
    strokeRect1(ctx, dx, dy, dw, dh);
    if (!opened) fillPolygon(ctx, shapes.small_gem_polygon, dx + shapes.door_key_offset[0], dy + shapes.door_key_offset[1]);
  }
  if (!collectedFlag) {
    ctx.fillStyle = css.gem;
    fillPolygon(ctx, shapes.gem_polygon, roundHalfUp(level.gem[0]) - camX, roundHalfUp(level.gem[1]) - camY);
  }
  const px = roundHalfUp(s.x) - camX, py = roundHalfUp(s.y) - camY;
  ctx.fillStyle = (s.flags & FLAG.DEAD) ? css.player_dead : (s.flags & FLAG.WON) ? css.player_won : css.player;
  ctx.fillRect(px, py, shapes.player_size[0], shapes.player_size[1]);
  if (collectedFlag) {
    ctx.fillStyle = css.gem;
    fillPolygon(ctx, shapes.small_gem_polygon, px + shapes.carried_gem_offset[0], py + shapes.carried_gem_offset[1]);
  }
}

/** A ghost: outlined, translucent player in its own colour (not part of pygame's frame). */
export function drawGhost(ctx, x, y, camX, camY, color, dead) {
  const px = roundHalfUp(x) - camX, py = roundHalfUp(y) - camY;
  ctx.globalAlpha = 0.45;
  ctx.fillStyle = color;
  ctx.fillRect(px, py, 12, 16);
  ctx.globalAlpha = 1;
  ctx.fillStyle = dead ? '#ff4d4d' : color;
  strokeRect1(ctx, px, py, 12, 16);
}

/** Fading trail through the hitbox centres of the last ``n`` states. */
export function drawTrail(ctx, xy, i, n, camX, camY, color) {
  const start = Math.max(0, i - n);
  if (i - start < 2) return;
  ctx.lineWidth = 1;
  for (let k = start + 1; k <= i; k++) {
    ctx.globalAlpha = 0.15 + 0.6 * ((k - start) / (i - start));
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(xy[2 * (k - 1)] + 6 - camX + 0.5, xy[2 * (k - 1) + 1] + 8 - camY + 0.5);
    ctx.lineTo(xy[2 * k] + 6 - camX + 0.5, xy[2 * k + 1] + 8 - camY + 0.5);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

/** Thumbnail: a single frame (index i) at world resolution. */
export function frameCanvas(level, traj, i) {
  const c = document.createElement('canvas');
  c.width = level.width; c.height = level.height;
  drawFrame(c.getContext('2d'), level, { x: traj.xy[2 * i], y: traj.xy[2 * i + 1], flags: traj.flags[i] }, 0, 0, c.width, c.height);
  return c;
}
