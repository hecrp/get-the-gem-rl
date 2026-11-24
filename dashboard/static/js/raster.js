// Pixel rules shared with the pygame renderer.
// Positions round with floor(v + 0.5); polygons are filled with pygame-ce's
// scanline rule, ported exactly (verified against pygame in
// tests/dashboard_js/raster.test.mjs with fixtures rendered by pygame).

export const roundHalfUp = (v) => Math.floor(v + 0.5);

/** [x, y, w, h] in pixels, each component rounded independently (like pygame.Rect). */
export function rectPx(x, y, w, h) {
  return [roundHalfUp(x), roundHalfUp(y), roundHalfUp(w), roundHalfUp(h)];
}

/** Horizontal spans [y, x0, x1] (inclusive) that pygame.draw.polygon fills. */
export function polygonSpans(points) {
  const n = points.length;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const spans = [];
  let miny = Math.min(...ys);
  let maxy = Math.max(...ys);
  if (miny === maxy) {
    // pygame-ce: one-row polygon spans the x range of all points but the last.
    const head = xs.slice(0, Math.max(1, n - 1));
    spans.push([miny, Math.min(...head), Math.max(...head)]);
    return spans;
  }
  for (let y = miny; y <= maxy; y++) {
    const inter = [];
    for (let i = 0; i < n; i++) {
      const ip = i ? i - 1 : n - 1;
      let y1 = ys[ip], y2 = ys[i], x1, x2;
      if (y1 < y2) { x1 = xs[ip]; x2 = xs[i]; }
      else if (y1 > y2) { y2 = ys[ip]; y1 = ys[i]; x2 = xs[ip]; x1 = xs[i]; }
      else continue;
      if ((y >= y1 && y < y2) || (y === maxy && y2 === maxy)) {
        const f = ((y - y1) * (x2 - x1)) / (y2 - y1);
        inter.push((i % 2 === 0 ? Math.floor(f) : Math.ceil(f)) + x1);
      }
    }
    inter.sort((a, b) => a - b);
    for (let i = 0; i + 1 < inter.length; i += 2) spans.push([y, inter[i], inter[i + 1]]);
  }
  for (let i = 0; i < n; i++) {
    const ip = i ? i - 1 : n - 1;
    const y = ys[i];
    if (miny < y && ys[ip] === y && y < maxy) {
      spans.push([y, Math.min(xs[i], xs[ip]), Math.max(xs[i], xs[ip])]);
    }
  }
  return spans;
}

/** Fill a polygon on a 2D context with pygame's pixels (offset ox, oy). */
export function fillPolygon(ctx, points, ox = 0, oy = 0) {
  for (const [y, x0, x1] of polygonSpans(points)) ctx.fillRect(x0 + ox, y + oy, x1 - x0 + 1, 1);
}

/** pygame.draw.rect(width=0). */
export function fillRect(ctx, x, y, w, h) {
  if (w > 0 && h > 0) ctx.fillRect(x, y, w, h);
}

/** pygame.draw.rect(width=1): the 1-px border inside the rectangle. */
export function strokeRect1(ctx, x, y, w, h) {
  if (w <= 0 || h <= 0) return;
  ctx.fillRect(x, y, w, 1);
  ctx.fillRect(x, y + h - 1, w, 1);
  ctx.fillRect(x, y, 1, h);
  ctx.fillRect(x + w - 1, y, 1, h);
}

/** Spike triangle for the segment at x of a hazard box (palette.spike_polygon). */
export function spikePolygon(x, top, bottom, rightEdge, shapes) {
  const right = Math.min(x + shapes.spike_span, rightEdge - 1);
  const tip = Math.min(x + shapes.spike_tip, right);
  return [[x, bottom - 1], [tip, top], [right, bottom - 1]];
}
