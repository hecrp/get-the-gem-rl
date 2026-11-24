// Client-side interval helpers (same formulas as dashboard/stats.py).
const Z = 1.959963984540054;
export function wilson(k, n) {
  if (!n) return { k, n, p: null, lo: null, hi: null };
  const p = k / n, d = 1 + (Z * Z) / n, c = (p + (Z * Z) / (2 * n)) / d;
  const h = (Z * Math.sqrt((p * (1 - p)) / n + (Z * Z) / (4 * n * n))) / d;
  return { k, n, p, lo: Math.max(0, c - h), hi: Math.min(1, c + h) };
}
export function newcombe(k1, n1, k2, n2) {
  if (!n1 || !n2) return { d: null, lo: null, hi: null };
  const a = wilson(k1, n1), b = wilson(k2, n2), d = a.p - b.p;
  return { d, lo: d - Math.hypot(a.p - a.lo, b.hi - b.p), hi: d + Math.hypot(a.hi - a.p, b.p - b.lo), n1, n2 };
}
