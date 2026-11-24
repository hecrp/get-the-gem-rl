// Formatting (Spanish UI): numbers, replay ids, outcomes, proportions with n.
const NBSP = ' ';

export const replayNumber = (n) => (Number.isInteger(n) ? '#' + String(n).padStart(4, '0') : `#${n}`);

export function fmtInt(n) {
  if (n == null || !Number.isFinite(n)) return '–';
  return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, NBSP);
}

export function fmtNum(v, digits = 2) {
  if (v == null || !Number.isFinite(v)) return '–';
  return v.toFixed(digits).replace('.', ',');
}

/** 1 230 000 -> "1,23 M"; 2 600 -> "2,6 k". */
export function fmtSteps(n) {
  if (n == null || !Number.isFinite(n)) return '–';
  const a = Math.abs(n);
  if (a >= 1e6) return fmtNum(n / 1e6, a >= 1e7 ? 1 : 2) + NBSP + 'M';
  if (a >= 1e3) return fmtNum(n / 1e3, a >= 1e4 ? 0 : 1) + NBSP + 'k';
  return String(Math.round(n));
}

export function fmtPct(p, digits = 0) {
  if (p == null || !Number.isFinite(p)) return '–';
  return fmtNum(100 * p, digits) + NBSP + '%';
}

/** Wilson block {p, lo, hi, n} -> "83 % [75–89] n=64" (never a % without n). */
export function fmtProp(w, digits = 0) {
  if (!w || w.n == null) return '–';
  if (!w.n) return 'n=0';
  return `${fmtPct(w.p, digits)} [${fmtNum(100 * w.lo, digits)}–${fmtNum(100 * w.hi, digits)}] n=${w.n}`;
}

const OUTCOMES = {
  won: ['✔', 'victoria', 'ok'], hazard: ['✘', 'pinchos', 'bad'], fall: ['✘', 'vacío', 'bad'],
  timeout: ['⏱', 'timeout', 'warn'], incomplete: ['…', 'incompleto', 'warn'],
};

/** {icon, text, cls} from an outcome block (end_reason first). */
export function outcomeLabel(outcome) {
  const key = outcome?.end_reason || (outcome?.result === 'incomplete' ? 'incomplete' : outcome?.result);
  const [icon, text, cls] = OUTCOMES[key] || ['?', key || 'desconocido', 'warn'];
  return { icon, text, cls, label: `${icon} ${text}` };
}

export function sourceLabel(source) {
  if (!source) return '–';
  switch (source.kind) {
    case 'checkpoint':
      return source.env_steps != null ? `checkpoint @ ${fmtSteps(source.env_steps)} pasos`
        : `checkpoint @ update ${source.update}`;
    case 'final': return 'política final';
    case 'reference': return 'solución de referencia';
    case 'scripted': return 'política scripted';
    case 'random': return 'política aleatoria';
    case 'human': return 'humano';
    default: return source.kind;
  }
}

export function levelLabel(level) {
  const ref = level?.ref || {};
  if (ref.kind === 'builtin') return `nivel ${ref.n}${level.name ? ' · ' + level.name : ''}`;
  if (ref.kind === 'pcg') return `${ref.preset || 'pcg'} · seed ${fmtInt(ref.seed)}`;
  if (ref.kind === 'custom') return `custom · ${String(level.hash || ref.hash || '').slice(7, 15)}`;
  return level?.name || '–';
}

export function fmtDuration(s) {
  if (s == null || !Number.isFinite(s) || s < 0) return '–';
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
