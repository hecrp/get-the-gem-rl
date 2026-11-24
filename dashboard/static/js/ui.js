export function toast(msg, ms = 3500) {
  const box = document.getElementById('toast');
  const el = document.createElement('div');
  el.className = 'msg'; el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => el.remove(), ms);
}
