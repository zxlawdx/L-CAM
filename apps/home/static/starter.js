/* Starter UI. The real bridge is window.pywebview.api (config/bridge.py). */
(() => {
  const root = document.getElementById('vela-starter');
  if (!root || root.dataset.starterMounted) return;
  root.dataset.starterMounted = "true";
  const $ = id => root.querySelector('#' + id);
  const events = new AbortController();
  let count = 0, disposed = false, toastTimer, ready = false;

  function applyTheme(dark) {
    root.classList.toggle('dark', dark);
    $('theme-toggle').setAttribute('aria-pressed', String(dark));
    $('theme-label').textContent = dark ? 'Light' : 'Dark';
    try { localStorage.setItem('vela-starter-theme', dark ? 'dark' : 'light'); }
    catch { /* A disabled storage backend must not break the interface. */ }
  }

  function showToast(message) {
    $('toast').textContent = message;
    $('toast').classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('show'), 3500);
  }

  function connection(state) {
    ready = state === 'ready';
    $('runtime-status').dataset.state = state;
    $('ready-pill').dataset.state = state;
    $('runtime-status').textContent = ready ? 'Python conectado' : state === 'waiting' ? 'Conectando ao Python…' : 'Python indisponível';
    $('ready-label').textContent = ready ? 'Application running' : 'Bridge não confirmada';
    $('bridge-status').textContent = ready ? 'Python ↔ JavaScript ready' : 'Verifique a conexão com o runtime.';
    $('increment-button').disabled = !ready;
    $('retry-bridge').hidden = state !== 'error';
  }

  async function call(method, ...args) {
    const api = window.pywebview?.api;
    if (typeof api?.[method] !== 'function') throw new Error('Execute este projeto no runtime desktop do Vela.');
    let timer;
    try {
      return await Promise.race([
        api[method](...args),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('O Python não respondeu em 5 segundos.')), 5000); }),
      ]);
    } finally { clearTimeout(timer); }
  }

  async function connect() {
    connection('waiting');
    try {
      const pong = await call('ping');
      if (disposed) return;
      if (pong !== 'pong') throw new Error('A bridge retornou uma resposta inesperada.');
      connection('ready');
      $('counter-error').textContent = '';
    } catch (error) {
      if (disposed) return;
      connection('error');
      $('counter-error').textContent = error.message;
      console.error('[Vela starter] Bridge:', error);
    }
  }

  async function incrementCounter() {
    if (!ready || $('increment-button').disabled) return;
    $('increment-button').disabled = true;
    $('increment-button').textContent = 'Incrementando…';
    $('counter-error').textContent = '';
    try {
      // No local fallback: Python must produce the new value.
      const value = await call('increment_counter', count);
      if (disposed) return;
      if (!Number.isSafeInteger(value) || value !== count + 1) throw new Error('Resposta inválida do contador Python.');
      count = value;
      $('counter-value').textContent = String(count);
    } catch (error) {
      if (disposed) return;
      connection('error');
      $('counter-error').textContent = error.message;
      console.error('[Vela starter] Counter:', error);
    } finally {
      if (!disposed) {
        $('increment-button').disabled = !ready;
        $('increment-button').textContent = 'Increment';
      }
    }
  }

  async function starterAction(event) {
    event.preventDefault();
    try {
      const result = await call('starter_action', event.currentTarget.dataset.action);
      if (disposed) return;
      if (!result?.ok) throw new Error(result?.error || 'Não foi possível abrir o recurso.');
    } catch (error) {
      if (!disposed) { showToast(error.message); console.error('[Vela starter] Action:', error); }
    }
  }

  const listen = (node, name, handler) => node.addEventListener(name, handler, { signal: events.signal });
  let dark = false;
  try { dark = localStorage.getItem('vela-starter-theme') === 'dark'; } catch {}
  applyTheme(dark);
  listen($('theme-toggle'), 'click', () => applyTheme(!root.classList.contains('dark')));
  listen($('increment-button'), 'click', incrementCounter);
  listen($('retry-bridge'), 'click', connect);
  root.querySelectorAll('[data-action]').forEach(node => listen(node, 'click', starterAction));
  listen(window, 'pywebviewready', connect);
  window.VelaPage = {
    beforeUnmount() { disposed = true; events.abort(); clearTimeout(toastTimer); },
  };
  connect();
})();
