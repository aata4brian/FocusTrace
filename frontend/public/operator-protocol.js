(() => {
  if (location.pathname !== '/operator') return;

  const PRE = ['welcome', 'positioning', 'activity', 'calibration'];
  const POST = ['finish', 'debrief', 'closing'];
  let protocol = null;
  let state = null;
  let readyStep = Number(sessionStorage.getItem('tracefokus-protocol-ready-step') || 0);
  let transitionStage = 0;
  let transitionKey = '';
  let transitionAcknowledged = new Set(JSON.parse(sessionStorage.getItem('tracefokus-protocol-transitions') || '[]'));
  let stopAcknowledged = false;
  let postStep = 0;
  let loadedPostSession = '';
  let gate = null;

  const token = () => sessionStorage.getItem('tracefokus-operator') || '';
  const authHeaders = () => ({Authorization: `Bearer ${token()}`});
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
  const saveTransitions = () => sessionStorage.setItem('tracefokus-protocol-transitions', JSON.stringify([...transitionAcknowledged]));
  const sessionId = () => state?.session_id || 'NO_SESSION';
  const transitionAckKey = () => `${sessionId()}:${state?.revision ?? 0}:${state?.next_block || ''}`;
  const stopAckKey = () => `tracefokus-protocol-stop-ack:${sessionId()}:${state?.revision ?? 0}`;
  const postStepKey = () => `tracefokus-protocol-post-step:${sessionId()}`;
  const postDoneKey = () => `tracefokus-protocol-post-done:${sessionId()}`;
  const postDone = () => Boolean(state?.session_id && sessionStorage.getItem(postDoneKey()) === '1');

  function itemHtml(key, item, badge) {
    if (!item) return '';
    const checks = Array.isArray(item.checklist) && item.checklist.length
      ? `<ul>${item.checklist.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '';
    return `<div class="tf-gate-card">
      <div class="tf-gate-top"><span>${esc(badge || 'PANDUAN PENELITI')}</span><b>${esc(item.title || key)}</b></div>
      ${item.say ? `<div class="tf-read"><small>UCAPKAN KEPADA PESERTA</small><p>“${esc(item.say)}”</p></div>` : ''}
      ${checks ? `<div class="tf-checks"><small>CHECKLIST PENELITI</small>${checks}</div>` : ''}
    </div>`;
  }

  function ensureUI() {
    if (gate) return;
    const style = document.createElement('style');
    style.textContent = `
      #tf-research-gate{position:fixed;z-index:92;right:22px;bottom:22px;width:min(620px,calc(100vw - 44px));max-height:min(78vh,760px);overflow:auto;background:#f7f8f4;border:1px solid #cfd9cf;border-radius:6px;box-shadow:0 20px 70px #18362235;font:12px Inter,"Segoe UI",Arial,sans-serif;color:#294338}
      #tf-research-gate.hidden{display:none}.tf-gate-wrap{padding:17px}.tf-gate-head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start;margin-bottom:12px}.tf-gate-head>div>span{display:block;font-size:9px;letter-spacing:1.5px;color:#73887a;font-weight:700;margin-bottom:4px}.tf-gate-head h2{font-size:16px;margin:0;color:#29483b}.tf-gate-state{font-size:9px;background:#e8efe7;border:1px solid #d5e1d2;padding:5px 8px;border-radius:3px;color:#4d6b55;white-space:nowrap}
      .tf-gate-card{border:1px solid #dbe3da;background:white;border-radius:4px;padding:15px}.tf-gate-top span{font-size:9px;letter-spacing:1.25px;color:#7b8d82;display:block;margin-bottom:4px}.tf-gate-top b{font-size:14px;color:#2e4a3e}.tf-read{margin-top:12px;background:#edf3e9;border-left:3px solid #6f936b;padding:12px 13px}.tf-read small,.tf-checks small{font-size:8px;letter-spacing:1.2px;color:#718472;font-weight:700}.tf-read p{font-size:13px;line-height:1.65;color:#344f40;margin:7px 0 0}.tf-checks{margin-top:12px}.tf-checks ul{margin:7px 0 0;padding-left:18px}.tf-checks li{line-height:1.55;margin:3px 0;color:#56685c}
      .tf-gate-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:12px;flex-wrap:wrap}.tf-gate-actions button{border-radius:3px;min-height:42px;padding:10px 14px;font:600 11px Inter,"Segoe UI",Arial,sans-serif}.tf-gate-next{border:1px solid #315f4e;background:#315f4e;color:#fff}.tf-gate-secondary{border:1px solid #d3ddd1;background:#fff;color:#496452}.tf-gate-ok{border:1px solid #6f936b;background:#edf3e9;color:#35583d}.tf-gate-note{margin:10px 0 0;color:#74837a;font-size:10px;line-height:1.5}.tf-gate-ready{border-color:#9bb497;background:#f2f7ef}.tf-gate-lock{outline:2px solid #cfa880!important;outline-offset:2px!important}
      @media(max-width:800px){#tf-research-gate{right:10px;bottom:10px;width:calc(100vw - 20px);max-height:72vh}.tf-gate-wrap{padding:14px}.tf-read p{font-size:12px}}
    `;
    document.head.appendChild(style);
    gate = document.createElement('aside');
    gate.id = 'tf-research-gate';
    document.body.appendChild(gate);
  }

  function actionButtons() {
    return [...document.querySelectorAll('button')].filter(b => {
      if (b.closest('#tf-research-gate')) return false;
      const t = (b.textContent || '').trim().toUpperCase();
      return t.includes('START SESSION') || t.includes('MULAI BLOK BERIKUTNYA') || t.includes('STOP SESSION');
    });
  }

  function currentLock() {
    if (!state || !protocol) return null;
    const preparingNext = state.state === 'FINISHED' && postDone();
    if ((state.state === 'READY' || preparingNext) && readyStep < PRE.length) return 'start';
    if (state.state === 'TRANSITION' && state.next_block) {
      const key = transitionAckKey();
      if (!transitionAcknowledged.has(key)) return 'next';
    }
    if (state.state === 'TRANSITION' && !state.next_block && !stopAcknowledged) return 'stop';
    return null;
  }

  function decorateLockedAction() {
    const lock = currentLock();
    for (const b of actionButtons()) {
      const t = (b.textContent || '').toUpperCase();
      const isTarget = lock === 'start' ? t.includes('START SESSION') : lock === 'next' ? t.includes('MULAI BLOK BERIKUTNYA') : lock === 'stop' ? t.includes('STOP SESSION') : false;
      b.classList.toggle('tf-gate-lock', Boolean(isTarget));
      if (isTarget) b.title = 'Selesaikan panduan peneliti terlebih dahulu.';
      else if (b.title === 'Selesaikan panduan peneliti terlebih dahulu.') b.removeAttribute('title');
    }
  }

  document.addEventListener('click', event => {
    const button = event.target.closest?.('button');
    if (!button || button.closest('#tf-research-gate')) return;
    const text = (button.textContent || '').trim().toUpperCase();
    const lock = currentLock();
    const blocked = (lock === 'start' && text.includes('START SESSION')) || (lock === 'next' && text.includes('MULAI BLOK BERIKUTNYA')) || (lock === 'stop' && text.includes('STOP SESSION'));
    if (blocked) {
      event.preventDefault();
      event.stopPropagation();
      render();
      gate?.scrollTo({top:0,behavior:'smooth'});
    }
  }, true);

  function renderReady() {
    const key = PRE[Math.min(readyStep, PRE.length - 1)];
    const item = protocol[key];
    const nextParticipant = state?.state === 'FINISHED' && postDone();
    if (readyStep >= PRE.length) {
      return `<div class="tf-gate-wrap tf-gate-ready"><div class="tf-gate-head"><div><span>${nextParticipant?'PESERTA BERIKUTNYA':'SEBELUM SESI'}</span><h2>Semua instruksi awal selesai</h2></div><span class="tf-gate-state">SIAP START</span></div><p class="tf-gate-note">Jalankan pre-flight, konfirmasi persetujuan peserta, lalu tekan START SESSION. Kalibrasi dimulai segera setelah tombol ditekan.</p></div>`;
    }
    return `<div class="tf-gate-wrap"><div class="tf-gate-head"><div><span>${nextParticipant?'PESERTA BERIKUTNYA · ':''}TAHAP ${readyStep + 1} / ${PRE.length}</span><h2>Selesaikan sebelum START SESSION</h2></div><span class="tf-gate-state">BELUM START</span></div>${itemHtml(key,item,'BACA SEBELUM LANJUT')}<div class="tf-gate-actions"><button class="tf-gate-next" data-gate="ready-next">Sudah selesai · lanjut →</button></div><p class="tf-gate-note">START SESSION belum boleh ditekan sebelum empat tahap awal selesai.</p></div>`;
  }

  function renderTransition() {
    const next = state.next_block;
    const key = transitionAckKey();
    if (transitionKey !== key) { transitionKey = key; transitionStage = 0; }
    if (transitionAcknowledged.has(key)) {
      return `<div class="tf-gate-wrap tf-gate-ready"><div class="tf-gate-head"><div><span>TRANSITION</span><h2>Instruksi ${esc(next)} sudah dibacakan</h2></div><span class="tf-gate-state">SIAP NEXT</span></div><p class="tf-gate-note">Pastikan peserta dan perangkat siap, lalu tekan tombol “Mulai blok berikutnya” pada Session Console.</p></div>`;
    }
    if (transitionStage === 0) {
      return `<div class="tf-gate-wrap"><div class="tf-gate-head"><div><span>TRANSITION</span><h2>Jangan mulai ${esc(next)} dulu</h2></div><span class="tf-gate-state">VIDEO TETAP MEREKAM</span></div>${itemHtml('transition',protocol.transition,'1 · UCAPKAN TRANSISI')}<div class="tf-gate-actions"><button class="tf-gate-next" data-gate="transition-next">Transisi sudah dibacakan →</button></div></div>`;
    }
    return `<div class="tf-gate-wrap"><div class="tf-gate-head"><div><span>BLOK BERIKUTNYA</span><h2>Bacakan instruksi ${esc(next)}</h2></div><span class="tf-gate-state">${esc(next)}</span></div>${itemHtml(next,protocol[next],`2 · SEBELUM ${next}`)}<div class="tf-gate-actions"><button class="tf-gate-ok" data-gate="transition-ack">Instruksi ${esc(next)} sudah dibacakan ✓</button></div><p class="tf-gate-note">Setelah tombol ini ditekan, Session Console mengizinkan “Mulai blok berikutnya”.</p></div>`;
  }

  function renderStopGate() {
    if (stopAcknowledged) {
      return `<div class="tf-gate-wrap tf-gate-ready"><div class="tf-gate-head"><div><span>AKHIR BLOK KE-6</span><h2>Siap menutup sesi</h2></div><span class="tf-gate-state">SIAP STOP</span></div><p class="tf-gate-note">Tekan STOP SESSION pada Session Console. Setelah berkas ditutup, panduan akan lanjut ke penyelesaian, debriefing, dan penutupan.</p></div>`;
    }
    return `<div class="tf-gate-wrap"><div class="tf-gate-head"><div><span>AKHIR BLOK KE-6</span><h2>Penyelesaian eksperimen</h2></div><span class="tf-gate-state">SEBELUM STOP</span></div><div class="tf-gate-card"><div class="tf-gate-top"><span>CHECKLIST</span><b>Pastikan sesi siap ditutup</b></div><div class="tf-checks"><ul><li>Blok keenam sudah benar-benar selesai.</li><li>Jangan menutup browser atau terminal.</li><li>Siapkan STOP SESSION untuk menutup video dan berkas dengan aman.</li></ul></div></div><div class="tf-gate-actions"><button class="tf-gate-ok" data-gate="stop-ack">Siap · izinkan STOP SESSION ✓</button></div></div>`;
  }

  function renderPost() {
    const idx = Math.min(postStep, POST.length - 1);
    const key = POST[idx];
    const item = protocol[key];
    const last = idx === POST.length - 1;
    return `<div class="tf-gate-wrap"><div class="tf-gate-head"><div><span>SESI SUDAH DITUTUP · ${idx + 1}/${POST.length}</span><h2>${esc(item?.title || key)}</h2></div><span class="tf-gate-state">POST-SESSION</span></div>${itemHtml(key,item,last?'TAHAP TERAKHIR':'BACAKAN SEKARANG')}<div class="tf-gate-actions">${last?'<button class="tf-gate-ok" data-gate="post-finish">Penutupan selesai ✓</button>':'<button class="tf-gate-next" data-gate="post-next">Sudah dibacakan · lanjut →</button>'}</div></div>`;
  }

  function render() {
    if (!protocol || !state) return;
    ensureUI();
    if (state.state === 'READY') gate.innerHTML = renderReady();
    else if (state.state === 'TRANSITION' && state.next_block) gate.innerHTML = renderTransition();
    else if (state.state === 'TRANSITION' && !state.next_block) gate.innerHTML = renderStopGate();
    else if (state.state === 'FINISHED') gate.innerHTML = postDone() ? renderReady() : renderPost();
    else if (state.state === 'ABORTED' || state.state === 'INTERRUPTED') gate.innerHTML = `<div class="tf-gate-wrap"><div class="tf-gate-head"><div><span>SESI TIDAK SELESAI</span><h2>${esc(state.state)}</h2></div></div><p class="tf-gate-note">Periksa integritas data dan ikuti prosedur penutupan sesuai kebutuhan penelitian.</p></div>`;
    else gate.classList.add('hidden');
    if (['READY','TRANSITION','FINISHED','ABORTED','INTERRUPTED'].includes(state.state)) gate.classList.remove('hidden');
    decorateLockedAction();

    gate.querySelector('[data-gate="ready-next"]')?.addEventListener('click', () => {
      readyStep = Math.min(PRE.length, readyStep + 1);
      sessionStorage.setItem('tracefokus-protocol-ready-step', String(readyStep));
      render();
    });
    gate.querySelector('[data-gate="transition-next"]')?.addEventListener('click', () => { transitionStage = 1; render(); });
    gate.querySelector('[data-gate="transition-ack"]')?.addEventListener('click', () => {
      const k = transitionAckKey();
      transitionAcknowledged.add(k); saveTransitions(); render();
    });
    gate.querySelector('[data-gate="stop-ack"]')?.addEventListener('click', () => {
      stopAcknowledged = true;
      sessionStorage.setItem(stopAckKey(), '1');
      render();
    });
    gate.querySelector('[data-gate="post-next"]')?.addEventListener('click', () => {
      postStep = Math.min(POST.length - 1, postStep + 1);
      sessionStorage.setItem(postStepKey(), String(postStep));
      render();
    });
    gate.querySelector('[data-gate="post-finish"]')?.addEventListener('click', () => {
      sessionStorage.setItem(postDoneKey(), '1');
      sessionStorage.removeItem(postStepKey());
      sessionStorage.removeItem('tracefokus-protocol-transitions');
      transitionAcknowledged = new Set();
      stopAcknowledged = false;
      readyStep = 0;
      sessionStorage.setItem('tracefokus-protocol-ready-step', '0');
      render();
    });
  }

  async function loadProtocol() {
    if (!token()) return;
    try {
      const r = await fetch('/api/setup', {headers: authHeaders(), cache:'no-store'});
      if (!r.ok) return;
      const data = await r.json();
      protocol = data?.content?.operator_protocol || null;
    } catch (_) {}
  }

  async function loadState() {
    if (!token()) return;
    try {
      const r = await fetch('/api/state', {headers: authHeaders(), cache:'no-store'});
      if (!r.ok) return;
      const previous = state?.state;
      state = await r.json();
      if (previous === 'READY' && state.state === 'CALIBRATION') readyStep = PRE.length;
      if (state.state === 'TRANSITION' && !state.next_block) {
        stopAcknowledged = sessionStorage.getItem(stopAckKey()) === '1';
      } else {
        stopAcknowledged = false;
      }
      if (state.state === 'FINISHED' && loadedPostSession !== sessionId()) {
        loadedPostSession = sessionId();
        postStep = Number(sessionStorage.getItem(postStepKey()) || 0);
      }
      if (state.state !== 'TRANSITION') {
        transitionStage = 0;
        transitionKey = '';
      }
      render();
    } catch (_) {}
  }

  async function tick() {
    if (!protocol) await loadProtocol();
    await loadState();
  }

  setInterval(tick, 500);
  setTimeout(tick, 250);
})();
