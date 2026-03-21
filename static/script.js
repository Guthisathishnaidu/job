/* RoleMatch AI — script.js */

let selectedFiles = [];

// ── Word counter ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('paste-area');
  const ct = document.getElementById('ta-counter');
  if (ta && ct) {
    ta.addEventListener('input', () => {
      const w = ta.value.trim().split(/\s+/).filter(Boolean).length;
      ct.textContent = w + (w === 1 ? ' word' : ' words');
    });
  }
});

// ── Drag & Drop ──────────────────────────────────────────
function onDragOver(e) { e.preventDefault(); document.getElementById('dropzone').classList.add('over'); }
function onDragLeave()  { document.getElementById('dropzone').classList.remove('over'); }
function onDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('over');
  addFiles([...e.dataTransfer.files]);
}
function onFileSelected(input) { addFiles([...input.files]); }

function addFiles(newFiles) {
  const existing = new Set(selectedFiles.map(f => f.name));
  for (const f of newFiles) {
    if (!existing.has(f.name)) { selectedFiles.push(f); existing.add(f.name); }
  }
  renderPills();
  document.getElementById('paste-area').value = '';
  document.getElementById('ta-counter').textContent = '0 words';
}
function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderPills();
}
function renderPills() {
  const c = document.getElementById('file-pills');
  c.innerHTML = '';
  selectedFiles.forEach(f => {
    const p = document.createElement('div');
    p.className = 'file-pill';
    p.innerHTML = `
      <span class="file-pill-icon">📄</span>
      <span class="file-pill-name" title="${esc(f.name)}">${esc(f.name)}</span>
      <button class="file-pill-remove" onclick="removeFile('${esc(f.name).replace(/'/g,"\\'")}')">✕</button>
    `;
    c.appendChild(p);
  });
}

// ── Submit ───────────────────────────────────────────────
async function submitResume() {
  const pasteText = document.getElementById('paste-area').value.trim();
  const btn = document.getElementById('submit-btn');

  if (!selectedFiles.length && !pasteText) {
    showError('Please upload a resume file or paste your skills text first.');
    return;
  }

  setLoading(true);
  btn.classList.add('loading');
  document.getElementById('btn-label').textContent = 'Analysing…';
  animateSteps();

  try {
    let data;
    if (selectedFiles.length > 0) {
      const form = new FormData();
      selectedFiles.forEach(f => form.append('resumes', f));
      const res = await fetch('/api/recommend', { method: 'POST', body: form });
      data = await res.json();
    } else {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: pasteText }),
      });
      data = await res.json();
    }

    if      (data.error)            showError(data.error);
    else if (data.results_per_file) renderMulti(data.results_per_file);
    else                            renderSingle(data);

  } catch (err) {
    showError('Cannot reach the server. Make sure Flask is running on port 5000.');
    console.error(err);
  } finally {
    setLoading(false);
    btn.classList.remove('loading');
    document.getElementById('btn-label').textContent = 'Analyse & Recommend';
  }
}

// ── Animated loading steps ───────────────────────────────
function animateSteps() {
  const ids = ['step-1','step-2','step-3','step-4'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active','done');
  });
  if (document.getElementById('step-1'))
    document.getElementById('step-1').classList.add('active');

  let i = 0;
  const t = setInterval(() => {
    const el = document.getElementById(ids[i]);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
    i++;
    if (i >= ids.length) { clearInterval(t); return; }
    const next = document.getElementById(ids[i]);
    if (next) next.classList.add('active');
  }, 600);
}

// ── Render single ────────────────────────────────────────
function renderSingle(data) {
  const results = data.results || [];
  if (!results.length) { showError('No matching roles found. Add more detail to your resume.'); return; }
  showResults();
  setBest(results[0], '', data.resume_preview);
  const list = document.getElementById('results-list');
  list.innerHTML = '';
  buildRows(results, list);
  document.getElementById('rl-count').textContent = `${results.length} roles found`;
  document.getElementById('results-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Render multi ─────────────────────────────────────────
function renderMulti(perFile) {
  const first = perFile.find(r => r.results && r.results.length);
  if (!first) { showError(perFile.map(r => `${r.filename}: ${r.error}`).join(' | ')); return; }
  showResults();
  setBest(first.results[0], first.filename, first.resume_preview);
  const list = document.getElementById('results-list');
  list.innerHTML = '';
  let total = 0;
  for (const fr of perFile) {
    if (fr.error) {
      const d = document.createElement('div');
      d.style.cssText = 'padding:7px 4px;color:#f43f5e;font-size:.75rem;font-family:var(--mono)';
      d.textContent = `⚠ ${fr.filename}: ${fr.error}`;
      list.appendChild(d);
      continue;
    }
    if (perFile.filter(f => !f.error).length > 1) {
      const h = document.createElement('div');
      h.className = 'rfile-hdr';
      h.textContent = `📄 ${fr.filename}`;
      list.appendChild(h);
    }
    buildRows(fr.results, list);
    total += fr.results.length;
  }
  document.getElementById('rl-count').textContent = `${total} roles found`;
  document.getElementById('results-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Build result rows ────────────────────────────────────
function buildRows(results, container) {
  const max    = results[0].score;
  const medals = ['🥇','🥈','🥉'];
  const cls    = ['g','s','b'];

  results.forEach((r, i) => {
    const pct = max > 0 ? (r.score / max) * 100 : 0;
    const row = document.createElement('div');
    row.className = 'rrow';
    row.style.animationDelay = `${i * 0.055}s`;
    const rk = i < 3 ? medals[i] : `#${i+1}`;
    const rc = i < 3 ? cls[i]    : 'n';
    row.innerHTML = `
      <div class="rrank ${rc}">${rk}</div>
      <span class="rname">${esc(r.title)}</span>
      <span class="rscore">${(r.score*100).toFixed(1)}%</span>
      <div class="rbar" style="grid-column:2/4">
        <div class="rbar-fill" data-pct="${pct}"></div>
      </div>
    `;
    container.appendChild(row);
  });

  requestAnimationFrame(() => setTimeout(() => {
    container.querySelectorAll('.rbar-fill').forEach(b => { b.style.width = b.dataset.pct + '%'; });
  }, 80));
}

// ── Set best match card ──────────────────────────────────
function setBest(result, filename, preview) {
  document.getElementById('bc-title').textContent = result.title;
  document.getElementById('bc-pct').textContent   = (result.score * 100).toFixed(1) + '%';
  document.getElementById('bc-file').textContent  = filename ? `📄 ${filename}` : '';
  const prev = document.getElementById('bc-preview');
  if (preview) { prev.textContent = '▸ ' + preview; prev.style.display = ''; }
  else         { prev.style.display = 'none'; }
  const pct = Math.min(100, result.score * 180);
  setTimeout(() => { document.getElementById('bc-bar').style.width = pct + '%'; }, 100);
}

// ── State helpers ────────────────────────────────────────
function setLoading(on) {
  if (on) {
    hide('empty-state'); hide('error-state'); hide('results-inner');
    show('loading-state');
  } else {
    hide('loading-state');
  }
}
function showResults() {
  hide('empty-state'); hide('error-state'); hide('loading-state');
  show('results-inner');
}
function showError(msg) {
  hide('empty-state'); hide('loading-state'); hide('results-inner');
  document.getElementById('error-text').textContent = msg;
  show('error-state');
}
function show(id) { document.getElementById(id)?.classList.remove('hidden'); }
function hide(id) { document.getElementById(id)?.classList.add('hidden'); }
function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Keyboard shortcut ────────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') submitResume();
});