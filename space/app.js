/* Tahqiq static Space: reads board.json from the results dataset; submits via an HF discussion using OAuth. */
/* OAuth (PKCE) against huggingface.co, no library: HF injects window.huggingface.variables in static Spaces. */
const HFV = (window.huggingface && window.huggingface.variables) || {};
const OAUTH = { clientId: HFV.OAUTH_CLIENT_ID, scopes: (HFV.OAUTH_SCOPES || "openid profile write-discussions"), provider: (HFV.OPENID_PROVIDER_URL || "https://huggingface.co") };
const b64url = buf => btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const rand = n => b64url(crypto.getRandomValues(new Uint8Array(n)));
async function oauthLoginUrl() {
  if (!OAUTH.clientId) throw new Error("OAuth is not configured on this Space (hf_oauth: true in README)");
  const verifier = rand(32), state = rand(16);
  const challenge = b64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  localStorage.setItem("hf_pkce", JSON.stringify({ verifier, state }));
  const redirect = location.origin + location.pathname;
  const q = new URLSearchParams({ client_id: OAUTH.clientId, redirect_uri: redirect, response_type: "code", scope: OAUTH.scopes, state, code_challenge: challenge, code_challenge_method: "S256" });
  return `${OAUTH.provider}/oauth/authorize?${q}`;
}
async function oauthHandleRedirectIfPresent() {
  const u = new URL(location.href); const code = u.searchParams.get("code"), state = u.searchParams.get("state");
  if (!code) return null;
  const saved = JSON.parse(localStorage.getItem("hf_pkce") || "{}"); localStorage.removeItem("hf_pkce");
  if (!saved.verifier || saved.state !== state) throw new Error("OAuth state mismatch");
  const redirect = location.origin + location.pathname;
  const r = await fetch(`${OAUTH.provider}/oauth/token`, { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ grant_type: "authorization_code", code, redirect_uri: redirect, client_id: OAUTH.clientId, code_verifier: saved.verifier }) });
  if (!r.ok) throw new Error("token exchange failed: " + r.status);
  const tok = await r.json();
  let userInfo = {};
  try { userInfo = await fetch(`${OAUTH.provider}/oauth/userinfo`, { headers: { Authorization: `Bearer ${tok.access_token}` } }).then(x => x.json()); } catch {}
  return { accessToken: tok.access_token, accessTokenExpiresAt: new Date(Date.now() + (tok.expires_in || 3600) * 1000).toISOString(), userInfo };
}

const RESULTS_REPO = "HeshamHaroon/tahqiq-results";
const REQUESTS_REPO = "HeshamHaroon/tahqiq-requests";
const BOARD_URL = `https://huggingface.co/datasets/${RESULTS_REPO}/resolve/main/board.json`;
const DS_LABEL = { sada: "SADA", casablanca: "Casablanca", commonvoice: "CV 18", masc_clean: "MASC clean", masc_noisy: "MASC noisy", mgb2: "MGB-2" };
const DS_INFO = {
  sada: { n: 6193, hours: 10.75, elm: 6189, dialects: "najdi, gulf, hijazi, msa, egyptian" },
  casablanca: { n: 6818, hours: 7.85, elm: 6726, dialects: "maghrebi, levantine, egyptian, gulf, yemeni" },
  commonvoice: { n: 10471, hours: 12.66, elm: 10471, dialects: "mostly unlabelled" },
  masc_clean: { n: 8612, hours: 10.49, elm: 8612, dialects: "unlabelled" },
  masc_noisy: { n: 9394, hours: 8.92, elm: 9393, dialects: "unlabelled" },
  mgb2: { n: 5365, hours: 9.58, elm: 5365, dialects: "unlabelled (≈70% MSA broadcast)" },
};
const S = { slice: "full", norm: "elm", metric: "wer", ci: true, acc: false, excl: true, sort: "avg", dir: 1 };
const contam = (m, d) => ((board.contamination || {}).cells || {})[m] && board.contamination.cells[m][d];
const pct = x => (S.acc ? (100 - 100 * x) : (100 * x)).toFixed(1);
const pctRaw = x => (100 * x).toFixed(1);
const fmtCI = c => S.acc ? `[${pct(c.hi)}, ${pct(c.lo)}]` : `[${pct(c.lo)}, ${pct(c.hi)}]`;
let board = { models: [], queue: [], generated_at: "" };

async function loadBoard() {
  try {
    board = await fetch(BOARD_URL + "?t=" + Date.now(), { cache: "no-store" }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  } catch (e) { board = { models: [], queue: [], generated_at: "" }; }
  const datasets = allDatasets();
  document.getElementById("meta-pills").innerHTML = [
    `${board.models.length} systems`, `${datasets.length} test sets`, `${board.queue.filter(q => ["PENDING", "QUEUED", "RUNNING", "APPROVED"].includes(q.status)).length} in queue`,
    board.generated_at ? `updated ${board.generated_at.slice(0, 16).replace("T", " ")} UTC` : "no results yet",
  ].map(t => `<span class="pill">${t}</span>`).join("");
  renderLB(); renderDialects(); renderProd(); renderDatasets(); renderQueue(); renderFoot();
}

function allDatasets() {
  const set = new Set();
  for (const m of board.models) for (const sl of Object.values(m.scores || {})) for (const d of Object.keys(sl.datasets || {})) set.add(d);
  return ["casablanca", "commonvoice", "masc_clean", "masc_noisy", "sada", "mgb2"].filter(d => set.has(d));
}

function rowsFor(datasets) {
  // Every submitted system shows in every view: if it was only run on the common slice, its
  // common-slice numbers are used and the row says so.
  return board.models.filter(m => m.scores && (m.scores[S.slice] || m.scores.api || m.scores.full)).map(m => {
    const used = m.scores[S.slice] ? S.slice : (m.scores.api ? "api" : "full");
    const sc = m.scores[used]; m._used = used; const per = {}; let sum = 0, n = 0;
    for (const d of datasets) { const c = sc.datasets[d] && sc.datasets[d][S.norm]; if (c) { per[d] = c[S.metric]; const k = contam(m.model_id, d); if (!(S.excl && k && k.declared)) { sum += c[S.metric].point; n++; } } }
    return { m, sc, per, avg: n ? sum / n : null, n, numeral: sc.probes?.numeral, entity: sc.probes?.entity, vs: sc.vs_baseline };
  });
}

function cell(ci, mid, d) {
  if (!ci) return `<td class="pending">·</td>`;
  const w = Math.min(1, ci.hi) * 60, l = Math.max(0, ci.lo) * 60;
  const bar = S.ci ? `<span class="bar"><b style="left:${l}px;width:${Math.max(1, w - l)}px"></b><i style="left:${Math.min(1, ci.point) * 60}px"></i></span>` : "";
  const k = mid ? contam(mid, d) : null;
  const flag = k ? `<sup class="flag ${k.declared ? "decl" : ""} ${k.anomaly ? "anom" : ""} ${k.fingerprint ? "fp" : ""}" title="${k.declared ? "in-domain (declared): " + (k.note || "") + " " : ""}${k.anomaly ? "anomalous advantage: residual " + k.residual + ", z " + k.z + ", expected WER " + (100 * k.expected_wer).toFixed(1) + " " : ""}${k.fingerprint ? "convention match: distance " + k.fp_distance + " vs typical " + k.fp_typical + ", z " + k.fp_z + " " : ""}${k.devgap ? "dev-vs-test gap " + k.dev_gap + " (dev WER " + (100 * (k.dev_wer_dev || 0)).toFixed(1) + "), z " + k.dev_z + " " : ""}${k.pertgap ? "perturbation gap: delta " + k.pert_delta + " vs own " + k.pert_own_median + ", z " + k.pert_z + " " : ""}${k.likelihood ? "likelihood probe: reference preferred in " + (100 * k.lik_ref_preferred_rate).toFixed(0) + "% of utterances vs own " + (100 * k.lik_own_median).toFixed(0) + "%" : ""}">${k.declared ? "D" : ""}${k.anomaly ? "A" : ""}${k.fingerprint ? "F" : ""}${k.devgap ? "G" : ""}${k.pertgap ? "P" : ""}${k.likelihood ? "L" : ""}</sup>` : "";
  return `<td class="num ${k && k.declared && S.excl ? "excluded" : ""}">${pct(ci.point)}${flag}${bar}${S.ci ? `<span class="ci">${fmtCI(ci)}</span>` : ""}</td>`;
}
const probeCell = p => p && p.n ? `<td class="num">${(100 * p.acc.point).toFixed(0)}%<span class="ci">n=${p.n}${S.ci ? " " + fmtCI(p.acc) : ""}</span></td>` : `<td class="pending">·</td>`;

function renderLB() {
  const datasets = allDatasets();
  const rows = rowsFor(datasets);
  const val = (r, k) => k === "rtfx" ? -(r.m.timing?.rtfx ?? -1) : k === "params" ? (r.m.profile?.params_m ?? 1e9) : k === "stream" ? (r.m.profile?.streaming ?? "zz") : k === "avg" ? (r.avg ?? 9) : k === "numeral" ? -(r.numeral?.acc.point ?? -1) : k === "entity" ? -(r.entity?.acc.point ?? -1) : k === "vs" ? (r.vs?.delta_wer.point ?? 9) : (r.per[k]?.point ?? 9);
  // systems whose output cannot be scored against Arabic references (output_note) always sort last
  const nc = r => (r.m.profile?.output_note ? 1 : 0);
  rows.sort((a, b) => nc(a) - nc(b) || (val(a, S.sort) > val(b, S.sort) ? S.dir : -S.dir));
  const keys = ["", "", "avg", ...datasets, "numeral", "entity", "vs", "rtfx", "params", "stream"];
  const th = ["#", "System", `Avg ${S.acc ? "accuracy" : S.metric.toUpperCase()}`, ...datasets.map(d => DS_LABEL[d] || d), "Numerals", "Entities", "Δ vs Whisper v3", "RTFx", "Params", "Streaming"];
  document.querySelector("#lb thead").innerHTML = "<tr>" + th.map((t, i) => `<th data-key="${keys[i]}" class="${keys[i] === S.sort ? "sorted" : ""}">${t}</th>`).join("") + "</tr>";
  document.querySelector("#lb tbody").innerHTML = rows.length ? rows.map((r, i) => {
    const d = r.vs ? 100 * r.vs.delta_wer.point : null;
    const vs = r.vs ? `<td class="num delta ${d < 0 ? "neg" : "pos"}">${d > 0 ? "+" : ""}${d.toFixed(1)}${S.ci ? `<span class="ci">${fmtCI(r.vs.delta_wer)} p=${r.vs.mcnemar.p_value.toExponential(1)}</span>` : ""}</td>` : `<td class="pending">baseline</td>`;
    return `<tr data-id="${r.m.model_id}"><td>${r.m.profile?.output_note ? "·" : i + 1}</td><td><span class="model">${r.m.display_name}</span>${r.m.profile?.output_note ? `<span class="kind nc" title="${esc(r.m.profile.output_note)}">not comparable: ${esc(r.m.profile.output_note.split(";")[0])}</span>` : ""}<span class="kind">${[r.m.submitted_by && r.m.submitted_by !== "maintainer" ? "community" : "", r.m._used !== S.slice ? (r.m._used === "api" ? "common slice only" : "full sets") : ""].filter(Boolean).join(" · ")}</span></td><td class="num"><b>${r.avg == null ? "·" : pct(r.avg)}</b><span class="ci">${r.n}/${datasets.length} sets</span></td>${datasets.map(d => cell(r.per[d], r.m.model_id, d)).join("")}${probeCell(r.numeral)}${probeCell(r.entity)}${vs}${prodCells(r.m)}</tr>`;
  }).join("") : `<tr><td colspan="${th.length}" class="pending">No results published yet. The first runs are in progress.</td></tr>`;
  document.getElementById("lb-sub").textContent = `${S.slice === "full" ? "full test sets" : "common stratified slice"} · ${S.norm} normalizer · ${S.acc ? "accuracy = 100 − " + S.metric.toUpperCase() : S.metric.toUpperCase()} %`;
  document.querySelectorAll("#lb th[data-key]").forEach(h => h.onclick = () => { const k = h.dataset.key; if (!k) return; if (S.sort === k) S.dir = -S.dir; else { S.sort = k; S.dir = 1; } renderLB(); });
  document.querySelectorAll("#lb tbody tr[data-id]").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
}

function fmtParams(p) { if (!p) return "·"; return p >= 1000 ? (p / 1000).toFixed(1) + "B" : p + "M"; }
// A timing is "contended" when other processes actually used the GPU during the timed passes.
// New measurements sample their SM% (resident-but-idle neighbours count 0); older ones only counted processes.
function contended(t) {
  if (t.gpu_other_sm_pct_mean != null) return t.gpu_other_sm_pct_mean > 5;
  return t.gpu_other_processes > 0;
}
function contendNote(t) {
  return t.gpu_other_sm_pct_mean != null ? `other processes used ${t.gpu_other_sm_pct_mean}% of the GPU (mean) during measurement`
    : `${t.gpu_other_processes} other processes on the GPU during measurement`;
}
function gpuShort(t) { return (t.gpu || "").replace(/^NVIDIA\s+/, "").replace(/\s+\d+GB.*$/, ""); }
function prodCells(m) {
  const t = m.timing || {}, p = m.profile || {};
  const warn = contended(t) ? " (shared)" : "";
  return `<td class="num">${t.rtfx != null ? t.rtfx + "×" + warn : "·"}</td><td class="num">${fmtParams(p.params_m) === "·" && p.params_label ? p.params_label : fmtParams(p.params_m)}</td><td>${p.streaming && p.streaming !== "unknown" ? streamCell(p) : "·"}</td>`;
}
function esc(x) { return String(x).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;"); }
function streamCell(p) { return p.streaming_note ? `<span class="decl-note" title="${esc(p.streaming_note)}">${p.streaming}</span>` : p.streaming; }
// declared cell; values that are the architecture default (card silent) carry a mark explained under the table
function declCell(p, v) {
  if (v == null || v === "unknown" || v === "") return "unknown";
  const dflt = String(p.basis || "").startsWith("architecture");
  return dflt ? `${esc(v)}<span class="dflt" title="${esc(p.basis)}">*</span>` : esc(v);
}
function renderProd() {
  const rows = board.models.filter(m => m.scores && (m.scores[S.slice] || m.scores.api || m.scores.full))
    .sort((a, b) => (b.timing?.rtfx ?? -1) - (a.timing?.rtfx ?? -1));
  const th = ["System", "RTFx", "Latency p50 / p95 (s)", "Peak VRAM", "Load (s)", "Params", "Weights", "License", "Released", "Streaming", "Timestamps", "Max input", "Languages"];
  document.querySelector("#prod thead").innerHTML = "<tr>" + th.map((t, i) => `<th class="${i >= 1 && i <= 4 ? "measured" : i >= 5 ? "declared" : ""}">${t}</th>`).join("") + "</tr>";
  document.querySelector("#prod tbody").innerHTML = rows.length ? rows.map(m => {
    const t = m.timing || {}, p = m.profile || {};
    const warn = contended(t) ? ` <span class="ci" title="${contendNote(t)}">shared GPU</span>`
      : (t.gpu ? ` <span class="ci" title="${t.exclusive_gpu ? "one model per GPU; " : ""}${contendNote(t)}">${gpuShort(t)}</span>` : "");
    const has = t.rtfx != null;
    return `<tr data-id="${m.model_id}"><td><span class="model">${m.display_name}</span></td>
      <td class="num">${has ? t.rtfx + "×" + warn : "pending"}</td>
      <td class="num">${has ? t.latency_p50_s.toFixed(2) + " / " + t.latency_p95_s.toFixed(2) : "·"}</td>
      <td class="num">${t.peak_vram_gb != null ? t.peak_vram_gb + " GB" : "·"}</td>
      <td class="num">${has ? t.load_s : "·"}</td>
      <td class="num">${fmtParams(p.params_m) === "·" && p.params_label ? p.params_label : fmtParams(p.params_m)}</td>
      <td class="num">${p.weights_gb != null ? p.weights_gb + " GB" : "·"}</td>
      <td>${p.license || "·"}</td><td class="num">${p.created_at || "·"}</td>
      <td>${p.streaming && p.streaming !== "unknown" ? streamCell(p) : "unknown"}${String(p.basis || "").startsWith("architecture") ? `<span class="dflt" title="${esc(p.basis)}">*</span>` : ""}</td><td>${declCell(p, p.timestamps)}</td><td>${declCell(p, p.max_audio)}</td><td>${declCell(p, p.languages)}</td></tr>`;
  }).join("") : `<tr><td colspan="13" class="pending">No results yet.</td></tr>`;
  document.querySelectorAll("#prod tbody tr[data-id]").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
}

function renderDialects() {
  const rows = board.models.filter(m => m.scores && (m.scores[S.slice] || m.scores.api || m.scores.full));
  const pick = m => m.scores[S.slice] || m.scores.api || m.scores.full;
  const dialects = [...new Set(rows.flatMap(m => Object.keys(pick(m).dialects || {})))].filter(d => d !== "unknown").sort();
  document.querySelector("#dialects thead").innerHTML = "<tr><th>System</th>" + dialects.map(d => `<th>${d}</th>`).join("") + "</tr>";
  document.querySelector("#dialects tbody").innerHTML = rows.map(m => {
    const dd = pick(m).dialects || {};
    return `<tr data-id="${m.model_id}"><td><span class="model">${m.display_name}</span>${pick(m) !== m.scores[S.slice] ? '<span class="kind">common slice only</span>' : ''}</td>` + dialects.map(d => { const c = dd[d] && dd[d][S.norm]; return c ? `<td class="num">${pct(c.wer.point)}<span class="ci">n=${c.n}${S.ci ? " " + fmtCI(c.wer) : ""}</span></td>` : `<td class="pending">·</td>`; }).join("") + "</tr>";
  }).join("") || `<tr><td class="pending">No results yet.</td></tr>`;
  document.querySelectorAll("#dialects tbody tr[data-id]").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
}

function renderDatasets() {
  document.getElementById("datasets").innerHTML = Object.entries(DS_INFO).map(([d, c]) => `<div class="card"><h3>${DS_LABEL[d] || d}</h3><div class="kv">utterances <b>${c.n.toLocaleString()}</b> · hours <b>${c.hours}</b></div><div class="kv">matched to Elm <b>${c.elm.toLocaleString()}</b></div><div class="kv">dialect labels: ${c.dialects}</div></div>`).join("");
}

function renderQueue() {
  const q = [...board.queue].sort((a, b) => (b.submitted_at || "").localeCompare(a.submitted_at || ""));
  document.querySelector("#queue tbody").innerHTML = q.length ? q.map(r => `<tr><td><a href="https://huggingface.co/${r.hf_id}" target="_blank" rel="noopener">${r.hf_id}</a></td><td><span class="status s-${r.status}">${r.status}</span></td><td class="num">${(r.submitted_at || "").slice(0, 16).replace("T", " ")}</td><td>${r.submitted_by || ""}</td><td class="note">${(r.error || "").slice(0, 160)}</td></tr>`).join("") : `<tr><td colspan="5" class="pending">Queue is empty.</td></tr>`;
}

function renderFoot() {
  const first = board.models[0];
  document.getElementById("foot").innerHTML = `${first ? `manifest ${first.manifest_sha256} · bootstrap ${first.n_boot} · baseline ${first.baseline} · ` : ""}<a href="https://huggingface.co/datasets/${RESULTS_REPO}">results</a> · <a href="https://huggingface.co/datasets/${REQUESTS_REPO}/discussions">queue</a> · <a href="https://github.com/h9-tec/tahqiq">code</a>`;
}

function openCard(id) {
  const m = board.models.find(x => x.model_id === id); if (!m) return;
  const c = m.card || {}, run = m.run || {};
  const cov = m.scores[S.slice] ? Object.entries(m.scores[S.slice].coverage).map(([d, v]) => `${DS_LABEL[d] || d} ${v.scored}/${v.n}`).join(" · ") : "";
  document.querySelector("#card .card-body").innerHTML = `<h3>${m.display_name}</h3><dl>
    <dt>weights</dt><dd>${m.hf_id ? `<a href="https://huggingface.co/${m.hf_id}" target="_blank" rel="noopener">${m.hf_id}</a>` : "·"}</dd>
    <dt>org · license</dt><dd>${c.org || "·"} · ${c.license || "·"}${c.params ? " · " + c.params : ""}</dd>
    <dt>submitted by</dt><dd>${m.submitted_by || "maintainer"}</dd>
    <dt>language hint</dt><dd>${run.language_hint ?? "·"}</dd>
    <dt>decode</dt><dd>${JSON.stringify(run.decode_params || {})}</dd>
    <dt>env</dt><dd>${JSON.stringify(run.env || {})}</dd>
    <dt>run</dt><dd>${run.run_id || "·"} · ${(run.finished || "").slice(0, 16)} · code ${run.git_sha ? run.git_sha.slice(0, 8) : "·"}</dd>
    <dt>coverage</dt><dd>${cov}</dd>
    <dt>outputs</dt><dd><a href="https://huggingface.co/datasets/${RESULTS_REPO}/tree/main/results/${m.model_id}" target="_blank" rel="noopener">results/${m.model_id}/</a></dd>
  </dl>`;
  document.getElementById("card").showModal();
}

/* ---------- tabs + controls ---------- */
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".tab").forEach(s => s.classList.toggle("on", s.id === "tab-" + b.dataset.tab));
  location.hash = b.dataset.tab;
});
if (location.hash && document.querySelector(`.tabs button[data-tab="${location.hash.slice(1)}"]`)) document.querySelector(`.tabs button[data-tab="${location.hash.slice(1)}"]`).click();
document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
  const k = Object.keys(b.dataset)[0]; S[k] = b.dataset[k];
  b.parentElement.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  renderLB(); renderDialects(); renderProd();
});
document.getElementById("show-ci").onchange = e => { S.ci = e.target.checked; renderLB(); renderDialects(); };
document.getElementById("show-acc").onchange = e => { S.acc = e.target.checked; S.dir = S.acc ? -1 : 1; renderLB(); renderDialects(); };
document.getElementById("excl-decl").onchange = e => { S.excl = e.target.checked; renderLB(); };

/* ---------- OAuth submission ---------- */
let oauth = null;
try { oauth = JSON.parse(localStorage.getItem("hf_oauth") || "null"); if (oauth && Date.parse(oauth.accessTokenExpiresAt) < Date.now()) oauth = null; } catch { oauth = null; }
loadBoard();
try {
  const fresh = await oauthHandleRedirectIfPresent();
  if (fresh) { oauth = fresh; try { localStorage.setItem("hf_oauth", JSON.stringify(fresh)); } catch {} history.replaceState(null, "", location.pathname + "#submit"); document.querySelector('.tabs button[data-tab="submit"]').click(); }
} catch (e) { console.warn("oauth redirect handling failed", e); document.getElementById("submit-msg").textContent = "Sign-in failed: " + e.message; }

function showAuth() {
  const ok = !!oauth;
  document.getElementById("auth").hidden = ok;
  document.getElementById("submit-form").hidden = !ok;
  if (ok) document.getElementById("who").textContent = oauth.userInfo?.name || oauth.userInfo?.preferred_username || "you";
}
document.getElementById("login").onclick = async () => { try { location.href = await oauthLoginUrl(); } catch (e) { document.querySelector("#auth .note").textContent = e.message; } };
document.getElementById("submit-form").onsubmit = async e => {
  e.preventDefault();
  const f = new FormData(e.target); const hf_id = (f.get("hf_id") || "").trim();
  const msg = document.getElementById("submit-msg"); msg.textContent = "Checking model…";
  try {
    const r = await fetch(`https://huggingface.co/api/models/${hf_id}`); if (!r.ok) throw new Error(`model ${hf_id} not found or not public`);
    msg.textContent = "Opening request…";
    const body = JSON.stringify({ display_name: f.get("display_name") || "", notes: f.get("notes") || "", submitted_via: "space" });
    const res = await fetch(`https://huggingface.co/api/datasets/${REQUESTS_REPO}/discussions`, {
      method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${oauth.accessToken}` },
      body: JSON.stringify({ title: `submit: ${hf_id}`, description: "```json\n" + body + "\n```\n\nSubmitted through the Tahqiq Space." }),
    });
    if (!res.ok) throw new Error(`HF API ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const d = await res.json();
    msg.innerHTML = `Submitted. Track it here: <a href="https://huggingface.co/datasets/${REQUESTS_REPO}/discussions/${d.num}" target="_blank" rel="noopener">request #${d.num}</a>. The worker picks it up within a few minutes.`;
    e.target.reset();
  } catch (err) { msg.textContent = "Could not submit: " + err.message; }
};
showAuth();
setInterval(loadBoard, 5 * 60 * 1000);
