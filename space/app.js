/* Harfiyyan static Space: reads board.json from the results dataset; submits via an HF discussion using OAuth. */
import { oauthLoginUrl, oauthHandleRedirectIfPresent } from "https://esm.sh/@huggingface/hub@2.6.11";

const RESULTS_REPO = "HeshamHaroon/harfiyyan-results";
const REQUESTS_REPO = "HeshamHaroon/harfiyyan-requests";
const BOARD_URL = `https://huggingface.co/datasets/${RESULTS_REPO}/resolve/main/board.json`;
const DS_LABEL = { sada: "SADA", casablanca: "Casablanca", commonvoice: "CV 18", masc_clean: "MASC clean", masc_noisy: "MASC noisy", mgb2: "MGB-2" };
const DS_INFO = {
  sada: { n: 6193, hours: 10.75, elm: 6189, dialects: "najdi, gulf, hijazi, msa, egyptian" },
  casablanca: { n: 6818, hours: 7.85, elm: 6726, dialects: "maghrebi, levantine, egyptian, gulf, yemeni" },
  commonvoice: { n: 10471, hours: 12.66, elm: 10471, dialects: "mostly unlabelled" },
  masc_clean: { n: 8612, hours: 10.49, elm: 8612, dialects: "unlabelled" },
  masc_noisy: { n: 9394, hours: 8.92, elm: 9393, dialects: "unlabelled" },
};
const S = { slice: "full", norm: "elm", metric: "wer", ci: true, sort: "avg", dir: 1 };
const pct = x => (100 * x).toFixed(1);
const fmtCI = c => `[${pct(c.lo)}, ${pct(c.hi)}]`;
let board = { models: [], queue: [], generated_at: "" };

async function loadBoard() {
  try {
    board = await fetch(BOARD_URL + "?t=" + Date.now(), { cache: "no-store" }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  } catch (e) { board = { models: [], queue: [], generated_at: "" }; }
  const datasets = allDatasets();
  document.getElementById("meta-pills").innerHTML = [
    `${board.models.length} systems`, `${datasets.length} test sets`, `${board.queue.filter(q => ["PENDING", "RUNNING", "APPROVED"].includes(q.status)).length} in queue`,
    board.generated_at ? `updated ${board.generated_at.slice(0, 16).replace("T", " ")} UTC` : "no results yet",
  ].map(t => `<span class="pill">${t}</span>`).join("");
  renderLB(); renderDialects(); renderDatasets(); renderQueue(); renderFoot();
}

function allDatasets() {
  const set = new Set();
  for (const m of board.models) for (const sl of Object.values(m.scores || {})) for (const d of Object.keys(sl.datasets || {})) set.add(d);
  return ["casablanca", "commonvoice", "masc_clean", "masc_noisy", "sada", "mgb2"].filter(d => set.has(d));
}

function rowsFor(datasets) {
  return board.models.filter(m => m.scores && m.scores[S.slice]).map(m => {
    const sc = m.scores[S.slice]; const per = {}; let sum = 0, n = 0;
    for (const d of datasets) { const c = sc.datasets[d] && sc.datasets[d][S.norm]; if (c) { per[d] = c[S.metric]; sum += c[S.metric].point; n++; } }
    return { m, sc, per, avg: n ? sum / n : null, n, numeral: sc.probes?.numeral, entity: sc.probes?.entity, vs: sc.vs_baseline };
  });
}

function cell(ci) {
  if (!ci) return `<td class="pending">·</td>`;
  const w = Math.min(1, ci.hi) * 60, l = Math.max(0, ci.lo) * 60;
  const bar = S.ci ? `<span class="bar"><b style="left:${l}px;width:${Math.max(1, w - l)}px"></b><i style="left:${Math.min(1, ci.point) * 60}px"></i></span>` : "";
  return `<td class="num">${pct(ci.point)}${bar}${S.ci ? `<span class="ci">${fmtCI(ci)}</span>` : ""}</td>`;
}
const probeCell = p => p && p.n ? `<td class="num">${(100 * p.acc.point).toFixed(0)}%<span class="ci">n=${p.n}${S.ci ? " " + fmtCI(p.acc) : ""}</span></td>` : `<td class="pending">·</td>`;

function renderLB() {
  const datasets = allDatasets();
  const rows = rowsFor(datasets);
  const val = (r, k) => k === "avg" ? (r.avg ?? 9) : k === "numeral" ? -(r.numeral?.acc.point ?? -1) : k === "entity" ? -(r.entity?.acc.point ?? -1) : k === "vs" ? (r.vs?.delta_wer.point ?? 9) : (r.per[k]?.point ?? 9);
  rows.sort((a, b) => (val(a, S.sort) > val(b, S.sort) ? S.dir : -S.dir));
  const keys = ["", "", "avg", ...datasets, "numeral", "entity", "vs"];
  const th = ["#", "System", `Avg ${S.metric.toUpperCase()}`, ...datasets.map(d => DS_LABEL[d] || d), "Numerals", "Entities", "Δ vs Whisper v3"];
  document.querySelector("#lb thead").innerHTML = "<tr>" + th.map((t, i) => `<th data-key="${keys[i]}" class="${keys[i] === S.sort ? "sorted" : ""}">${t}</th>`).join("") + "</tr>";
  document.querySelector("#lb tbody").innerHTML = rows.length ? rows.map((r, i) => {
    const d = r.vs ? 100 * r.vs.delta_wer.point : null;
    const vs = r.vs ? `<td class="num delta ${d < 0 ? "neg" : "pos"}">${d > 0 ? "+" : ""}${d.toFixed(1)}${S.ci ? `<span class="ci">${fmtCI(r.vs.delta_wer)} p=${r.vs.mcnemar.p_value.toExponential(1)}</span>` : ""}</td>` : `<td class="pending">baseline</td>`;
    return `<tr data-id="${r.m.model_id}"><td>${i + 1}</td><td><span class="model">${r.m.display_name}</span><span class="kind">${r.m.submitted_by === "maintainer" || !r.m.submitted_by ? "" : "community"}</span></td><td class="num"><b>${r.avg == null ? "·" : pct(r.avg)}</b><span class="ci">${r.n}/${datasets.length} sets</span></td>${datasets.map(d => cell(r.per[d])).join("")}${probeCell(r.numeral)}${probeCell(r.entity)}${vs}</tr>`;
  }).join("") : `<tr><td colspan="${th.length}" class="pending">No results published yet. The first runs are in progress.</td></tr>`;
  document.getElementById("lb-sub").textContent = `${S.slice === "full" ? "full test sets" : "common stratified slice"} · ${S.norm} normalizer · ${S.metric.toUpperCase()} %`;
  document.querySelectorAll("#lb th[data-key]").forEach(h => h.onclick = () => { const k = h.dataset.key; if (!k) return; if (S.sort === k) S.dir = -S.dir; else { S.sort = k; S.dir = 1; } renderLB(); });
  document.querySelectorAll("#lb tbody tr[data-id]").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
}

function renderDialects() {
  const rows = board.models.filter(m => m.scores && m.scores[S.slice]);
  const dialects = [...new Set(rows.flatMap(m => Object.keys(m.scores[S.slice].dialects || {})))].filter(d => d !== "unknown").sort();
  document.querySelector("#dialects thead").innerHTML = "<tr><th>System</th>" + dialects.map(d => `<th>${d}</th>`).join("") + "</tr>";
  document.querySelector("#dialects tbody").innerHTML = rows.map(m => {
    const dd = m.scores[S.slice].dialects || {};
    return `<tr data-id="${m.model_id}"><td><span class="model">${m.display_name}</span></td>` + dialects.map(d => { const c = dd[d] && dd[d][S.norm]; return c ? `<td class="num">${pct(c.wer.point)}<span class="ci">n=${c.n}${S.ci ? " " + fmtCI(c.wer) : ""}</span></td>` : `<td class="pending">·</td>`; }).join("") + "</tr>";
  }).join("") || `<tr><td class="pending">No results yet.</td></tr>`;
  document.querySelectorAll("#dialects tbody tr[data-id]").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
}

function renderDatasets() {
  document.getElementById("datasets").innerHTML = Object.entries(DS_INFO).map(([d, c]) => `<div class="card"><h3>${DS_LABEL[d] || d}</h3><div class="kv">utterances <b>${c.n.toLocaleString()}</b> · hours <b>${c.hours}</b></div><div class="kv">matched to Elm <b>${c.elm.toLocaleString()}</b></div><div class="kv">dialect labels: ${c.dialects}</div></div>`).join("") + `<div class="card"><h3>MGB-2</h3><div class="kv pending">pending (QCRI license form)</div></div>`;
}

function renderQueue() {
  const q = [...board.queue].sort((a, b) => (b.submitted_at || "").localeCompare(a.submitted_at || ""));
  document.querySelector("#queue tbody").innerHTML = q.length ? q.map(r => `<tr><td><a href="https://huggingface.co/${r.hf_id}" target="_blank" rel="noopener">${r.hf_id}</a></td><td><span class="status s-${r.status}">${r.status}</span></td><td class="num">${(r.submitted_at || "").slice(0, 16).replace("T", " ")}</td><td>${r.submitted_by || ""}</td><td class="note">${(r.error || "").slice(0, 160)}</td></tr>`).join("") : `<tr><td colspan="5" class="pending">Queue is empty.</td></tr>`;
}

function renderFoot() {
  const first = board.models[0];
  document.getElementById("foot").innerHTML = `${first ? `manifest ${first.manifest_sha256} · bootstrap ${first.n_boot} · baseline ${first.baseline} · ` : ""}<a href="https://huggingface.co/datasets/${RESULTS_REPO}">results</a> · <a href="https://huggingface.co/datasets/${REQUESTS_REPO}/discussions">queue</a> · <a href="https://github.com/h9-tec/harfiyyan">code</a>`;
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
  renderLB(); renderDialects();
});
document.getElementById("show-ci").onchange = e => { S.ci = e.target.checked; renderLB(); renderDialects(); };

/* ---------- OAuth submission ---------- */
let oauth = null;
try { oauth = JSON.parse(localStorage.getItem("hf_oauth") || "null"); if (oauth && Date.parse(oauth.accessTokenExpiresAt) < Date.now()) oauth = null; } catch { oauth = null; }
try {
  const fresh = await oauthHandleRedirectIfPresent();
  if (fresh) { oauth = fresh; try { localStorage.setItem("hf_oauth", JSON.stringify(fresh)); } catch {} history.replaceState(null, "", location.pathname + location.hash); }
} catch (e) { console.warn("oauth redirect handling failed", e); }

function showAuth() {
  const ok = !!oauth;
  document.getElementById("auth").hidden = ok;
  document.getElementById("submit-form").hidden = !ok;
  if (ok) document.getElementById("who").textContent = oauth.userInfo?.name || oauth.userInfo?.preferred_username || "you";
}
document.getElementById("login").onclick = async () => { location.href = await oauthLoginUrl({ scopes: "openid profile write-discussions" }); };
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
      body: JSON.stringify({ title: `submit: ${hf_id}`, description: "```json\n" + body + "\n```\n\nSubmitted through the Harfiyyan Space." }),
    });
    if (!res.ok) throw new Error(`HF API ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const d = await res.json();
    msg.innerHTML = `Submitted. Track it here: <a href="https://huggingface.co/datasets/${REQUESTS_REPO}/discussions/${d.num}" target="_blank" rel="noopener">request #${d.num}</a>. The worker picks it up within a few minutes.`;
    e.target.reset();
  } catch (err) { msg.textContent = "Could not submit: " + err.message; }
};
showAuth();
loadBoard();
setInterval(loadBoard, 5 * 60 * 1000);
