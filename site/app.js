/* Harfiyyan leaderboard: renders dist/summary.json. No build step, no framework. */
(async function () {
  const S = { slice: "full", norm: "elm", metric: "wer", ci: true, sort: "avg", dir: 1 };
  const summary = await fetch("summary.json").then(r => r.json());
  const datasets = Object.keys(summary.datasets);
  const DS_LABEL = { sada: "SADA", casablanca: "Casablanca", commonvoice: "CV 18", masc_clean: "MASC clean", masc_noisy: "MASC noisy", mgb2: "MGB-2" };
  const pct = x => (100 * x).toFixed(1);
  const fmtCI = (c, scale = 100) => `[${(scale * c.lo).toFixed(1)}, ${(scale * c.hi).toFixed(1)}]`;

  document.getElementById("meta-pills").innerHTML = [
    `${summary.models.length} systems`, `${datasets.length} test sets`,
    `manifest ${summary.meta.manifest_sha256.slice(0, 12)}`, `built ${summary.meta.generated_at.slice(0, 10)}`,
  ].map(t => `<span class="pill">${t}</span>`).join("");

  function rowsFor() {
    return summary.models.filter(m => m.scores[S.slice]).map(m => {
      const sc = m.scores[S.slice];
      const per = {}; let sum = 0, n = 0;
      for (const d of datasets) {
        const cell = sc.datasets[d] && sc.datasets[d][S.norm];
        if (cell) { per[d] = cell[S.metric]; sum += cell[S.metric].point; n++; }
      }
      return { m, sc, per, avg: n ? sum / n : null, n, numeral: sc.probes.numeral, vs: sc.vs_baseline };
    });
  }

  function cell(ci, extra = "") {
    if (!ci) return `<td class="pending">pending</td>`;
    const w = Math.max(0, Math.min(1, ci.hi)) * 100, l = Math.max(0, ci.lo) * 100;
    const bar = S.ci ? `<span class="bar"><b style="left:${l * 0.6}px;width:${Math.max(1, (w - l) * 0.6)}px"></b><i style="left:${ci.point * 60}px"></i></span>` : "";
    const sub = S.ci ? `<span class="ci">${fmtCI(ci)}</span>` : "";
    return `<td class="num ${extra}">${pct(ci.point)}${bar}${sub}</td>`;
  }

  function renderLB() {
    const rows = rowsFor();
    const key = S.sort;
    rows.sort((a, b) => {
      const va = key === "avg" ? a.avg : key === "numeral" ? -(a.numeral?.acc.point ?? -1) : key === "vs" ? (a.vs?.delta_wer.point ?? 9) : (a.per[key]?.point ?? 9);
      const vb = key === "avg" ? b.avg : key === "numeral" ? -(b.numeral?.acc.point ?? -1) : key === "vs" ? (b.vs?.delta_wer.point ?? 9) : (b.per[key]?.point ?? 9);
      return (va ?? 9) > (vb ?? 9) ? S.dir : -S.dir;
    });
    const th = ["#", "System", `Avg ${S.metric.toUpperCase()}`, ...datasets.map(d => DS_LABEL[d] || d), "Numeral probe", "Δ vs Whisper v3"];
    const keys = ["", "", "avg", ...datasets, "numeral", "vs"];
    document.querySelector("#lb thead").innerHTML = "<tr>" + th.map((t, i) => `<th data-key="${keys[i]}" class="${keys[i] === S.sort ? "sorted" : ""}">${t}</th>`).join("") + "</tr>";
    document.querySelector("#lb tbody").innerHTML = rows.map((r, i) => {
      const vs = r.vs ? `<td class="num delta ${r.vs.delta_wer.point < 0 ? "neg" : "pos"}">${(100 * r.vs.delta_wer.point).toFixed(1) > 0 ? "+" : ""}${(100 * r.vs.delta_wer.point).toFixed(1)}${S.ci ? `<span class="ci">${fmtCI(r.vs.delta_wer)} p=${r.vs.mcnemar.p_value.toExponential(1)}</span>` : ""}</td>` : `<td class="pending">baseline</td>`;
      const nm = r.numeral && r.numeral.n ? `<td class="num">${(100 * r.numeral.acc.point).toFixed(0)}%<span class="ci">n=${r.numeral.n} ${S.ci ? fmtCI(r.numeral.acc) : ""}</span></td>` : `<td class="pending">—</td>`;
      return `<tr data-id="${r.m.model_id}"><td>${i + 1}</td><td><span class="model">${r.m.display_name}</span><span class="kind">${r.m.kind}</span></td><td class="num"><b>${r.avg == null ? "—" : pct(r.avg)}</b><span class="ci">${r.n}/${datasets.length} sets</span></td>${datasets.map(d => cell(r.per[d])).join("")}${nm}${vs}</tr>`;
    }).join("");
    document.getElementById("lb-sub").textContent = `${S.slice === "full" ? "full test sets" : "common stratified slice"} · ${S.norm} normalizer · ${S.metric.toUpperCase()} %`;
    document.querySelectorAll("#lb th[data-key]").forEach(h => h.onclick = () => { const k = h.dataset.key; if (!k) return; if (S.sort === k) S.dir = -S.dir; else { S.sort = k; S.dir = 1; } renderLB(); });
    document.querySelectorAll("#lb tbody tr").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
  }

  function renderDialects() {
    const rows = summary.models.filter(m => m.scores[S.slice]);
    const dialects = [...new Set(rows.flatMap(m => Object.keys(m.scores[S.slice].dialects)))].filter(d => d !== "unknown").sort();
    document.querySelector("#dialects thead").innerHTML = "<tr><th>System</th>" + dialects.map(d => `<th>${d}</th>`).join("") + "</tr>";
    document.querySelector("#dialects tbody").innerHTML = rows.map(m => {
      const dd = m.scores[S.slice].dialects;
      return `<tr data-id="${m.model_id}"><td><span class="model">${m.display_name}</span></td>` + dialects.map(d => { const c = dd[d] && dd[d][S.norm]; return c ? `<td class="num">${pct(c.wer.point)}<span class="ci">n=${c.n}${S.ci ? " " + fmtCI(c.wer) : ""}</span></td>` : `<td class="pending">—</td>`; }).join("") + "</tr>";
    }).join("");
    document.querySelectorAll("#dialects tbody tr").forEach(tr => tr.onclick = () => openCard(tr.dataset.id));
  }

  function renderDatasets() {
    document.getElementById("datasets").innerHTML = datasets.map(d => { const c = summary.datasets[d]; return `<div class="card"><h3>${DS_LABEL[d] || d}</h3><div class="kv">utterances <b>${c.n.toLocaleString()}</b> · hours <b>${c.hours}</b></div><div class="kv">Elm members <b>${c.elm_members.toLocaleString()}</b> · common slice <b>${c.n_api_slice}</b></div><div class="kv">dialect labels: ${c.dialects.filter(x => x !== "unknown").join(", ") || "none"}</div></div>`; }).join("");
  }

  function openCard(id) {
    const m = summary.models.find(x => x.model_id === id); if (!m) return;
    const c = m.card || {}, run = m.run || {};
    const cov = m.scores[S.slice] ? Object.entries(m.scores[S.slice].coverage).map(([d, v]) => `${DS_LABEL[d] || d} ${v.scored}/${v.n}`).join(" · ") : "";
    document.querySelector("#card .card-body").innerHTML = `<h3>${m.display_name}</h3><dl>
      <dt>model id</dt><dd>${m.model_id}</dd>
      <dt>weights</dt><dd>${m.hf_id ? `<a href="https://huggingface.co/${m.hf_id}" target="_blank" rel="noopener">${m.hf_id}</a>` : "—"}</dd>
      <dt>org · license</dt><dd>${c.org || "—"} · ${c.license || "—"}${c.params ? " · " + c.params : ""}</dd>
      <dt>language hint</dt><dd>${run.language_hint ?? "—"}</dd>
      <dt>decode</dt><dd>${JSON.stringify(run.decode_params || {})}</dd>
      <dt>env</dt><dd>${JSON.stringify(run.env || {})}</dd>
      <dt>run</dt><dd>${run.run_id || "—"} · ${(run.finished || "").slice(0, 16)} · code ${run.git_sha ? run.git_sha.slice(0, 8) : "—"}</dd>
      <dt>coverage</dt><dd>${cov}</dd>
      <dt>outputs</dt><dd><a href="https://github.com/h9-tec/harfiyyan/tree/main/results/${m.model_id}" target="_blank" rel="noopener">results/${m.model_id}/</a></dd>
    </dl>`;
    document.getElementById("card").showModal();
  }

  document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
    const k = Object.keys(b.dataset)[0]; S[k] = b.dataset[k];
    b.parentElement.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    renderLB(); renderDialects();
  });
  document.getElementById("show-ci").onchange = e => { S.ci = e.target.checked; renderLB(); renderDialects(); };

  document.getElementById("foot").innerHTML = `manifest ${summary.meta.manifest_sha256} · code ${summary.meta.git_sha} · bootstrap ${summary.meta.n_boot} · baseline ${summary.meta.baseline} · <a href="https://github.com/h9-tec/harfiyyan">github.com/h9-tec/harfiyyan</a> · <a href="https://github.com/h9-tec/harfiyyan/blob/main/docs/methodology.md">methodology</a>`;
  renderLB(); renderDialects(); renderDatasets();
})();
