// Dashboard renderer — reads static JSONL from ./data/ and populates
// the charts + per-run table. No fetch to any live endpoint; the whole
// page works as a file:// open once the JSONLs are written.

async function loadJsonl(url) {
  const resp = await fetch(url);
  if (!resp.ok) return [];
  const txt = await resp.text();
  return txt.split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l));
}

function groupBy(arr, key) {
  const out = new Map();
  for (const item of arr) {
    const k = item[key];
    if (!out.has(k)) out.set(k, []);
    out.get(k).push(item);
  }
  return out;
}

function meanOr(vals, fallback = null) {
  const nums = vals.filter((v) => v != null && !Number.isNaN(v));
  if (!nums.length) return fallback;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function computeAxisMeans(scores) {
  // { prompt_id: { axis: meanScore } }
  const byPrompt = groupBy(scores, "prompt_id");
  const out = new Map();
  for (const [pid, rows] of byPrompt.entries()) {
    const perAxis = new Map();
    for (const row of rows) {
      for (const s of row.scores || []) {
        if (!perAxis.has(s.axis)) perAxis.set(s.axis, []);
        perAxis.get(s.axis).push(s.score);
      }
    }
    const collapsed = {};
    for (const [axis, vals] of perAxis.entries()) {
      collapsed[axis] = meanOr(vals);
    }
    out.set(pid, collapsed);
  }
  return out;
}

function computeRunLevel(traces, scores) {
  // Merge by (prompt_id, run) so the table can show per-run means.
  const scoreByKey = new Map();
  for (const s of scores) {
    scoreByKey.set(`${s.prompt_id}::${s.run}`, s);
  }
  return traces.map((t) => {
    const s = scoreByKey.get(`${t.prompt_id}::${t.run}`);
    const axisScores = (s && s.scores) || [];
    const nums = axisScores.map((a) => a.score).filter((v) => v != null);
    return {
      prompt_id: t.prompt_id,
      tier:      t.tier,
      run:       t.run,
      wall_s:    (t.wall_ms || 0) / 1000,
      n_tools:   t.n_tool_calls || 0,
      error:     t.error,
      mean:      nums.length ? meanOr(nums) : null,
      axes:      axisScores,
    };
  });
}

function palette(i, total) {
  // Distinct-but-cohesive HSL wheel.
  const hue = Math.round((i * 360) / Math.max(total, 1));
  return `hsl(${hue}, 62%, 52%)`;
}

function renderRubricChart(rubricByPrompt) {
  const ctx = document.getElementById("chart-rubric");
  // Discover axis universe → dataset per axis, x=prompt.
  const promptIds = Array.from(rubricByPrompt.keys());
  const axes = new Set();
  for (const map of rubricByPrompt.values()) {
    Object.keys(map).forEach((a) => axes.add(a));
  }
  const axisList = Array.from(axes);
  const datasets = axisList.map((axis, i) => ({
    label: axis,
    backgroundColor: palette(i, axisList.length),
    borderColor: palette(i, axisList.length),
    data: promptIds.map((pid) => rubricByPrompt.get(pid)[axis] ?? null),
  }));
  new Chart(ctx, {
    type: "bar",
    data: { labels: promptIds, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 5,
             title: { display: true, text: "score (0–5)" } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
}

function renderWallChart(runLevel) {
  const ctx = document.getElementById("chart-wall");
  const byPrompt = groupBy(runLevel, "prompt_id");
  const labels = Array.from(byPrompt.keys());
  const data = labels.map((pid) => {
    const rows = byPrompt.get(pid);
    return meanOr(rows.map((r) => r.wall_s));
  });
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "mean wall-clock (s)",
        backgroundColor: "#2f6cff",
        data,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true,
                     title: { display: true, text: "seconds" } } },
    },
  });
}

function renderToolsChart(runLevel) {
  const ctx = document.getElementById("chart-tools");
  const byPrompt = groupBy(runLevel, "prompt_id");
  const labels = Array.from(byPrompt.keys());
  const data = labels.map((pid) => {
    const rows = byPrompt.get(pid);
    return meanOr(rows.map((r) => r.n_tools));
  });
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "mean tool calls",
        backgroundColor: "#7d5fff",
        data,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero: true,
                     title: { display: true, text: "count" } } },
    },
  });
}

function renderRunsTable(runLevel) {
  const tbody = document.querySelector("#runs tbody");
  tbody.innerHTML = "";
  for (const [i, r] of runLevel.entries()) {
    const parent = document.createElement("tr");
    parent.className = "parent";
    parent.dataset.idx = String(i);
    const meanCell = r.mean == null
      ? "<span class='bad'>—</span>"
      : (r.mean >= 4 ? "<span class='good'>" : "<span>")
        + r.mean.toFixed(2) + "</span>";
    const errCell = r.error
      ? `<span class="bad">${escapeHtml(r.error)}</span>`
      : "";
    parent.innerHTML = `
      <td>${escapeHtml(r.prompt_id)} <small style="color:#888">(${r.tier})</small></td>
      <td>${r.run}</td>
      <td>${r.wall_s.toFixed(1)}</td>
      <td>${r.n_tools}</td>
      <td>${meanCell}</td>
      <td>${errCell}</td>
    `;
    const detail = document.createElement("tr");
    detail.className = "detail";
    const inner = r.axes.length
      ? `<ul class="axis-list">${r.axes.map((a) => `
          <li>
            <strong>${escapeHtml(a.axis)}</strong>:
            <span class="axis-score">${a.score ?? "—"}</span>
            — ${escapeHtml(a.rationale || "")}
          </li>`).join("")}</ul>`
      : "<em class='axis-error'>no axis scores</em>";
    detail.innerHTML = `<td colspan="6">${inner}</td>`;
    parent.addEventListener("click", () => detail.classList.toggle("open"));
    tbody.appendChild(parent);
    tbody.appendChild(detail);
  }
}

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

(async function main() {
  const [traces, scores] = await Promise.all([
    loadJsonl("data/traces.jsonl"),
    loadJsonl("data/scores.jsonl"),
  ]);
  document.getElementById("counts").textContent =
    `${traces.length} run(s), ${scores.length} scored`;

  if (!traces.length) return;

  const rubricByPrompt = computeAxisMeans(scores);
  const runLevel = computeRunLevel(traces, scores);

  renderRubricChart(rubricByPrompt);
  renderWallChart(runLevel);
  renderToolsChart(runLevel);
  renderRunsTable(runLevel);
})();
