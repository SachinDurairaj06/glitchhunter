// GlitchHunter dashboard: polls the API and renders the agent's pipeline, decisions, findings,
// approvals, the evaluation scoreboard and the detector catalog.
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { hunt: null, next: 0, workDir: "", running: false, timer: null, approvalIds: null,
    events: [], families: {}, games: [], health: null };
  const token = new URLSearchParams(location.search).get("token") || "";

  const STAGES = [
    ["Survey", ["survey"]],
    ["Screen sizes", ["sweep_viewports"]],
    ["Zoom & replay", ["zoom_in", "replay", "look", "soak", "dismiss"]],
    ["Visual bisect", ["bisect", "show_commit"]],
    ["Fix", ["propose_fix"]],
    ["Verify", ["verify_fix"]],
    ["Human", ["request_human"]],
    ["Report", ["file_report"]],
  ];
  const FAMILY_TEXT = {
    classic: "single frames and simple temporal checks",
    motion: "dense optical flow (DIS) over the clip",
    compare: "against the approved golden build",
    lighting: "same surfaces lit differently than in the golden build",
    pixels: "invalid pixels and layout",
  };

  async function api(path, opts = {}) {
    const headers = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(path, { ...opts, headers });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  }

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "style") node.setAttribute("style", v);
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const pretty = (name) => (name || "").replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
  const pct = (v) => `${Math.round((v || 0) * 100)}%`;
  const fam = (detector) => state.families[detector] || "classic";

  function fileUrl(path) {
    if (!path) return null;
    const norm = path.replace(/\\/g, "/");
    const root = state.workDir.replace(/\\/g, "/");
    if (root && norm.toLowerCase().startsWith(root.toLowerCase())) {
      return "/files/" + norm.slice(root.length).replace(/^\/+/, "").split("/").map(encodeURIComponent).join("/");
    }
    return null;
  }

  function thumbs(paths) {
    const urls = (paths || []).map(fileUrl).filter(Boolean);
    if (!urls.length) return null;
    return el("div", { class: "thumbs" }, urls.map((u) =>
      el("img", { src: u, loading: "lazy", alt: "evidence", onclick: () => openViewer(u),
                  onerror: (e) => e.target.remove() })));
  }

  function openViewer(url) {
    $("viewer-img").src = url;
    $("viewer").showModal();
  }

  function openDiff(title, diff) {
    $("diff-title").textContent = title;
    const body = $("diff-body");
    body.replaceChildren(...(diff || "(no diff)").split("\n").map((line) => {
      const cls = line.startsWith("+") && !line.startsWith("+++") ? "add"
        : line.startsWith("-") && !line.startsWith("---") ? "del" : line.startsWith("@@") ? "hunk" : null;
      return el("div", { class: cls }, line || " ");
    }));
    $("diff-viewer").showModal();
  }

  // --- header -----------------------------------------------------------------------------------
  async function loadHealth() {
    const h = await api("/api/health");
    state.health = h;
    state.workDir = h.work_dir;
    const cv = h.opencv;
    const g = h.game || {};
    const chips = [
      ["OpenCV", cv.version],
      ["DNN", (cv.dnn_engines || []).join(" / ").replace(/ENGINE_/g, "").toLowerCase()],
      ["build", cv.distribution],
      ["CPU", cv.machine],
      ["baseline", g.baseline || "-"],
      ["scenarios", (g.scenarios || []).length],
      ["screen sizes", 1 + (g.viewports || []).length],
      ["planner", h.planner],
      ["approvals", h.approval_mode],
    ];
    $("chips").replaceChildren(...chips.map(([k, v]) => el("span", { class: "chip" }, `${k} `, el("b", {}, v))));
    $("planner").value = h.planner;
    const native = g.driver === "bridge";
    $("engine").textContent = native ? "native build over the bridge" : "web build in headless Chromium";
    $("engine").className = `engine${native ? " native" : ""}`;
    $("play").replaceChildren(...(native || !g.baseline ? [] : [
      el("a", { href: "/game/main/", target: "_blank", rel: "noopener" }, "Play latest build"),
      el("a", { href: `/game/${encodeURIComponent(g.baseline)}/`, target: "_blank", rel: "noopener" }, `Play ${g.baseline}`),
    ]));
  }

  async function loadGames() {
    const data = await api("/api/games");
    state.games = data.games;
    const pick = $("game");
    pick.replaceChildren(...data.games.map((g) => el("option", { value: g.name, selected: g.name === data.current ? "" : null },
      `${g.title} · ${g.engine}${g.built ? "" : " (not built)"}`)));
    if (!data.current) pick.prepend(el("option", { value: "", selected: "" }, "your game (--repo)"));
  }

  $("game").addEventListener("change", async (e) => {
    const name = e.target.value;
    if (!name) return;
    $("status").textContent = `Switching to ${name}${state.games.find((g) => g.name === name && !g.built) ? " (building its history first)" : ""}...`;
    try {
      await api("/api/game", { method: "POST", body: JSON.stringify({ game: name }) });
      await Promise.all([loadHealth(), loadGames()]);
      $("status").textContent = `Ready: ${state.health.game.title}. Start a hunt.`;
    } catch (err) {
      $("status").textContent = `Could not switch: ${err.message}`;
      loadGames();
    }
  });

  // --- tabs -------------------------------------------------------------------------------------
  for (const button of document.querySelectorAll(".tabs button")) {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  }
  function showTab(name) {
    for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("active", b.dataset.tab === name);
    for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.id === `tab-${name}`);
    if (name === "score") loadScoreboard();
    if (name === "history") loadHunts();
  }

  // --- pipeline and KPIs ------------------------------------------------------------------------
  function renderPipeline() {
    const counts = STAGES.map(() => 0);
    let active = -1;
    for (const e of state.events) {
      if (e.kind !== "decision") continue;
      const i = STAGES.findIndex(([, tools]) => tools.includes(e.tool));
      if (i >= 0) { counts[i] += 1; active = i; }
    }
    $("pipeline").replaceChildren(...STAGES.map(([name], i) => el("li", {
      class: [counts[i] ? "done" : "", state.running && i === active ? "active" : ""].join(" ").trim() || null,
      title: `${counts[i]} decision(s)`,
    }, el("div", { class: "name" }, name), el("div", { class: "count" }, counts[i]))));
  }

  function renderKpis(findings, outcome) {
    const live = findings.filter((f) => f.status !== "dismissed");
    const culprits = new Set(findings.filter((f) => f.culprit).map((f) => f.culprit.sha));
    const reported = findings.filter((f) => f.status === "reported").length;
    const verified = findings.filter((f) => f.verification && f.verification.passed).length;
    const builds = findings.reduce((n, f) => n + ((f.bisect && f.bisect.tested) || 0), 0);
    const last = state.events.length ? state.events[state.events.length - 1].t || 0 : 0;
    const secs = (outcome && outcome.seconds) || last;
    const kpis = [
      ["Findings", live.length],
      ["Root causes", culprits.size],
      ["Fixes verified", verified, verified > 0],
      ["Reported", reported, reported > 0],
      ["Builds bisected", builds],
      ["Time", secs ? `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s` : "-"],
    ];
    $("kpis").replaceChildren(...kpis.map(([k, v, good]) => el("div", { class: `kpi${good ? " good" : ""}` },
      el("div", { class: "v" }, v), el("div", { class: "k" }, k))));
  }

  // --- timeline ---------------------------------------------------------------------------------
  function sawLine(ev) {
    if (!ev || !ev.detector) return null;
    const parts = [el("b", {}, pretty(ev.detector))];
    if (ev.confidence != null) parts.push(` confidence ${Number(ev.confidence).toFixed(2)}`);
    if (ev.area_px != null) parts.push(` · ${Math.round(ev.area_px)} px`);
    if (ev.repro) parts.push(` · seen ${ev.repro}`);
    return el("div", { class: "saw" }, "OpenCV saw ", ...parts);
  }

  function argText(args) {
    if (!args) return "";
    const bits = Object.entries(args).filter(([, v]) => v != null && v !== "")
      .map(([k, v]) => (k === "finding_id" ? v : `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`));
    return bits.join(" ");
  }

  function renderEvent(e) {
    const t = el("div", { class: "t" }, `${(e.t || 0).toFixed(1)}s`);
    switch (e.kind) {
      case "hunt_start":
        return el("li", { class: "milestone" }, t, el("div", { class: "text" },
          `Hunt on ${e.game || "the game"}: ${e.scenarios ? e.scenarios.length : "?"} scenarios, head ${e.head}, approved build ${e.baseline}, planner ${e.planner}`));
      case "decision":
        return el("li", { class: "decision" }, t, el("div", {},
          el("div", { class: "text" }, el("span", { class: "tool" }, e.tool), argText(e.args)),
          sawLine(e.evidence),
          el("div", { class: "why" }, e.reason)));
      case "result":
        return el("li", { class: `result ${e.ok ? "ok" : "err"}` }, t, el("div", {},
          el("div", { class: "text" }, e.summary || e.error), thumbs(e.images)));
      case "bisect_step":
        return el("li", { class: "bisect" }, el("div", { class: "t" }), el("div", { class: "text" },
          `bisect ${e.finding} · ${e.commit} → ${e.verdict} (${e.runs} capture${e.runs === 1 ? "" : "s"})`));
      case "fallback":
      case "planner_error":
      case "tool_error":
      case "warning":
        return el("li", { class: "error" }, t, el("div", { class: "text" },
          `${e.kind}: ${e.reason || e.error || (e.traceback || "").split("\n").slice(-2).join(" ")}`));
      case "hunt_end":
        return el("li", { class: "milestone" }, t, el("div", { class: "text" },
          `Done: ${e.ended} · ${e.steps} steps · ${e.captures} captures · ${e.seconds}s`));
      default:
        return null;
    }
  }

  async function pollTrace() {
    if (!state.hunt) return;
    const data = await api(`/api/hunts/${state.hunt}/trace?since=${state.next}`);
    const list = $("timeline");
    const atBottom = list.scrollTop + list.clientHeight >= list.scrollHeight - 30;
    for (const e of data.events) {
      state.events.push(e);
      const node = renderEvent(e);
      if (node) list.append(node);
    }
    state.next = data.next;
    if (atBottom) list.scrollTop = list.scrollHeight;
  }

  // --- findings ---------------------------------------------------------------------------------
  function lifecycle(f) {
    const steps = [
      ["detected", true],
      ["reproduced", f.repro ? f.repro.hits > 0 : null],
      ["root cause", f.culprit ? true : null],
      ["fix", f.fix ? !f.fix.deleted : null],
      ["verified", f.verification ? f.verification.passed : null],
      ["approved", f.approval && f.approval.choice ? f.approval.choice === "approve" : null],
      ["reported", f.status === "reported" ? true : null],
    ];
    return el("div", { class: "life" }, steps.map(([name, ok]) =>
      el("span", { class: ok === true ? "on" : ok === false ? "fail" : null }, name)));
  }

  function bisectStrip(f) {
    const b = f.bisect;
    if (!b || !b.steps || !b.steps.length) return null;
    const dots = [];
    b.steps.forEach((s, i) => {
      if (i) dots.push(el("span", { class: "arrow" }, "›"));
      const cls = s.bad == null ? "skip" : s.bad ? "bad" : "good";
      dots.push(el("span", { class: `step ${cls}`,
        title: `${s.commit} ${s.subject || ""}: ${s.bad == null ? "skipped (does not build)" : s.bad ? "glitch present" : "clean"}${s.known_from ? ` (known from ${s.known_from})` : ""}` }));
    });
    dots.push(el("span", { class: "culprit" }, `= ${b.culprit}`));
    return el("div", {}, el("div", { class: "bisect-strip", title: "visual bisect: each square is a build the agent tested" }, dots),
      el("div", { class: "muted", style: "font-size:12px" },
        `${b.tested} of ${b.range} commits tested · ${b.repeats} capture(s) per commit`
        + (b.via_merge ? ` · found inside merge ${b.via_merge}` : "")
        + ((b.skipped || []).length ? ` · skipped broken build ${b.skipped.join(", ")}` : "")));
  }

  function heroImage(f) {
    const ev = f.evidence || {};
    const key = ["golden_vs_now", "before_after_fix", "closeup", "heatmap", "before_after", "annotated", "strip", "frame"]
      .find((k) => ev[k] && fileUrl(ev[k]));
    if (!key) return null;
    const url = fileUrl(ev[key]);
    return el("img", { class: "hero", src: url, alt: `${f.detector} evidence`, loading: "lazy",
      onclick: () => openViewer(url), onerror: (e) => e.target.remove() });
  }

  function renderFinding(f) {
    const family = fam(f.detector);
    const ev = f.evidence || {};
    const extra = ["before_closeup", "closeup", "heatmap", "series", "tone_curve", "strip", "motion_plot", "diff",
      "before_after", "before_after_fix"].filter((k) => ev[k]);
    const rows = [
      ["seen in", [f.scenario, ...(f.also_seen || [])].join(", ")],
      ["object", f.object || "whole screen"],
      ["evidence", f.note],
      ["reproduced", f.repro ? `${f.repro.hits}/${f.repro.runs}${f.repro.deterministic ? " · deterministic" : " · intermittent"}` : null],
      ["root cause", f.culprit ? el("span", {}, `${f.culprit.sha.slice(0, 10)} “${f.culprit.subject}” `,
        el("button", { class: "linkish", onclick: () => openDiff(`${f.culprit.sha.slice(0, 10)} ${f.culprit.subject}`, f.culprit.diff) }, "diff")) : null],
      ["same cause as", f.duplicate_of || null],
      ["fix", f.fix ? el("span", {}, `${f.fix.strategy || "revert"} on ${f.fix.branch}${f.fix.deleted ? " (deleted)" : ""} `,
        f.fix.diff ? el("button", { class: "linkish", onclick: () => openDiff(f.fix.branch, f.fix.diff) }, "diff") : null) : null],
      ["verified", f.verification ? `${f.verification.passed ? "PASS" : "FAIL"}: ${f.verification.reason}` : null],
      ["decision", f.approval && f.approval.choice ? `${f.approval.choice} by ${f.approval.by}${f.approval.note ? `: ${f.approval.note}` : ""}` : null],
    ].filter(([, v]) => v);
    const reportUrl = fileUrl(f.report);
    if (reportUrl) rows.push(["report", el("a", { href: reportUrl, target: "_blank", rel: "noopener" }, `${f.id}.md`)]);
    const dim = ["dismissed", "duplicate"].includes(f.status);
    return el("article", { class: `finding${dim ? " dimmed" : ""}`, style: `--fam: var(--${family})` },
      el("h3", {}, el("span", { class: "title" }, `${f.id} · ${pretty(f.detector)}`),
        el("span", { class: "badge fam" }, family),
        el("span", { class: `badge ${f.severity}` }, f.severity),
        el("span", { class: `badge ${f.status}` }, f.status.replace(/_/g, " "))),
      el("div", { class: "muted", style: "font-size:12px" },
        `score ${f.score.toFixed(2)} · confidence ${f.confidence.toFixed(2)}${f.zooms ? ` · camera moved closer ${f.zooms}x` : ""}`),
      el("div", { class: "bar" }, el("span", { style: `width:${pct(f.confidence)}` })),
      dim ? null : heroImage(f),
      lifecycle(f),
      bisectStrip(f),
      el("dl", {}, rows.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])),
      dim ? null : thumbs(extra.map((k) => ev[k])));
  }

  const ORDER = { reported: 0, awaiting_human: 1, fix_verified: 2, fix_proposed: 3, bisected: 4, confirmed: 5,
    candidate: 6, rejected: 7, duplicate: 8, dismissed: 9 };

  async function pollFindings() {
    if (!state.hunt) return;
    const data = await api(`/api/hunts/${state.hunt}`);
    state.running = data.status === "running" || data.status === "running elsewhere";
    const findings = (data.findings || []).slice().sort((a, b) => (ORDER[a.status] ?? 5) - (ORDER[b.status] ?? 5));
    const byStatus = {};
    for (const f of findings) byStatus[f.status] = (byStatus[f.status] || 0) + 1;
    $("findings-count").textContent = Object.entries(byStatus).map(([s, n]) => `${n} ${s.replace(/_/g, " ")}`).join(" · ");
    const sig = JSON.stringify(findings) + state.running;
    if (sig !== state.findingsSig) { // re-render only on change: keeps the reader's scroll and open links
      state.findingsSig = sig;
      $("findings").replaceChildren(...(findings.length ? findings.map(renderFinding)
        : [el("p", { class: "muted" }, state.running ? "Surveying: findings appear as the agent works." : "No findings.")]));
    }
    const o = data.outcome || {};
    $("status").textContent = data.status === "running elsewhere"
      ? `Hunt ${state.hunt} is running in another process (CLI or eval); findings appear when it ends`
      : state.running
        ? `Hunt ${state.hunt} running · step ${data.step} · planner ${data.planner}`
        : `Hunt ${state.hunt}: ${o.ended || data.status}`;
    $("stop").disabled = data.status !== "running";
    renderPipeline();
    renderKpis(findings, data.outcome);
  }

  // --- approvals --------------------------------------------------------------------------------
  async function pollApprovals() {
    const data = await api("/api/approvals");
    const box = $("approvals");
    $("approvals-panel").hidden = !data.pending.length;
    const ids = data.pending.map((r) => r.id).join(",");
    if (ids === state.approvalIds) return; // unchanged: keep any note the reviewer is typing
    state.approvalIds = ids;
    box.replaceChildren(...data.pending.map((r) => {
      const note = el("textarea", { rows: "2", placeholder: "Optional note for the report" });
      const ev = (r.context && r.context.evidence) || {};
      const buttons = r.options.map((opt, i) => el("button", {
        class: i === 0 ? "ok" : "bad",
        onclick: async () => {
          await api(`/api/approvals/${r.id}`, { method: "POST", body: JSON.stringify({ choice: opt, note: note.value }) });
          pollApprovals();
        },
      }, opt));
      return el("div", { class: "approval" },
        el("div", {}, el("b", {}, `${r.finding_id}: `), r.question),
        thumbs(["before_after_fix", "golden_vs_now", "closeup", "heatmap", "annotated"].map((k) => ev[k]).filter(Boolean)),
        note, el("div", { class: "row" }, buttons));
    }));
  }

  // --- scoreboard -------------------------------------------------------------------------------
  function meter(v) {
    const cls = v >= 0.9 ? "" : v > 0 ? " low" : " zero";
    return el("div", { class: `meter${cls}` }, el("div", { class: "bar" }, el("span", { style: `width:${pct(v)}` })),
      el("span", {}, pct(v)));
  }

  async function loadScoreboard() {
    const data = await api("/api/eval");
    const box = $("scoreboard");
    if (!data.results.length) {
      box.replaceChildren(el("p", { class: "muted" }, "No evaluation yet: run python -m glitchhunter eval --game <game>."));
      return;
    }
    box.replaceChildren(...data.results.map((r) => {
      const game = state.games.find((g) => g.name === r.game) || { title: r.game, engine: "" };
      const modes = Object.entries(r.modes || {});
      const full = (r.modes || {}).full;
      const fp = r.false_positive_check || {};
      return el("section", { class: "panel score" },
        el("h2", {}, game.title), el("div", { class: "meta" },
          `${game.engine} · ${game.bugs || "?"} planted bugs in ${game.commits || "?"} commits · ${r.repeats} run(s) per mode · ${r.started}`),
        el("table", {},
          el("thead", {}, el("tr", {}, ["mode", "found", "right culprit", "fix verified", "false alarms / run"].map((h) => el("th", {}, h)))),
          el("tbody", {}, modes.map(([mode, m]) => el("tr", {},
            el("td", {}, mode.replace(/_/g, " ")), el("td", {}, meter(m.recall)), el("td", {}, meter(m.culprit_accuracy)),
            el("td", {}, meter(m.fix_verified)), el("td", {}, (m.false_alarms / Math.max(1, m.runs)).toFixed(1)))))),
        el("div", { class: "fp" }, `Clean builds: ${fp.runs_with_false_alarm ?? "?"} of ${fp.runs ?? "?"} captures raised anything`),
        full && full.per_bug ? el("div", { class: "bugs" }, Object.entries(full.per_bug).map(([bug, v]) =>
          el("div", {}, `${v.detected >= 1 ? "✓" : v.detected > 0 ? "◐" : "✗"} ${bug.replace(/_/g, " ")}`))) : null);
    }));
  }

  // --- detectors --------------------------------------------------------------------------------
  async function loadDetectors() {
    const data = await api("/api/detectors");
    for (const [family, list] of Object.entries(data.families)) for (const d of list) state.families[d.name] = family;
    $("detectors").replaceChildren(...Object.entries(data.families).map(([family, list]) =>
      el("section", { class: "panel family", style: `--fam: var(--${family})` },
        el("h2", {}, `${family} · ${list.length}`), el("div", { class: "what" }, FAMILY_TEXT[family] || ""),
        list.map((d) => el("div", { class: "det" }, el("code", {}, d.name),
          d.needs_golden ? el("span", { class: "badge", style: "margin-left:6px" }, "vs golden") : null,
          el("p", {}, d.description))))));
  }

  // --- hunts ------------------------------------------------------------------------------------
  async function loadHunts() {
    const rows = await api("/api/hunts");
    $("hunts").replaceChildren(...rows.map((h) => el("tr", { onclick: () => { select(h.hunt); showTab("hunt"); } },
      el("td", {}, el("code", {}, h.hunt)), el("td", {}, h.game || "-"), el("td", {}, h.reported ?? "-"),
      el("td", {}, h.status === "running" ? "running" : h.steps ?? "-"),
      el("td", {}, h.seconds ? `${Math.round(h.seconds)}s` : "-"), el("td", { class: "muted" }, h.ended || ""))));
    return rows;
  }

  function select(huntId) {
    state.hunt = huntId;
    state.next = 0;
    state.events = [];
    state.findingsSig = null;
    $("timeline").replaceChildren();
    tick();
  }

  async function tick() {
    try {
      await Promise.all([pollTrace(), pollApprovals()]);
      await pollFindings();
    } catch (err) {
      $("status").textContent = `API error: ${err.message}`;
    }
    clearTimeout(state.timer);
    state.timer = setTimeout(tick, state.running ? 1000 : 5000);
  }

  $("start").addEventListener("click", async () => {
    $("start").disabled = true;
    try {
      const res = await api("/api/hunts", { method: "POST", body: JSON.stringify({ planner: $("planner").value }) });
      select(res.hunt);
      loadHunts();
    } catch (err) {
      $("status").textContent = `Could not start: ${err.message}`;
    } finally {
      $("start").disabled = false;
    }
  });
  $("stop").addEventListener("click", () => state.hunt && api(`/api/hunts/${state.hunt}/stop`, { method: "POST" }));

  (async () => {
    await Promise.all([loadHealth(), loadGames(), loadDetectors()]);
    const rows = await loadHunts();
    const wanted = new URLSearchParams(location.search).get("hunt");
    if (wanted) select(wanted);
    else if (rows.length) select(rows[0].hunt);
    else { renderPipeline(); renderKpis([], null); tick(); }
    setInterval(loadHunts, 15000);
  })();
})();
