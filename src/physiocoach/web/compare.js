// Compare tab: the same record through every generator, with unfaithful sentences highlighted.
(() => {
  const { api, esc } = window.PC;
  const $ = (id) => document.getElementById("cmp-" + id);
  const NAMES = {
    "template": ["A · Template", "hand-written baseline"],
    "t5": ["B · Flan-T5", "fine-tuned, deterministic"],
    "t5-guided": ["B+ · Flan-T5 guided", "sampled, checker-filtered"],
    "llm": ["C · LLM", "few-shot, local Ollama"],
  };
  const state = { kind: "rep", backends: new Set(["template", "t5", "t5-guided", "llm"]), available: {} };
  // running faithfulness score per generator since page load (memory only)
  const score = {};
  const resetScore = () => { for (const b of Object.keys(NAMES)) score[b] = { ok: 0, n: 0 }; };
  resetScore();

  const scoreLine = (b) => score[b].n ? `Session score: ${score[b].ok}/${score[b].n} faithful` : "";

  function renderScore() {
    const any = Object.values(score).some((s) => s.n);
    $("score").innerHTML = !any ? `<span class="muted small">Run comparisons to build a score.</span>`
      : Object.keys(NAMES).map((b) => {
        const { ok, n } = score[b];
        const short = NAMES[b][0].split(" · ")[0];
        const pct = n ? Math.round(100 * ok / n) : 0;
        const level = pct >= 90 ? "good" : pct >= 50 ? "warn" : "bad";
        return `<div class="item"><div class="top"><span class="name">${short}</span>
          <span class="val">${n ? `Faithful ${ok}/${n} · ${pct}%` : `<span class="muted">no results</span>`}</span></div>
          <div class="track"><div class="fill ${level}" style="width:${pct}%"></div></div></div>`;
      }).join("") + `<button type="button" class="link" id="cmp-resetScore">Reset score</button>`;
    for (const el of $("grid").querySelectorAll(".score-line")) el.textContent = scoreLine(el.dataset.b);
  }
  $("score").addEventListener("click", (e) => {
    if (e.target.id !== "cmp-resetScore") return;
    resetScore(); renderScore();
  });
  renderScore();

  function renderBackendChips() {
    $("backends").innerHTML = Object.keys(NAMES).map((b) => {
      const avail = state.available[b] !== false;
      const on = state.backends.has(b) && avail;
      const short = NAMES[b][0].split(" · ")[0];
      if (!avail) return `<span class="chip na" data-b="${b}" title="not available on this server">${short} · off</span>`;
      return `<span class="chip ${on ? "on" : "off"}" data-b="${b}" title="click to ${on ? "skip" : "include"}">${short}</span>`;
    }).join("");
  }
  $("backends").addEventListener("click", (e) => {
    const b = e.target.closest(".chip")?.dataset.b; if (!b || state.available[b] === false) return;
    state.backends.has(b) ? state.backends.delete(b) : state.backends.add(b);
    renderBackendChips();
  });

  $("kind").addEventListener("click", (e) => {
    const btn = e.target.closest("button"); if (!btn) return;
    [...$("kind").children].forEach((x) => x.classList.toggle("active", x === btn));
    state.kind = btn.dataset.v; newRecord();
  });

  async function init() {
    const [ex, st] = await Promise.all([api("/api/exercises"), api("/api/status")]);
    $("exercise").innerHTML = Object.entries(ex).map(([k, v]) => `<option value="${k}">${esc(v.display.en)}</option>`).join("");
    for (const [b, info] of Object.entries(st.backends)) state.available[b] = !!info.available;
    for (const b of Object.keys(NAMES)) if (state.available[b] === false) state.backends.delete(b);
    renderBackendChips();
    $("exercise").addEventListener("change", newRecord);
    await newRecord();
  }

  async function newRecord() {
    const ex = $("exercise").value;
    const rec = state.kind === "rep"
      ? await api(`/api/sample/rep?exercise=${ex}&p_fault=${$("pfault").value}`)
      : await api(`/api/sample/session?exercise=${ex}&sets=2&reps_per_set=10`);
    $("json").value = JSON.stringify(rec, null, 2);
    $("grid").innerHTML = "";
    $("facts").innerHTML = `<h2>What the analysis found</h2><div class="muted small">Click Compare to generate feedback.</div>`;
  }
  $("randomBtn").addEventListener("click", newRecord);

  function renderFacts(f) {
    const side = f.side ? ` · ${f.side} side` : "";
    if (f.kind === "rep") {
      const present = f.present.map((p) => `<span class="chip fault">${esc(p.label)} ${p.p.toFixed(2)}${p.deficit ? ` · ${p.deficit}°` : ""}</span>`).join("")
        || `<span class="muted small">none, a clean rep</span>`;
      const absent = f.absent.map((l) => `<span class="chip clear">${esc(l)}</span>`).join("");
      $("facts").innerHTML = `<h2>What the analysis found</h2><h3>${esc(f.exercise)}${side}</h3>
        <div class="group"><div>Detected faults (probability · degrees short)</div><div class="chips">${present}</div></div>
        <div class="group"><div>Checked and fine</div><div class="chips">${absent || '<span class="muted small">-</span>'}</div></div>
        <div class="group"><div>A cue must mention</div><div class="small">${f.required.map(esc).join(", ") || "nothing (praise is fine)"}</div></div>`;
    } else {
      const counts = f.counts.map((c) => `<span class="chip fault">${esc(c.label)} · ${c.count}</span>`).join("")
        || `<span class="muted small">no faults</span>`;
      $("facts").innerHTML = `<h2>What the analysis found</h2><h3>${esc(f.exercise)}${side}</h3>
        <div class="group"><div>${f.total} reps · ${f.sets} sets</div><div class="chips">${counts}</div></div>
        ${f.fatigue ? `<div class="group"><div>Fatigue pattern</div><div class="small">${esc(f.fatigue)}</div></div>` : ""}
        <div class="group"><div>A summary must mention</div><div class="small">${f.required.map(esc).join(", ") || "nothing in particular"}</div></div>`;
    }
  }

  function placeholder(b) {
    const slow = b === "llm" ? " (the LLM takes several seconds)" : "";
    return `<div class="card" id="cmp-card-${b}"><div class="head"><span class="name">${NAMES[b][0]}</span><span class="sub">${NAMES[b][1]}</span></div>
      <div class="meta"><span class="spinner"></span> generating${slow}…</div></div>`;
  }

  function renderCard(r) {
    const [name, sub] = NAMES[r.backend];
    if (!r.available) {
      return `<div class="card"><div class="head"><span class="name">${name}</span><span class="sub">${sub}</span>
        <span class="right"><span class="badge muted">unavailable</span></span></div>
        <div class="unavailable">${esc(r.error)}</div></div>`;
    }
    const ann = r.annotation;
    const sentences = ann.sentences.map((s) => s.problems.length
      ? `<span class="s badline" title="${esc(s.problems.join("; "))}">${esc(s.text)}</span>`
      : `<span class="s">${esc(s.text)}</span>`).join(" ");
    const problems = ann.sentences.flatMap((s) => s.problems);
    const dec = r.decoding && r.decoding.candidates
      ? ` · ${r.decoding.candidates} samples, ${r.decoding.distinct} distinct, ${r.decoding.faithful} passed` : "";
    return `<div class="card ${r.faithful ? "" : "rejected"}">
      <div class="head"><span class="name">${name}</span><span class="sub">${sub}</span>
        <span class="right"><span class="badge ${r.faithful ? "good" : "bad"}">${r.faithful ? "Faithful" : "Rejected"}</span></span></div>
      <div class="sent">${sentences}</div>
      ${problems.length ? `<ul class="problems">${problems.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
      ${ann.missed.length ? `<div class="missed">Missed: ${ann.missed.map(esc).join(", ")}</div>` : ""}
      ${r.replacement ? `<div class="replacement"><b>Patient hears instead:</b> ${esc(r.replacement)}</div>` : ""}
      <div class="meta">${Math.round(r.latency_ms).toLocaleString()} ms${dec}</div>
      <div class="meta score-line" data-b="${r.backend}">${scoreLine(r.backend)}</div></div>`;
  }

  $("goBtn").addEventListener("click", async () => {
    $("error").textContent = "";
    let data;
    try { data = JSON.parse($("json").value); } catch (e) { $("error").textContent = "The JSON is not valid: " + e.message; return; }
    const backends = Object.keys(NAMES).filter((b) => state.backends.has(b));
    if (!backends.length) { $("error").textContent = "Pick at least one generator."; return; }
    $("grid").innerHTML = backends.map(placeholder).join("");
    $("goBtn").disabled = true;
    // one request per generator so fast ones appear immediately
    const key = state.kind === "rep" ? "record" : "session";
    let factsShown = false;
    await Promise.all(backends.map(async (b) => {
      try {
        const res = await api("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ [key]: data, backends: [b] }) });
        if (!factsShown) { renderFacts(res.facts); factsShown = true; }
        const r = res.results[0];
        if (r.available) { score[b].n++; if (r.faithful) score[b].ok++; }  // count before rendering so the card shows it
        $(`card-${b}`).outerHTML = renderCard(r);
        renderScore();
      } catch (e) {
        $(`card-${b}`).outerHTML = renderCard({ backend: b, available: false, error: e.message });
      }
    }));
    $("goBtn").disabled = false;
  });

  init().catch((e) => { $("error").textContent = e.message; });
})();
