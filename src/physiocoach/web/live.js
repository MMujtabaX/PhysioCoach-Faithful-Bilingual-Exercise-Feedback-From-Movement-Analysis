// Live session tab: a simulated workout with a spoken cue after every rep and a summary at the end.
(() => {
  const { api, post, esc, sleep } = window.PC;
  const $ = (id) => document.getElementById("live-" + id);
  const panel = document.getElementById("tab-live");
  const state = { lang: "en", running: false, exercises: {}, threshold: 0.5 };

  $("lang").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || state.running) return;
    [...$("lang").children].forEach((x) => x.classList.toggle("active", x === b));
    state.lang = b.dataset.v;
  });

  async function init() {
    const [ex, st] = await Promise.all([api("/api/exercises"), api("/api/status")]);
    state.exercises = ex;
    $("exercise").innerHTML = Object.entries(ex).map(([k, v]) => `<option value="${k}">${esc(v.display.en)}</option>`).join("");
    for (const opt of $("backend").options) {
      const info = st.backends[opt.value];
      if (info && !info.available) { opt.disabled = true; opt.textContent += " (not available)"; }
    }
    if (st.tts && st.tts.configured === "none") { $("speak").checked = false; $("speak").disabled = true; }
  }

  function playAudio(url) {
    return new Promise((resolve) => {
      const a = $("audio");
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      a.onended = finish; a.onerror = finish;
      a.src = url;
      a.play().catch(finish);
      setTimeout(finish, 12000);  // never hang on a stuck clip
    });
  }

  function setCue(text, lang) {
    const c = $("cue");
    c.textContent = text;
    c.className = "cue" + (lang === "ur" ? " ur" : "");
    c.lang = lang === "ur" ? "ur" : "en";
  }

  function clearCurrent() { panel.querySelectorAll(".tile.current").forEach((t) => t.classList.remove("current")); }

  function buildTiles(session) {
    const bySet = {};
    session.reps.forEach((r, i) => (bySet[r.set] = bySet[r.set] || []).push(i));
    $("tilesWrap").innerHTML = Object.entries(bySet).map(([s, idx]) =>
      `<div class="set-label">Set ${s}</div><div class="tiles">${idx.map((i) => `<div class="tile" id="live-tile-${i}"></div>`).join("")}</div>`).join("");
  }

  function updateTally(counts, labels) {
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const max = Math.max(1, ...entries.map((e) => e[1]));
    $("tally").innerHTML = entries.length ? entries.map(([code, n]) =>
      `<div class="bar"><span>${esc(labels[code] || code)}</span><div class="track"><div class="fill" style="width:${(100 * n / max).toFixed(0)}%"></div></div><span class="n">${n}</span></div>`).join("")
      : `<div class="muted small">No faults yet. Nice form.</div>`;
  }

  async function run() {
    $("error").textContent = ""; $("summary").hidden = true;
    state.running = true; $("startBtn").disabled = true; $("stopBtn").disabled = false;
    const exercise = $("exercise").value;
    const sets = Math.min(4, Math.max(1, +$("sets").value || 2));
    const reps = Math.min(15, Math.max(3, +$("reps").value || 8));
    const backend = $("backend").value;
    let speak = $("speak").checked;
    const pace = +$("pace").value;
    const lang = state.lang;
    const labels = state.exercises[exercise].faults;
    try {
      const session = await api(`/api/sample/session?exercise=${exercise}&sets=${sets}&reps_per_set=${reps}`);
      buildTiles(session);
      const counts = {};
      updateTally(counts, labels);
      for (let i = 0; i < session.reps.length && state.running; i++) {
        const rec = session.reps[i];
        $("where").textContent = `Set ${rec.set} · Rep ${rec.rep}`;
        $("of").textContent = `of ${reps}${session.side ? ` · ${session.side} side` : ""}`;
        clearCurrent();
        $(`tile-${i}`).classList.add("current");
        $("bar").style.width = `${(100 * i / session.reps.length).toFixed(1)}%`;
        await sleep(pace * 0.4);  // "doing the rep"
        if (!state.running) break;

        let fb;
        try {
          fb = await post("/api/feedback/rep", { record: rec, lang, backend, speak, live: true });
        } catch (e) {
          if (speak && /speech/i.test(e.message)) {  // keep going without audio
            speak = false; $("error").textContent = "Speech is not available, continuing without audio. " + e.message;
            fb = await post("/api/feedback/rep", { record: rec, lang, backend, speak: false, live: true });
          } else throw e;
        }
        const present = Object.entries(rec.faults).filter(([, p]) => p >= state.threshold).map(([c]) => c);
        present.forEach((c) => (counts[c] = (counts[c] || 0) + 1));
        $(`tile-${i}`).classList.add(present.length ? "faulty" : "clean");
        $(`tile-${i}`).title = `Set ${rec.set} rep ${rec.rep}: ${fb.text}`;
        setCue(fb.text, lang);
        $("repFaults").innerHTML = present.length
          ? present.map((c) => `<span class="chip fault">${esc(labels[c] || c)}</span>`).join("")
          : `<span class="chip clear">clean rep</span>`;
        const t = fb.timings_ms;
        $("meta").textContent = `${fb.backend}${fb.fallback ? " → safe template" : ""} · generated in ${Math.round(t.generate)} ms` +
          (t.speak !== undefined ? ` · speech ${Math.round(t.speak)} ms${t.speak_cached ? " (cached)" : ""}` : "");
        updateTally(counts, labels);
        const started = performance.now();
        if (speak && fb.audio_url) await playAudio(fb.audio_url);
        await sleep(Math.max(0, pace * 0.6 - (performance.now() - started)));
      }
      $("bar").style.width = "100%";
      clearCurrent();
      if (state.running) {
        $("where").textContent = "Session complete"; $("of").textContent = "";
        setCue("Preparing your summary…", "en");
        const sb = await post("/api/feedback/session", { session, lang, backend, speak, live: true });
        $("summary").hidden = false;
        $("summaryText").textContent = sb.text;
        $("summaryText").className = "text" + (lang === "ur" ? " ur" : "");
        $("summaryMeta").textContent = `${sb.backend}${sb.fallback ? " → safe template" : ""} · ${Math.round(sb.timings_ms.generate)} ms`;
        setCue("Well done. Your summary is below.", "en");
        if (speak && sb.audio_url) await playAudio(sb.audio_url);
      }
    } catch (e) {
      $("error").textContent = e.message;
    } finally {
      state.running = false; $("startBtn").disabled = false; $("stopBtn").disabled = true;
    }
  }

  $("startBtn").addEventListener("click", run);
  $("stopBtn").addEventListener("click", () => {
    state.running = false; $("audio").pause();
    $("where").textContent = "Stopped"; $("of").textContent = "";
  });
  init().catch((e) => { $("error").textContent = e.message; });
})();
