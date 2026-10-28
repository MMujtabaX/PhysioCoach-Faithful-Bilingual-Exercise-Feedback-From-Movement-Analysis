// Demo tab: one record in, one piece of feedback out.
(() => {
  const { api } = window.PC;
  const $ = (id) => document.getElementById("demo-" + id);
  const state = { kind: "rep", lang: "en" };

  function segment(id, key) {
    $(id).addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      [...$(id).children].forEach((x) => x.classList.toggle("active", x === b));
      state[key] = b.dataset.v;
      if (key === "kind") randomRecord();
    });
  }
  segment("kind", "kind"); segment("lang", "lang");

  async function loadStatus() {
    try {
      const s = await api("/api/status");
      const chips = [
        ["Template", true], ["Flan-T5", s.backends.t5.available], ["LLM", s.backends.llm.available],
        ["Translator: " + s.translator.configured, s.translator.configured !== "none"],
        ["TTS: " + s.tts.configured, s.tts.configured !== "none"],
      ];
      $("status").innerHTML = chips.map(([t, on]) => `<span class="chip ${on ? "ok" : ""}">${t}</span>`).join("");
    } catch (e) { /* status is cosmetic */ }
  }

  async function loadExercises() {
    const ex = await api("/api/exercises");
    $("exercise").innerHTML = Object.entries(ex).map(([k, v]) => `<option value="${k}">${v.display.en}</option>`).join("");
    $("exercise").addEventListener("change", randomRecord);
  }

  async function randomRecord() {
    const ex = $("exercise").value;
    const path = state.kind === "rep"
      ? `/api/sample/rep?exercise=${ex}&p_fault=0.45`
      : `/api/sample/session?exercise=${ex}&sets=2&reps_per_set=10`;
    const rec = await api(path);
    $("json").value = JSON.stringify(rec, null, 2);
  }
  $("randomBtn").addEventListener("click", randomRecord);

  function badge(text, cls) { return `<span class="badge ${cls}">${text}</span>`; }

  $("goBtn").addEventListener("click", async () => {
    $("error").textContent = ""; $("goBtn").disabled = true; $("goBtn").textContent = "Generating…";
    try {
      const data = JSON.parse($("json").value);
      const body = { lang: state.lang, backend: $("backend").value, speak: $("speak").checked, check: $("check").checked,
                     live: state.kind === "rep" && $("live").checked };
      body[state.kind === "rep" ? "record" : "session"] = data;
      const fb = await api(`/api/feedback/${state.kind}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const out = $("out");
      out.textContent = fb.text;
      out.className = "feedback" + (fb.lang === "ur" ? " ur" : "");
      out.lang = fb.lang === "ur" ? "ur" : "en";

      const b = [];
      b.push(fb.faithful_raw ? badge("Generator output faithful", "good") : badge("Generator output failed check", "bad"));
      if (fb.fallback) b.push(badge("Replaced by safe template", "warn"));
      if (fb.translation) {
        const t = fb.translation;
        b.push(t.method === "template"
          ? badge(t.fallback ? "Translation rejected → Urdu template" : (body.live ? "Live: hand-written Urdu" : "Urdu from template"), t.fallback ? "warn" : "good")
          : badge(`Translated (${t.method}${t.cached ? ", cached" : ""})`, "good"));
      }
      $("badges").innerHTML = b.join("");

      const audio = $("audio");
      if (fb.audio_url) { audio.src = fb.audio_url + "?t=" + Date.now(); audio.hidden = false; audio.play().catch(() => {}); }
      else { audio.hidden = true; audio.removeAttribute("src"); }

      const rows = [["Backend", fb.backend]];
      if (fb.lang === "ur") rows.push(["English", fb.text_en]);
      if (fb.raw !== fb.text_en) rows.push(["Raw output", fb.raw]);
      if (fb.decoding && fb.decoding.candidates) rows.push(["Guided decoding",
        `${fb.decoding.candidates} candidates · ${fb.decoding.distinct} distinct · ${fb.decoding.faithful} faithful`]);
      if (fb.check.issues.length) rows.push(["Check issues", fb.check.issues.join("; ")]);
      if (fb.translation && fb.translation.machine_translation) rows.push(["Rejected MT", fb.translation.machine_translation, "ur"]);
      rows.push(["Timing", Object.entries(fb.timings_ms).filter(([k]) => k !== "speak_cached").map(([k, v]) => `${k} ${v} ms`).join(" · ")
        + (fb.timings_ms.speak_cached ? " (audio from cache)" : "")]);
      $("facts").innerHTML = rows.map(([k, v, cls]) => `<dt>${k}</dt><dd class="${cls || ""}">${String(v).replace(/</g, "&lt;")}</dd>`).join("");
      $("raw").textContent = JSON.stringify(fb, null, 2);
    } catch (e) {
      $("error").textContent = e.message;
    } finally {
      $("goBtn").disabled = false; $("goBtn").textContent = "Generate feedback";
    }
  });

  loadStatus(); loadExercises().then(randomRecord).catch((e) => { $("error").textContent = e.message; });
})();
