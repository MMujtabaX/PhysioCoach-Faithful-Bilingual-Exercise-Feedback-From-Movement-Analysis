// Tab router: shows one panel at a time (the others are only hidden, never rebuilt), driven by the URL hash.
(() => {
  const TABS = ["demo", "compare", "live"];
  const nav = document.getElementById("nav");

  // the old addresses /compare and /live serve this page; normalise them to /#compare and /#live
  const fromPath = { "/compare": "compare", "/live": "live" }[location.pathname.replace(/\/+$/, "")];
  if (location.pathname !== "/") history.replaceState(null, "", "/" + (location.hash || (fromPath ? "#" + fromPath : "")));

  function show() {
    const want = location.hash.slice(1);
    const tab = TABS.includes(want) ? want : "demo";
    for (const t of TABS) document.getElementById(`tab-${t}`).hidden = t !== tab;
    for (const a of nav.querySelectorAll("a[data-tab]")) {
      const on = a.dataset.tab === tab;
      a.classList.toggle("active", on);
      on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
    }
    const title = document.getElementById(`tab-${tab}`).dataset.title;
    document.title = `${title} · PhysioCoach-NLP`;
  }

  window.addEventListener("hashchange", () => { show(); window.scrollTo(0, 0); });
  show();
})();
