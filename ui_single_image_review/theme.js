(function () {
  var storageKey = "cv-single-ui-theme";
  var THEME_CYCLE = ["blue", "berry", "mono"];
  var THEME_LABELS = { blue: "berry", berry: "mono", mono: "blue" };
  var body = document.body;
  var toggle = null;

  function normalizeTheme(theme) {
    return THEME_CYCLE.indexOf(theme) !== -1 ? theme : "blue";
  }

  function applyTheme(theme) {
    theme = normalizeTheme(theme);
    body.setAttribute("data-theme", theme);
    localStorage.setItem(storageKey, theme);
    if (toggle) {
      var next = THEME_LABELS[theme] || "berry";
      toggle.setAttribute("aria-label", "Switch to " + next + " theme");
    }
  }

  function cycleTheme() {
    var current = normalizeTheme(body.getAttribute("data-theme") || "blue");
    var nextIndex = (THEME_CYCLE.indexOf(current) + 1) % THEME_CYCLE.length;
    applyTheme(THEME_CYCLE[nextIndex]);
  }

  document.addEventListener("DOMContentLoaded", function () {
    toggle = document.getElementById("themeToggle");
    var saved = localStorage.getItem(storageKey);
    applyTheme(normalizeTheme(saved));

    if (toggle) {
      toggle.addEventListener("click", cycleTheme);
    }
  });
})();
