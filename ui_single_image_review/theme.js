(function () {
  var storageKey = "cv-single-ui-theme";
  var body = document.body;
  var toggle = null;

  function applyTheme(theme) {
    body.setAttribute("data-theme", theme);
    localStorage.setItem(storageKey, theme);
    if (toggle) {
      toggle.setAttribute("aria-label", "Switch to " + (theme === "blue" ? "berry" : "blue") + " theme");
    }
  }

  function toggleTheme() {
    var current = body.getAttribute("data-theme") || "blue";
    applyTheme(current === "blue" ? "berry" : "blue");
  }

  document.addEventListener("DOMContentLoaded", function () {
    toggle = document.getElementById("themeToggle");
    var saved = localStorage.getItem(storageKey);
    applyTheme(saved === "berry" ? "berry" : "blue");

    if (toggle) {
      toggle.addEventListener("click", toggleTheme);
    }
  });
})();
