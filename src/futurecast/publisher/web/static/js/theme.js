/**
 * Light / dark theme controller.
 *
 * The inline script in <head> sets <html data-theme> before first paint.
 * This keeps the toggle button in sync, persists an explicit choice, and
 * follows the OS setting until the user picks a theme themselves.
 */

(function () {
  const STORAGE_KEY = "fc-theme";
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function savedTheme() {
    try {
      const value = localStorage.getItem(STORAGE_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch (err) {
      return null;
    }
  }

  function currentTheme() {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      const label = `Switch to ${theme === "dark" ? "light" : "dark"} theme`;
      btn.setAttribute("aria-label", label);
      btn.setAttribute("data-tooltip", label);
    }
  }

  window.toggleTheme = function () {
    const next = currentTheme() === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch (err) {
      // Storage blocked (private mode etc.): the toggle still works for this page view.
    }
    applyTheme(next);
  };

  if (media && media.addEventListener) {
    media.addEventListener("change", (e) => {
      if (!savedTheme()) applyTheme(e.matches ? "dark" : "light");
    });
  }

  window.addEventListener("DOMContentLoaded", () => applyTheme(currentTheme()));
})();
