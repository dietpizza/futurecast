/**
 * Horizon selector. Each horizon has its own trained win-probability and risk models on the
 * server; switching only changes which one the watchlist shows.
 */

let currentTimeframe = "15m";

function setTimeframe(tf) {
  if (currentTimeframe === tf) return;
  currentTimeframe = tf;
  document.querySelectorAll(".tf-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-tf") === tf);
  });
  if (typeof renderAll === "function") {
    renderAll();
  }
}
