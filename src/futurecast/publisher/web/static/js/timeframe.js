/**
 * Horizon selector: the window the watchlist's momentum columns and signal are measured over.
 * The server computes every horizon; switching only changes which one the rows show.
 */

let currentTimeframe = "4h";

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
