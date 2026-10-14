/**
 * CoinDCX Futures Search & Explorer Sidebar Controller.
 * Provides instant search, sector filtering, and 1-click coin tracking.
 */

let coindcxCatalog = [];
let activeSidebarSector = "ALL";
let currentSidebarQuery = "";

// Open / Close / Toggle Sidebar
function openSidebar() {
  const sidebar = document.getElementById("coindcx-sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const input = document.getElementById("sidebar-search-input");

  if (sidebar && backdrop) {
    sidebar.classList.add("open");
    backdrop.classList.add("active");
    if (input) {
      setTimeout(() => input.focus(), 150);
    }
    renderSidebarList();
  }
}

function closeSidebar() {
  const sidebar = document.getElementById("coindcx-sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");

  if (sidebar && backdrop) {
    sidebar.classList.remove("open");
    backdrop.classList.remove("active");
  }
}

function toggleSidebar() {
  const sidebar = document.getElementById("coindcx-sidebar");
  if (sidebar && sidebar.classList.contains("open")) {
    closeSidebar();
  } else {
    openSidebar();
  }
}

// Fetch Catalog from Backend
async function loadCoinDCXCatalog() {
  try {
    const resp = await fetch("/api/coindcx/futures");
    if (resp.ok) {
      coindcxCatalog = await resp.json();
      const countEl = document.getElementById("sidebar-total-count");
      if (countEl) {
        countEl.textContent = `${coindcxCatalog.length} CoinDCX futures`;
      }
      renderSidebarList();
    }
  } catch (err) {
    console.error("Failed to load CoinDCX futures catalog:", err);
  }
}

// Sector Tab Filter
function setSidebarSector(sector) {
  activeSidebarSector = sector.toUpperCase();
  document.querySelectorAll(".sidebar-tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-sector") === activeSidebarSector);
  });
  renderSidebarList();
}

// Search Handler
function handleSidebarSearch(e) {
  let val = (e.target.value || "").trim();
  if (val.toLowerCase().includes("coindcx.com/")) {
    const parts = val.split("/");
    val = parts[parts.length - 1].split("?")[0].split("#")[0];
  }
  currentSidebarQuery = val.trim().toLowerCase();
  const clearBtn = document.getElementById("clear-search-btn");
  if (clearBtn) {
    clearBtn.style.display = currentSidebarQuery ? "grid" : "none";
  }
  renderSidebarList();
}

function clearSidebarSearch() {
  const input = document.getElementById("sidebar-search-input");
  const clearBtn = document.getElementById("clear-search-btn");
  if (input) {
    input.value = "";
    currentSidebarQuery = "";
  }
  if (clearBtn) {
    clearBtn.style.display = "none";
  }
  renderSidebarList();
}

// Add Coin from Sidebar
async function addFromSidebar(symbol) {
  if (typeof quickAdd === "function") {
    await quickAdd(symbol);
    renderSidebarList();
  }
}

// Render Coin List inside Sidebar
function renderSidebarList() {
  const listContainer = document.getElementById("sidebar-coin-list");
  if (!listContainer) return;

  const q = currentSidebarQuery;
  const sector = activeSidebarSector;

  const filtered = coindcxCatalog.filter((item) => {
    if (sector !== "ALL" && item.sector !== sector) return false;
    if (!q) return true;
    return [item.symbol, item.base, item.name, item.description, item.sector]
      .some((field) => (field || "").toLowerCase().includes(q));
  });

  if (filtered.length === 0) {
    let customOptionHtml = "";
    if (q) {
      const normalized = typeof normalizeCoinDCXSymbol === "function" ? normalizeCoinDCXSymbol(q) : `B-${q.toUpperCase()}_USDT`;
      customOptionHtml = `
        <button class="btn btn-primary" onclick="addFromSidebar('${esc(normalized)}')">Track ${esc(normalized)}</button>
      `;
    }

    listContainer.innerHTML = `
      <div class="sidebar-empty-state">
        <div class="sidebar-empty-title">No matching pairs</div>
        <p>${q ? "Not in the catalog? You can still track the symbol directly." : "Try another sector or search term."}</p>
        ${customOptionHtml}
      </div>
    `;
    return;
  }

  const html = filtered.map((coin) => {
    const isAdded = Boolean(window.coinsData && window.coinsData[coin.symbol]);
    const action = isAdded
      ? `<button class="btn btn-sm btn-add-coin is-added" disabled>Added</button>`
      : `<button class="btn btn-sm btn-add-coin" onclick="addFromSidebar('${esc(coin.symbol)}')">Add</button>`;

    return `
      <div class="sidebar-coin">
        <div class="sidebar-coin-main">
          <div class="sidebar-coin-title">
            <span class="sidebar-symbol">${esc(coin.base || coin.symbol)}</span>
            <span class="sidebar-name">${esc(coin.name)}</span>
            <span class="sidebar-sector">${esc(typeof sectorLabel === "function" ? sectorLabel(coin.sector) : coin.sector)}</span>
          </div>
          <div class="sidebar-desc" title="${esc(coin.description)}">${esc(coin.description)}</div>
        </div>
        <div class="sidebar-coin-actions">
          <a class="icon-btn" href="${esc(coin.coindcx_url)}" target="_blank" rel="noopener noreferrer"
             aria-label="Open ${esc(coin.symbol)} on CoinDCX" title="Open on CoinDCX">
            <svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3H3.5v9.5H13V10M9 3h4v4M13 3L7.5 8.5" /></svg>
          </a>
          ${action}
        </div>
      </div>
    `;
  }).join("");

  listContainer.innerHTML = html;
}

// Global Keyboard Shortcuts
document.addEventListener("keydown", (e) => {
  // Press '/' or Ctrl+K / Cmd+K to open search sidebar
  const isSearchShortcut = e.key === "/" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k");
  const isInputFocused = ["INPUT", "TEXTAREA"].includes(document.activeElement.tagName);

  if (isSearchShortcut && !isInputFocused) {
    e.preventDefault();
    openSidebar();
  } else if (e.key === "Escape") {
    closeSidebar();
  }
});

// Bootstrapping
window.addEventListener("DOMContentLoaded", loadCoinDCXCatalog);
