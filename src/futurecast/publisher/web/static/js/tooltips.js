/**
 * Floating tooltip controller.
 *
 * One shared tooltip element serves every [data-tooltip] (body text) /
 * [data-tooltip-title] (optional heading) element via delegated events.
 * Text is rendered with textContent, never as HTML; use "\n" for line breaks.
 *
 * - Mouse: opens after a short hover delay; moving straight to another
 *   target while one is open switches instantly.
 * - Keyboard: opens on :focus-visible, closes on blur or Escape.
 * - Touch: tap a target to toggle, tap anywhere else to close.
 * - Live re-renders: rows are rebuilt on every stream tick, so when the
 *   anchored element is replaced the tooltip re-attaches to whatever is now
 *   under the pointer instead of flickering or pointing at a detached node.
 */

(function () {
  const SHOW_DELAY_MS = 350;
  const WARM_WINDOW_MS = 300;
  const GAP = 8;
  const EDGE = 12;
  const ARROW_INSET = 14;

  let tip = null;
  let titleEl = null;
  let bodyEl = null;
  let arrowEl = null;

  let target = null;
  let openedBy = null;
  let pendingTarget = null;
  let showTimer = 0;
  let lastHiddenAt = 0;
  let pointer = null;

  function init() {
    tip = document.getElementById("global-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "global-tooltip";
      tip.className = "fc-tooltip";
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    tip.setAttribute("aria-hidden", "true");
    tip.innerHTML = `<div class="fc-tooltip-title"></div><div class="fc-tooltip-body"></div><span class="fc-tooltip-arrow"></span>`;
    titleEl = tip.querySelector(".fc-tooltip-title");
    bodyEl = tip.querySelector(".fc-tooltip-body");
    arrowEl = tip.querySelector(".fc-tooltip-arrow");

    document.addEventListener("pointermove", onPointerMove, { passive: true });
    document.addEventListener("pointerover", onPointerOver);
    document.addEventListener("pointerout", onPointerOut);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") hide();
    });
    document.addEventListener("scroll", hide, { capture: true, passive: true });
    window.addEventListener("resize", hide);

    new MutationObserver(onDomChange).observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-tooltip", "data-tooltip-title"],
    });
  }

  function findTarget(node) {
    if (!node || node.nodeType !== 1) return null;
    const el = node.closest("[data-tooltip]");
    return el && el.getAttribute("data-tooltip") ? el : null;
  }

  function elementUnderPointer() {
    if (!pointer) return null;
    return findTarget(document.elementFromPoint(pointer.x, pointer.y));
  }

  // ───────── Event handlers ─────────

  function onPointerMove(e) {
    if (e.pointerType === "mouse" || e.pointerType === "pen") {
      pointer = { x: e.clientX, y: e.clientY };
    }
  }

  function onPointerOver(e) {
    if (e.pointerType === "touch") return;
    const el = findTarget(e.target);
    if (!el || el === target || el === pendingTarget) return;
    schedule(el);
  }

  function onPointerOut(e) {
    if (e.pointerType === "touch" || openedBy === "focus") return;
    const current = target || pendingTarget;
    if (!current) return;
    const next = e.relatedTarget;
    if (next && current.contains(next)) return;
    // Moving onto another target: its pointerover switches the tooltip over.
    if (findTarget(next)) return;
    hide();
  }

  function onPointerDown(e) {
    if (e.pointerType !== "touch") return;
    const el = findTarget(e.target);
    if (el && el !== target) {
      show(el, "touch");
    } else {
      hide();
    }
  }

  function onFocusIn(e) {
    const el = findTarget(e.target);
    if (el && e.target.matches(":focus-visible")) show(el, "focus");
  }

  function onFocusOut(e) {
    if (openedBy === "focus" && target && target.contains(e.target)) hide();
  }

  function onDomChange(mutations) {
    // The open tooltip's own text changed (e.g. the theme toggle's label): refresh in place.
    if (target && target.isConnected && mutations.some((m) => m.type === "attributes" && m.target === target)) {
      show(target, openedBy);
      return;
    }
    if (pendingTarget && !pendingTarget.isConnected) {
      clearTimeout(showTimer);
      pendingTarget = null;
      const replacement = elementUnderPointer();
      if (replacement) schedule(replacement);
    }
    if (target && !target.isConnected) {
      const replacement = openedBy === "hover" ? elementUnderPointer() : null;
      if (replacement) {
        show(replacement, "hover");
      } else {
        hide();
      }
    }
  }

  // ───────── Show / hide ─────────

  function schedule(el) {
    clearTimeout(showTimer);
    const warm = target !== null || performance.now() - lastHiddenAt < WARM_WINDOW_MS;
    if (warm) {
      show(el, "hover");
      return;
    }
    pendingTarget = el;
    showTimer = setTimeout(() => {
      pendingTarget = null;
      if (el.isConnected) show(el, "hover");
    }, SHOW_DELAY_MS);
  }

  function show(el, how) {
    clearTimeout(showTimer);
    pendingTarget = null;

    const body = el.getAttribute("data-tooltip");
    if (!body) return;
    const title = el.getAttribute("data-tooltip-title") || "";

    if (target && target !== el) target.removeAttribute("aria-describedby");
    target = el;
    openedBy = how;
    el.setAttribute("aria-describedby", tip.id);

    titleEl.textContent = title;
    titleEl.hidden = !title;
    bodyEl.textContent = body;

    position(el);
    tip.classList.add("visible");
    tip.setAttribute("aria-hidden", "false");
  }

  function hide() {
    clearTimeout(showTimer);
    pendingTarget = null;
    if (!target) return;
    target.removeAttribute("aria-describedby");
    target = null;
    openedBy = null;
    tip.classList.remove("visible");
    tip.setAttribute("aria-hidden", "true");
    lastHiddenAt = performance.now();
  }

  function position(el) {
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;

    let side = "top";
    let top = r.top - h - GAP;
    if (top < EDGE) {
      side = "bottom";
      top = r.bottom + GAP;
    }
    if (top + h > vh - EDGE) {
      side = "none";
      top = Math.max(EDGE, vh - h - EDGE);
    }

    const anchorX = r.left + r.width / 2;
    const left = Math.min(Math.max(EDGE, anchorX - w / 2), Math.max(EDGE, vw - w - EDGE));
    const arrowX = Math.min(Math.max(ARROW_INSET, anchorX - left), w - ARROW_INSET);

    tip.dataset.side = side;
    tip.style.left = `${Math.round(left)}px`;
    tip.style.top = `${Math.round(top)}px`;
    arrowEl.style.left = `${Math.round(arrowX)}px`;
  }

  window.addEventListener("DOMContentLoaded", init);
})();
