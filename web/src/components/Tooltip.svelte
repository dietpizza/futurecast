<script lang="ts">
  import { flushSync } from "svelte";

  /**
   * One floating tooltip for every [data-tooltip] (body text) / [data-tooltip-title] (optional heading)
   * element on the page, through delegated events. Text is plain; "\n" breaks a line.
   *
   * - Mouse: opens after a short hover delay; moving straight to another target while one is open switches at once.
   * - Keyboard: opens on :focus-visible, closes on blur or Escape.
   * - Touch: tap a target to toggle, tap anywhere else to close.
   * - Stream updates: when the anchored element is replaced, the tooltip moves to whatever is now under the
   *   pointer instead of pointing at a detached node; when its text changes, it updates in place.
   */
  const SHOW_DELAY_MS = 350;
  const WARM_WINDOW_MS = 300;
  const GAP = 8;
  const EDGE = 12;
  const ARROW_INSET = 14;

  let tip: HTMLDivElement;
  let title = $state("");
  let body = $state("");
  let visible = $state(false);
  let side = $state<"top" | "bottom" | "none">("top");
  let left = $state(0);
  let top = $state(0);
  let arrowX = $state(0);

  let target: Element | null = null;
  let openedBy: "hover" | "focus" | "touch" | null = null;
  let pendingTarget: Element | null = null;
  let showTimer = 0;
  let lastHiddenAt = 0;
  let pointer: { x: number; y: number } | null = null;

  function findTarget(node: EventTarget | null): Element | null {
    if (!(node instanceof Element)) return null;
    const el = node.closest("[data-tooltip]");
    return el && el.getAttribute("data-tooltip") ? el : null;
  }

  function elementUnderPointer(): Element | null {
    return pointer ? findTarget(document.elementFromPoint(pointer.x, pointer.y)) : null;
  }

  function onPointerMove(e: PointerEvent): void {
    if (e.pointerType === "mouse" || e.pointerType === "pen") pointer = { x: e.clientX, y: e.clientY };
  }

  function onPointerOver(e: PointerEvent): void {
    if (e.pointerType === "touch") return;
    const el = findTarget(e.target);
    if (!el || el === target || el === pendingTarget) return;
    schedule(el);
  }

  function onPointerOut(e: PointerEvent): void {
    if (e.pointerType === "touch" || openedBy === "focus") return;
    const current = target || pendingTarget;
    if (!current) return;
    const next = e.relatedTarget as Node | null;
    if (next && current.contains(next)) return;
    // Moving onto another target: its pointerover switches the tooltip over.
    if (findTarget(next)) return;
    hide();
  }

  function onPointerDown(e: PointerEvent): void {
    if (e.pointerType !== "touch") return;
    const el = findTarget(e.target);
    if (el && el !== target) show(el, "touch");
    else hide();
  }

  function onFocusIn(e: FocusEvent): void {
    const el = findTarget(e.target);
    if (el && (e.target as Element).matches(":focus-visible")) show(el, "focus");
  }

  function onFocusOut(e: FocusEvent): void {
    if (openedBy === "focus" && target && target.contains(e.target as Node)) hide();
  }

  function onDomChange(mutations: MutationRecord[]): void {
    if (target && target.isConnected && mutations.some((m) => m.type === "attributes" && m.target === target)) {
      show(target, openedBy!);
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
      if (replacement) show(replacement, "hover");
      else hide();
    }
  }

  function schedule(el: Element): void {
    clearTimeout(showTimer);
    const warm = target !== null || performance.now() - lastHiddenAt < WARM_WINDOW_MS;
    if (warm) {
      show(el, "hover");
      return;
    }
    pendingTarget = el;
    showTimer = window.setTimeout(() => {
      pendingTarget = null;
      if (el.isConnected) show(el, "hover");
    }, SHOW_DELAY_MS);
  }

  function show(el: Element, how: "hover" | "focus" | "touch"): void {
    clearTimeout(showTimer);
    pendingTarget = null;
    const text = el.getAttribute("data-tooltip");
    if (!text) return;
    if (target && target !== el) target.removeAttribute("aria-describedby");
    target = el;
    openedBy = how;
    el.setAttribute("aria-describedby", "global-tooltip");
    title = el.getAttribute("data-tooltip-title") || "";
    body = text;
    flushSync(); // measure with the new text before placing it
    position(el);
    visible = true;
  }

  function hide(): void {
    clearTimeout(showTimer);
    pendingTarget = null;
    if (!target) return;
    target.removeAttribute("aria-describedby");
    target = null;
    openedBy = null;
    visible = false;
    lastHiddenAt = performance.now();
  }

  function position(el: Element): void {
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;

    let s: typeof side = "top";
    let t = r.top - h - GAP;
    if (t < EDGE) {
      s = "bottom";
      t = r.bottom + GAP;
    }
    if (t + h > vh - EDGE) {
      s = "none";
      t = Math.max(EDGE, vh - h - EDGE);
    }
    const anchorX = r.left + r.width / 2;
    const l = Math.min(Math.max(EDGE, anchorX - w / 2), Math.max(EDGE, vw - w - EDGE));
    side = s;
    top = Math.round(t);
    left = Math.round(l);
    arrowX = Math.round(Math.min(Math.max(ARROW_INSET, anchorX - l), w - ARROW_INSET));
  }

  $effect(() => {
    const observer = new MutationObserver(onDomChange);
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-tooltip", "data-tooltip-title"] });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && hide();
    document.addEventListener("pointermove", onPointerMove, { passive: true });
    document.addEventListener("pointerover", onPointerOver);
    document.addEventListener("pointerout", onPointerOut);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("keydown", onKey);
    document.addEventListener("scroll", hide, { capture: true, passive: true });
    window.addEventListener("resize", hide);
    return () => {
      observer.disconnect();
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerover", onPointerOver);
      document.removeEventListener("pointerout", onPointerOut);
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("scroll", hide, { capture: true });
      window.removeEventListener("resize", hide);
    };
  });
</script>

<div
  bind:this={tip}
  id="global-tooltip"
  class="fc-tooltip"
  class:visible
  role="tooltip"
  aria-hidden={!visible}
  data-side={side}
  style:left="{left}px"
  style:top="{top}px"
>
  {#if title}<div class="mb-[3px] text-sm font-bold tracking-[0.02em] text-accent-text">{title}</div>{/if}
  <div class="whitespace-pre-line text-fg-2 tabular-nums">{body}</div>
  <span class="arrow" style:left="{arrowX}px"></span>
</div>

<style>
  .fc-tooltip {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 99999;
    pointer-events: none;
    width: max-content;
    max-width: 280px;
    padding: 8px 11px 8px 12px;
    background: var(--surface-raised);
    border: 1px solid var(--rule);
    box-shadow: inset 2px 0 0 var(--accent), var(--shadow-lg);
    font-size: 13.5px;
    line-height: 1.45;
    opacity: 0;
    visibility: hidden;
    transform: translateY(4px);
    transition: opacity 0.12s ease-out, transform 0.12s ease-out, visibility 0s linear 0.12s;
  }

  .fc-tooltip[data-side="bottom"] {
    transform: translateY(-4px);
  }

  .fc-tooltip.visible {
    opacity: 1;
    visibility: visible;
    transform: none;
    transition: opacity 0.12s ease-out, transform 0.12s ease-out, visibility 0s;
  }

  .arrow {
    position: absolute;
    width: 9px;
    height: 9px;
    margin-left: -5px;
    background: var(--surface-raised);
    border: 1px solid var(--rule);
    transform: rotate(45deg);
  }

  [data-side="top"] .arrow {
    bottom: -5px;
    border-top: 0;
    border-left: 0;
  }

  [data-side="bottom"] .arrow {
    top: -5px;
    border-bottom: 0;
    border-right: 0;
  }

  [data-side="none"] .arrow {
    display: none;
  }

  @media (prefers-reduced-motion: reduce) {
    .fc-tooltip,
    .fc-tooltip[data-side="bottom"] {
      transform: none;
    }
  }
</style>
