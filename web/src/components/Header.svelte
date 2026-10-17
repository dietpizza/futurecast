<script lang="ts">
  import { app } from "../lib/live.svelte";
  import Icon from "./Icon.svelte";

  const THEME_KEY = "fc-theme";
  const STATUS = { connecting: "Connecting", live: "Live", reconnecting: "Reconnecting" };

  // index.html sets <html data-theme> before first paint; this follows the OS until the user picks one.
  let theme = $state(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
  let now = $state(new Date());

  const themeLabel = $derived(`Switch to ${theme === "dark" ? "light" : "dark"} theme`);

  $effect(() => {
    document.documentElement.dataset.theme = theme;
  });

  $effect(() => {
    const timer = setInterval(() => (now = new Date()), 1000);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const follow = (e: MediaQueryListEvent) => {
      if (!savedTheme()) theme = e.matches ? "dark" : "light";
    };
    media.addEventListener("change", follow);
    return () => {
      clearInterval(timer);
      media.removeEventListener("change", follow);
    };
  });

  function savedTheme(): string | null {
    try {
      const value = localStorage.getItem(THEME_KEY);
      return value === "light" || value === "dark" ? value : null;
    } catch {
      return null;
    }
  }

  function toggleTheme(): void {
    theme = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Storage blocked (private mode etc.): the toggle still works for this page view.
    }
  }
</script>

<header class="sticky top-0 z-50 h-(--header-h) border-b border-border bg-header backdrop-blur-sm backdrop-saturate-180">
  <div class="mx-auto flex h-full max-w-(--page-w) items-center justify-between gap-4 px-6 max-sm:px-4">
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2.5">
        <!-- Same file as the favicon -->
        <img class="block size-7" src="/favicon.svg" alt="" width="28" height="28" />
        <span class="text-[22px] leading-none font-bold tracking-[0.08em] uppercase">Future<span class="text-accent-text">Cast</span></span>
      </div>
      <button id="theme-toggle" class="icon-btn border-border" onclick={toggleTheme} aria-label={themeLabel} data-tooltip={themeLabel}>
        <Icon name={theme === "dark" ? "sun" : "moon"} />
      </button>
    </div>

    <div class="flex items-center gap-4">
      <div
        class="caps group inline-flex items-center gap-2 text-[13px] font-semibold text-muted data-[state=live]:text-pos"
        data-state={app.connection}
        data-tooltip-title="Live data feed"
        data-tooltip="Real-time price and order book stream from the exchange."
      >
        <span
          class="size-[7px] bg-faint group-data-[state=live]:bg-pos-fill group-data-[state=live]:shadow-[0_0_8px_var(--pos-fill)] group-data-[state=reconnecting]:animate-[blink_1s_ease-in-out_infinite] group-data-[state=reconnecting]:bg-warn"
          aria-hidden="true"
        ></span>
        <span class="max-sm:hidden">{STATUS[app.connection]}</span>
      </div>
      <!-- Rajdhani's digits vary in width: a fixed minimum keeps the ticking clock from nudging its neighbours -->
      <span class="inline-block min-w-[6em] text-right text-sm tracking-[0.04em] text-fg-2 tabular-nums max-md:hidden">
        {now.toISOString().slice(11, 19)} UTC
      </span>
      <button
        class="sidebar-toggle-btn btn btn-primary max-sm:w-[34px] max-sm:justify-center max-sm:p-0"
        onclick={() => (app.sidebarOpen = !app.sidebarOpen)}
        aria-keyshortcuts="/"
      >
        <Icon name="plus" />
        <span class="max-sm:hidden">Add pair</span>
        <kbd class="hotkey-hint max-sm:hidden">/</kbd>
      </button>
    </div>
  </div>
</header>
