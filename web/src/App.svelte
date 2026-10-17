<script lang="ts">
  import { app, start, stopLive } from "./lib/live.svelte";
  import Glossary from "./components/Glossary.svelte";
  import Header from "./components/Header.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import Toolbar from "./components/Toolbar.svelte";
  import Tooltip from "./components/Tooltip.svelte";
  import Watchlist from "./components/Watchlist.svelte";

  start();
  // For previewing states from the console: futurecast.stopLive(), then edit futurecast.app.coins
  Object.assign(window, { futurecast: { app, stopLive } });

  // "/" or Ctrl/Cmd+K opens the add-pair drawer, Escape closes it
  function onKeydown(e: KeyboardEvent): void {
    const shortcut = e.key === "/" || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k");
    const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName || "");
    if (shortcut && !typing) {
      e.preventDefault();
      app.sidebarOpen = true;
    } else if (e.key === "Escape") {
      app.sidebarOpen = false;
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

<Header />
<main class="mx-auto w-full max-w-(--page-w) flex-[1_0_auto] px-6 pt-6 pb-8 max-sm:px-4 max-sm:pt-4">
  <Toolbar />
  <Watchlist />
</main>
<Glossary />
<Tooltip />
<Sidebar />
