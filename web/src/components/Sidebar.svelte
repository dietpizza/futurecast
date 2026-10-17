<script lang="ts">
  // Add-pair drawer: searches and filters the whole CoinDCX futures catalog in the page.
  import { addCoin, app } from "../lib/live.svelte";
  import { normalizeCoinDCXSymbol, sectorLabel } from "../lib/format";
  import Icon from "./Icon.svelte";

  const SECTORS = [
    ["ALL", "All"],
    ["MAJOR", "Majors"],
    ["MEME", "Meme"],
    ["L1", "L1"],
    ["DEFI", "DeFi"],
    ["AI", "AI"],
    ["INFRA", "Infra"],
    ["NFT_GAMING", "NFT / Gaming"],
    ["GENERAL_ALT", "Other"],
  ];

  let search = $state("");
  let sector = $state("ALL");
  let input: HTMLInputElement | undefined = $state();

  // A pasted CoinDCX link searches for the pair at its end
  const query = $derived.by(() => {
    let q = search.trim();
    if (q.toLowerCase().includes("coindcx.com/")) {
      const parts = q.split("/");
      q = parts[parts.length - 1].split("?")[0].split("#")[0];
    }
    return q.trim().toLowerCase();
  });

  const matches = $derived(
    app.catalog.filter((item) => {
      if (sector !== "ALL" && item.sector !== sector) return false;
      if (!query) return true;
      return [item.symbol, item.base, item.name, item.description, item.sector].some((f) => (f || "").toLowerCase().includes(query));
    }),
  );

  $effect(() => {
    if (!app.sidebarOpen) return;
    const timer = setTimeout(() => input?.focus(), 150);
    return () => clearTimeout(timer);
  });

  function close(): void {
    app.sidebarOpen = false;
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions (Escape closes it) -->
<div
  class={[
    "fixed inset-0 z-9990 bg-backdrop transition-[opacity,visibility] duration-200",
    app.sidebarOpen ? "visible opacity-100" : "invisible opacity-0",
  ]}
  onclick={close}
></div>

<aside
  class={[
    "coindcx-sidebar fixed inset-y-0 right-0 z-9995 flex w-[420px] max-w-screen flex-col border-l border-border bg-surface shadow-drawer",
    "transition-[translate,visibility] duration-250 ease-[cubic-bezier(0.16,1,0.3,1)]",
    app.sidebarOpen ? "visible translate-x-0" : "invisible translate-x-full",
  ]}
  aria-label="Add a CoinDCX futures pair"
>
  <div class="flex items-center justify-between pt-4 pr-4 pb-3 pl-5">
    <div class="flex items-baseline gap-2.5">
      <h2 class="text-2xl font-semibold tracking-[0.04em] uppercase">Add pair</h2>
      <span class="text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">
        {app.catalog.length ? `${app.catalog.length} CoinDCX futures` : "CoinDCX futures"}
      </span>
    </div>
    <button class="icon-btn" onclick={close} aria-label="Close"><Icon name="close-lg" /></button>
  </div>

  <div class="px-5 pb-3">
    <div class="relative">
      <Icon name="search" class="pointer-events-none absolute top-1/2 left-[11px] -translate-y-1/2 text-faint" />
      <input
        bind:this={input}
        bind:value={search}
        id="sidebar-search-input"
        type="text"
        class="h-[38px] w-full border border-border-strong bg-well pr-9 pl-[34px] text-[14.5px] text-fg transition-[border-color,box-shadow] placeholder:text-faint focus:border-accent focus:shadow-[0_0_0_3px_var(--focus-ring)] focus:outline-none"
        placeholder="Search symbol, name or paste a CoinDCX link"
        autocomplete="off"
        spellcheck="false"
      />
      {#if query}
        <button
          class="absolute top-1/2 right-1.5 grid size-[26px] -translate-y-1/2 place-items-center text-muted hover:bg-neu-soft hover:text-fg"
          onclick={() => { search = ""; input?.focus(); }}
          aria-label="Clear search"
        ><Icon name="close-sm" /></button>
      {/if}
    </div>
  </div>

  <div class="flex gap-1.5 overflow-x-auto border-b border-border px-5 pb-3 [scrollbar-width:none]" role="group" aria-label="Filter by sector">
    {#each SECTORS as [key, label] (key)}
      <button
        class={[
          "h-7 flex-none border px-[11px] text-[13px] font-semibold tracking-[0.08em] uppercase transition-colors",
          key === sector
            ? "border-accent bg-accent font-bold text-on-accent"
            : "border-border text-muted hover:border-rule hover:bg-accent-soft hover:text-accent-text",
        ]}
        aria-pressed={key === sector}
        onclick={() => (sector = key)}
      >{label}</button>
    {/each}
  </div>

  <div class="flex-1 overflow-y-auto overscroll-contain">
    {#each matches as coin (coin.symbol)}
      {@const added = Boolean(app.coins[coin.symbol])}
      <div class="sidebar-coin flex items-center gap-3 border-b border-border px-5 py-[11px]">
        <div class="min-w-0 flex-1">
          <div class="flex min-w-0 items-baseline gap-2">
            <span class="text-[17px] font-bold tracking-[0.02em]">{coin.base || coin.symbol}</span>
            <span class="truncate text-sm text-fg-2">{coin.name}</span>
            <span class="ml-auto flex-none text-xs font-semibold tracking-[0.08em] text-muted uppercase">{sectorLabel(coin.sector)}</span>
          </div>
          <div class="mt-0.5 truncate text-[13.5px] text-dim" title={coin.description}>{coin.description}</div>
        </div>
        <div class="flex flex-none items-center gap-1">
          <a
            class="icon-btn size-7"
            href={coin.coindcx_url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Open {coin.symbol} on CoinDCX"
            title="Open on CoinDCX"
          ><Icon name="external" /></a>
          {#if added}
            <button class="btn btn-sm min-w-16 justify-center border-transparent text-pos hover:border-transparent hover:bg-transparent hover:text-pos" disabled>Added</button>
          {:else}
            <button class="btn btn-sm min-w-16 justify-center" onclick={() => addCoin(coin.symbol)}>Add</button>
          {/if}
        </div>
      </div>
    {:else}
      {@const custom = query ? normalizeCoinDCXSymbol(query) : ""}
      <div class="px-6 py-10 text-center">
        <div class="text-base font-semibold">No matching pairs</div>
        <p class="mt-1 mb-4 text-sm text-dim">
          {query ? "Not in the catalog? You can still track the symbol directly." : "Try another sector or search term."}
        </p>
        {#if custom}
          <button class="btn btn-primary" onclick={() => addCoin(custom)}>Track {custom}</button>
        {/if}
      </div>
    {/each}
  </div>
</aside>

<style>
  .sidebar-coin:hover {
    background: linear-gradient(90deg, var(--accent-soft), transparent 60%), var(--surface-hover);
    box-shadow: inset 2px 0 0 var(--accent);
  }
</style>
