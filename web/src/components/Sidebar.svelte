<script lang="ts">
  // Add-pair drawer: the whole CoinDCX futures catalog as a market board (last price, change over a
  // chosen window, 24h volume), searched, filtered by sector and sorted in the page. Prices come from
  // the Binance perpetual behind each pair (GET /api/market), polled while the drawer is open.
  import { addCoin, app, BOARD_WINDOWS, boardPrefs, loadMarket, remember, selectBoardWindow } from "../lib/live.svelte";
  import { isNum, normalizeCoinDCXSymbol, sectorLabel, splitSymbol } from "../lib/format";
  import type { CatalogItem, MarketCoin } from "../lib/types";
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
  const POLL_MS = 10_000;

  type BoardKey = "pair" | "price" | "change" | "volume";
  type BoardSort = { key: BoardKey; dir: "asc" | "desc" };
  const FIRST: Record<BoardKey, "asc" | "desc"> = { pair: "asc", price: "desc", change: "desc", volume: "desc" };

  function storedSort(): BoardSort {
    try {
      const s = JSON.parse(localStorage.getItem("deltarisk.board.sort") || "null");
      if (s && s.key in FIRST && (s.dir === "asc" || s.dir === "desc")) return s;
    } catch {
      // fall through
    }
    return { key: "volume", dir: "desc" }; // the busiest pairs first, as on CoinDCX
  }

  let search = $state("");
  let sector = $state("ALL");
  let sort = $state<BoardSort>(storedSort());
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

  const market = $derived(app.market);
  const win = $derived(boardPrefs.window);

  function value(item: CatalogItem, m: MarketCoin | undefined): number | string | null {
    if (sort.key === "pair") return item.base || item.symbol;
    if (!m) return null;
    if (sort.key === "price") return m.price;
    if (sort.key === "volume") return m.volume_24h;
    return m.change[win];
  }

  const rows = $derived.by(() => {
    const coins = market?.coins || {};
    const found = app.catalog.filter((item) => {
      if (sector !== "ALL" && item.sector !== sector) return false;
      if (!query) return true;
      return [item.symbol, item.base, item.name, item.description, item.sector].some((f) => (f || "").toLowerCase().includes(query));
    });
    const sign = sort.dir === "asc" ? 1 : -1;
    const keyed = found.map((item, i) => ({ item, m: coins[item.symbol], i, v: value(item, coins[item.symbol]) }));
    keyed.sort((a, b) => {
      const ma = a.v === null || a.v === undefined, mb = b.v === null || b.v === undefined;
      if (ma || mb) return ma === mb ? a.i - b.i : ma ? 1 : -1; // pairs without data last, either way
      const cmp = typeof a.v === "string" ? a.v.localeCompare(b.v as string) : (a.v as number) - (b.v as number);
      return cmp ? sign * cmp : a.i - b.i;
    });
    return keyed;
  });

  const status = $derived.by(() => {
    if (!market) return { text: "Loading prices…", busy: true };
    if (market.offline) return { text: "Offline · no market data", busy: false };
    if (!market.markets) return { text: "Loading prices…", busy: true };
    if (win !== "24h" && market.backfilled < market.markets)
      return { text: `Filling ${win} changes · ${market.backfilled}/${market.markets}`, busy: true };
    return { text: `Live · ${market.markets} perpetuals`, busy: false };
  });

  function sortBy(key: BoardKey): void {
    sort = sort.key !== key ? { key, dir: FIRST[key] } : { key, dir: sort.dir === "asc" ? "desc" : "asc" };
    remember("deltarisk.board.sort", sort);
  }

  // Focus the search and poll the board while the drawer is open
  $effect(() => {
    if (!app.sidebarOpen) return;
    const focus = setTimeout(() => input?.focus(), 150);
    loadMarket();
    const poll = setInterval(loadMarket, POLL_MS);
    return () => {
      clearTimeout(focus);
      clearInterval(poll);
    };
  });

  function close(): void {
    app.sidebarOpen = false;
  }

  // CoinDCX-style figures: no currency sign, four to six significant digits
  function price(v: number): string {
    if (v >= 1000) return v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return v.toLocaleString("en-US", { minimumSignificantDigits: 4, maximumSignificantDigits: 6 });
  }

  function volume(v: number): string {
    if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return v.toFixed(0);
  }

  function change(v: number | null | undefined): string {
    if (!isNum(v)) return "—";
    return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)}%`;
  }
</script>

{#snippet head(key: BoardKey, label: string, cls = "")}
  {@const on = sort.key === key}
  <button
    class={["board-head caps", on && "is-sorted", cls]}
    onclick={() => sortBy(key)}
    aria-label="Sort by {label}{on ? (sort.dir === 'asc' ? ', ascending' : ', descending') : ''}"
  >
    <span>{label}</span>
    {#if on}<Icon name="chevron" class="size-3 {sort.dir === 'asc' ? 'rotate-180' : ''}" />{/if}
  </button>
{/snippet}

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
    "coindcx-sidebar fixed inset-y-0 right-0 z-9995 flex w-[640px] max-w-screen flex-col border-l border-rule bg-surface shadow-drawer",
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

  <div class="flex gap-1.5 overflow-x-auto px-5 pb-3 [scrollbar-width:none]" role="group" aria-label="Filter by sector">
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

  <!-- Change window and the board's state -->
  <div class="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-border px-5 pb-3">
    <nav class="inline-flex items-center gap-0.5 border border-border bg-surface p-[3px]" aria-label="Change window">
      <span class="caps pr-2.5 pl-2 text-[13px] font-semibold text-muted">Change</span>
      {#each BOARD_WINDOWS as name (name)}
        <button
          class={[
            "h-7 min-w-[40px] px-2 text-[14px] font-semibold tabular-nums transition-colors",
            name === win ? "notch bg-accent font-bold text-on-accent" : "text-fg-2 hover:bg-accent-soft hover:text-accent-text",
          ]}
          aria-pressed={name === win}
          onclick={() => selectBoardWindow(name)}
        >{name}</button>
      {/each}
    </nav>
    <span class={["flex items-center gap-2 text-[12.5px] font-semibold tracking-[0.06em] uppercase", status.busy ? "text-warn" : "text-dim"]}>
      <span class={["size-1.5", status.busy ? "animate-pulse bg-warn" : "bg-pos"]}></span>{status.text}
    </span>
  </div>

  <div class="board-row board-head-row border-b border-border bg-surface-muted text-[12px] font-semibold text-muted">
    {@render head("pair", "Pair")}
    {@render head("price", "LTP", "justify-self-end")}
    {@render head("change", `Chg · ${win}`, "justify-self-end")}
    {@render head("volume", "Vol 24h", "justify-self-end vol-head")}
    <span></span>
  </div>

  <div class="flex-1 overflow-y-auto overscroll-contain">
    {#each rows as { item, m } (item.symbol)}
      {@const added = Boolean(app.coins[item.symbol])}
      {@const c = m?.change[win]}
      {@const quote = splitSymbol(item.symbol).quote || "USDT"}
      <div class="board-row sidebar-coin border-b border-border">
        <div class="min-w-0">
          <div class="flex min-w-0 items-baseline gap-1 whitespace-nowrap">
            <span class="truncate text-[16.5px] font-bold tracking-[0.02em]">{item.base || item.symbol}</span>
            <span class="flex-none text-[12.5px] font-semibold text-faint">· {quote}</span>
          </div>
          <div class="flex min-w-0 items-baseline gap-2 text-[12.5px]" title={item.description}>
            <span class="truncate text-dim">{item.name}</span>
            <span class="flex-none font-semibold tracking-[0.08em] text-muted uppercase">{sectorLabel(item.sector)}</span>
          </div>
        </div>
        <span
          class="text-right text-[15px] font-semibold tabular-nums"
          title={quote === "INR" && m ? "USDT price of the Binance perpetual behind this INR pair" : undefined}
        >{m ? price(m.price) : "—"}{#if quote === "INR" && m}<span class="ml-0.5 text-[11px] text-faint">$</span>{/if}</span>
        <span class={["text-right text-[15px] font-semibold tabular-nums", !isNum(c) ? "text-dim" : c > 0 ? "val-pos" : c < 0 ? "val-neg" : "val-flat"]}>{change(c)}</span>
        <span class="text-right text-[14.5px] font-semibold text-fg-2 tabular-nums max-sm:hidden">{m ? volume(m.volume_24h) : "—"}</span>
        <div class="flex justify-end gap-0.5">
          <a
            class="icon-btn size-7"
            href={item.coindcx_url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Open {item.symbol} on CoinDCX"
            title="Open on CoinDCX"
          ><Icon name="external" /></a>
          {#if added}
            <span class="grid size-7 place-items-center text-pos" title="On your watchlist"><Icon name="check" /></span>
          {:else}
            <button class="add-btn icon-btn size-7" onclick={() => addCoin(item.symbol)} aria-label="Add {item.symbol}" title="Add to watchlist"><Icon name="plus" /></button>
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
  /* Pair, last price, change, 24h volume, actions; volume drops out on phones */
  .board-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 104px 92px 88px 64px;
    align-items: center;
    column-gap: 14px;
    padding: 8px 12px 8px 20px;
  }

  .board-head-row {
    height: 34px;
    padding-top: 0;
    padding-bottom: 0;
    position: relative;
  }

  /* HUD accent: a cyan tick on the head row's left edge, like the open watchlist card's bracket */
  .board-head-row::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--accent);
  }

  @media (width < 720px) {
    /* Not max-sm:hidden: this block's display rule would beat the utility */
    .board-head:global(.vol-head) {
      display: none;
    }

    .board-row {
      grid-template-columns: minmax(0, 1fr) 92px 80px 64px;
      column-gap: 10px;
      padding-left: 16px;
    }
  }

  .board-head {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font: inherit;
    letter-spacing: 0.06em;
    color: inherit;
    background: none;
    border: 0;
    padding: 0;
  }

  .board-head:hover,
  .board-head.is-sorted {
    color: var(--accent-text);
  }

  .sidebar-coin:hover {
    background: linear-gradient(90deg, var(--accent-soft), transparent 60%), var(--surface-hover);
    box-shadow: inset 2px 0 0 var(--accent);
  }

  .add-btn {
    color: var(--text-2);
    border-color: var(--border-strong);
  }
</style>
