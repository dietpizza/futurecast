<script lang="ts">
  import { app, sortBy } from "../lib/live.svelte";
  import { sortCoins, type SortKey } from "../lib/sort";
  import CoinRow from "./CoinRow.svelte";
  import Icon from "./Icon.svelte";

  const w = $derived(app.window);
  const coins = $derived(sortCoins(Object.entries(app.coins), app.sort, app.window, app.stopPct));
</script>

<!-- A sortable column head: click cycles its first direction, the other, then the added order.
     The sorted column shows a chevron (down: highest or last first; up: lowest or first first). -->
{#snippet sortable(key: SortKey, label: string, title: string, tip: string, tone: string)}
  {@const on = app.sort?.key === key}
  <button
    class={["sort-head caps", on && "is-sorted"]}
    onclick={() => sortBy(key)}
    aria-label="Sort by {title || label}{on ? (app.sort?.dir === 'asc' ? ', ascending' : ', descending') : ''}"
  >
    {#if tip}
      <span class={["th-help", tone]} data-tooltip-title={title} data-tooltip={tip}>{label}</span>
    {:else}
      <span class={tone}>{label}</span>
    {/if}
    {#if on}<Icon name="chevron" class="size-3 {app.sort?.dir === 'asc' ? 'rotate-180' : ''}" />{/if}
  </button>
{/snippet}

<!-- The list's box is drawn by the head and each row rather than by the section, so an open
     row can step out of it as its own card, with the rows around it closing off their groups. -->
<section class="watchlist" aria-label="Tracked pairs">
  <div
    class="row-grid caps sticky top-(--header-h) z-10 h-[38px] border border-border bg-surface-muted text-[12.5px] font-semibold whitespace-nowrap text-muted max-sm:hidden"
  >
    <span>{@render sortable("pair", "Pair", "", "", "")}</span>
    <span class="text-right">Price</span>
    <span>{@render sortable("call", `Call · ${w}`, "Call", "Long when momentum is bullish in a trending market and a long would hit your stop less than half the time within the window; Short the mirror; otherwise Wait. A rule, not a forecast: hover a call for how the next window went after past trending readings on that coin.", "")}</span>
    <span class="max-md:hidden">{@render sortable("tier", `Tier`, "Tier and leverage", "Megacap (BTC, ETH, SOL, BNB, XRP) 20×, shitcoin (meme coins, or under $50M traded on Binance in 24 hours) 3×, altcoin 10×. Change it in the open row.", "")}</span>
    <span class="group-start text-right">{@render sortable("long", `Long risk`, "Long risk", "Chance in % that a long opened now hits your stop within the window, at the coin's leverage. Below it: the 1-in-10 bad case as a price move.", "th-fwd")}</span>
    <span class="text-right">{@render sortable("short", `Short risk`, "Short risk", "Chance in % that a short opened now hits your stop within the window, at the coin's leverage. Below it: the 1-in-10 bad case as a price move.", "th-fwd")}</span>
    <span class="group-start max-md:hidden">{@render sortable("momentum", `Momentum · ${w}`, "Momentum", "What price did over the last few windows, −100 to +100. Bullish above +25, bearish below −25. A description, not a forecast.", "th-now")}</span>
    <span class="max-md:hidden"><span class="th-help th-now" data-tooltip-title="Market state" data-tooltip="Wild (volatility in the top 10% of 30 days), trending (a fairly straight move with momentum), quiet (volatility in the bottom quarter) or choppy.">State</span></span>
    <span class="max-md:hidden">{@render sortable("volatility", `Volatility`, "Volatility", "How wild the last few windows were against the last 30 days: calm (bottom quarter), normal, high (top quarter), wild (top 10%).", "th-now")}</span>
    <span class="max-lg:hidden">{@render sortable("funding", `Funding`, "Funding", "The latest settled funding rate and its percentile over 90 days. Top 10%: longs crowded; bottom 10%: shorts crowded.", "th-now")}</span>
    <span></span>
  </div>

  {#each coins as [symbol, coin] (symbol)}
    <CoinRow {symbol} {coin} />
  {:else}
    <div class="px-6 py-14 text-center">
      <h3 class="text-2xl font-semibold tracking-[0.04em] uppercase">Your watchlist is empty</h3>
      <p class="mx-auto mt-1.5 mb-[18px] max-w-[360px] text-[14.5px] text-dim">
        Add a CoinDCX futures pair to see how likely a long or a short is to hit your stop.
      </p>
      <button class="btn btn-primary mx-auto" onclick={() => (app.sidebarOpen = true)}>
        Add pair <kbd class="hotkey-hint">/</kbd>
      </button>
    </div>
  {/each}
</section>

<style>
  .sort-head {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font: inherit;
    letter-spacing: inherit;
    color: inherit;
    background: none;
    border: 0;
    padding: 0;
  }

  .sort-head:hover > :first-child {
    filter: brightness(1.25);
  }

  .sort-head.is-sorted {
    font-weight: 700;
  }
</style>
