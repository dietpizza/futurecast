<script lang="ts">
  import { app } from "../lib/live.svelte";
  import { HORIZONS, horizonOf, isBusy } from "../lib/momentum";

  const summary = $derived.by(() => {
    const coins = Object.values(app.coins);
    const counts = { LONG: 0, SHORT: 0 };
    let downloading = 0;
    for (const c of coins) {
      const signal = horizonOf(c, app.timeframe)?.signal;
      if (signal === "LONG" || signal === "SHORT") counts[signal]++;
      if (isBusy(c)) downloading++;
    }
    return { pairs: coins.length, ...counts, downloading };
  });
</script>

<div class="mb-5 flex flex-wrap items-end justify-between gap-4 max-sm:flex-col max-sm:items-stretch">
  <div>
    <p class="bullet mb-2.5 flex items-center gap-[9px] text-[13px] font-semibold tracking-widest text-muted uppercase before:bg-accent">
      CoinDCX futures
    </p>
    <h1 class="text-[40px] leading-none font-semibold tracking-[0.04em] uppercase">Watchlist</h1>
    <p class="mt-2 text-sm font-medium tracking-[0.02em] text-muted">
      {#if summary.pairs === 0}
        No pairs tracked
      {:else}
        {summary.pairs} {summary.pairs === 1 ? "pair" : "pairs"} ·
        <span class={summary.LONG ? "font-semibold text-pos" : ""}>{summary.LONG} long</span> ·
        <span class={summary.SHORT ? "font-semibold text-neg" : ""}>{summary.SHORT} short</span> at {app.timeframe}{#if summary.downloading}
          {" "}· {summary.downloading} downloading{/if}
      {/if}
    </p>
  </div>

  <nav class="inline-flex items-center gap-0.5 border border-border bg-surface p-[3px]" aria-label="Momentum horizon">
    <span
      class="caps pr-2.5 pl-2 text-[13px] font-semibold text-muted"
      data-tooltip-title="Momentum horizon"
      data-tooltip="The window the watchlist's columns and signal are measured over: the last move of this length, and what followed similar moves. 1h and 4h use 15-minute candles (180 days of history), 1d and 3d hourly candles (2 years)."
    >Horizon</span>
    {#each HORIZONS as tf (tf)}
      <button
        class={[
          "h-7 min-w-[42px] px-2.5 text-[15px] font-semibold tabular-nums transition-colors max-sm:flex-1",
          tf === app.timeframe ? "notch bg-accent font-bold text-on-accent" : "text-fg-2 hover:bg-accent-soft hover:text-accent-text",
        ]}
        data-tf={tf}
        aria-pressed={tf === app.timeframe}
        onclick={() => (app.timeframe = tf)}
      >{tf}</button>
    {/each}
  </nav>
</div>
