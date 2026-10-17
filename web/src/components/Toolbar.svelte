<script lang="ts">
  import { app, selectStop, selectWindow } from "../lib/live.svelte";
  import { isBusy, reading, riskTone, SIDES, STOPS, WINDOWS, windowOf } from "../lib/risk";

  // How many pairs have a side whose stop is hit more often than not within the window
  const summary = $derived.by(() => {
    const coins = Object.values(app.coins);
    let high = 0;
    let busy = 0;
    for (const c of coins) {
      const w = windowOf(c, app.window);
      if (w && SIDES.some((s) => { const r = reading(c, w, s, app.stopPct); return r.checked && riskTone(r.score) === "high"; })) high++;
      if (isBusy(c)) busy++;
    }
    return { pairs: coins.length, high, busy };
  });

  const SEG = "h-7 min-w-[42px] px-2.5 text-[15px] font-semibold tabular-nums transition-colors max-sm:flex-1 max-sm:min-w-0 max-sm:px-1";
  const ON = "notch bg-accent font-bold text-on-accent";
  const OFF = "text-fg-2 hover:bg-accent-soft hover:text-accent-text";
  const LABEL = "caps pr-2.5 pl-2 text-[13px] font-semibold text-muted max-sm:pl-1 max-sm:pr-1.5";
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
        <span class={summary.high ? "font-semibold text-risk-high" : ""}>{summary.high} likely to hit the stop</span> within {app.window}{#if summary.busy}
          {" "}· {summary.busy} loading{/if}
      {/if}
    </p>
  </div>

  <div class="flex flex-col items-end gap-2 max-sm:items-stretch">
    <nav class="inline-flex items-center gap-0.5 border border-border bg-surface p-[3px]" aria-label="Window">
      <span
        class={LABEL}
        data-tooltip-title="Window"
        data-tooltip="How long you plan to hold. Every risk number is for a position opened now and held this long: 5m to 1h from 5-minute candles (90 days), 2h to 4h from 15-minute candles (180 days), 8h from hourly candles (2 years)."
      >Window</span>
      {#each WINDOWS as name (name)}
        <button class={[SEG, name === app.window ? ON : OFF]} aria-pressed={name === app.window} onclick={() => selectWindow(name)}>{name}</button>
      {/each}
    </nav>
    <nav class="inline-flex items-center gap-0.5 border border-border bg-surface p-[3px]" aria-label="Stop loss">
      <span
        class={LABEL}
        data-tooltip-title="Stop loss"
        data-tooltip="Your stop as a loss of margin (the PnL % CoinDCX shows). With leverage it is a smaller price move: −5% at 20× is 0.25% of price. The risk score is the chance this stop is hit within the window."
      >Stop</span>
      {#each STOPS as stop (stop)}
        <button class={[SEG, stop === app.stopPct ? ON : OFF]} aria-pressed={stop === app.stopPct} onclick={() => selectStop(stop)}>−{stop}%</button>
      {/each}
    </nav>
  </div>
</div>
