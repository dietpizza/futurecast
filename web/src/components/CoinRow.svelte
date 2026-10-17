<script lang="ts">
  import { app, removeCoin, toggleExpanded } from "../lib/live.svelte";
  import { coindcxUrl, formatNumber, formatPrice, isNum, pctile, sectorLabel, sp, splitSymbol } from "../lib/format";
  import { callOf, callRecordText, isBusy, loadingStage, MOMENTUM_LABELS, momentumText, STATE_LABELS, stateText, tierLabel, VOL_LABELS, volatilityText, waitingNote, windowOf } from "../lib/risk";
  import type { Coin } from "../lib/types";
  import CoinDetails from "./CoinDetails.svelte";
  import Icon from "./Icon.svelte";
  import RiskCell from "./RiskCell.svelte";

  let { symbol, coin }: { symbol: string; coin: Coin } = $props();

  // The small line under a cell's value
  const SUB = "mt-0.5 text-[13px] whitespace-nowrap";
  const VAL = "text-[15px] font-semibold whitespace-nowrap";

  const name = $derived(app.window);
  const open = $derived(Boolean(app.expanded[symbol]));
  const pair = $derived(splitSymbol(symbol));
  const w = $derived(windowOf(coin, name));
  const waiting = $derived(waitingNote(coin));
  const call = $derived(w ? callOf(coin, w, name, app.stopPct) : null);
  const funding = $derived(coin.funding);
  const overridden = $derived(Boolean(coin.overrides?.tier || coin.overrides?.leverage));
  const detailsId = $derived(`details-${symbol.replace(/[^a-zA-Z0-9]/g, "_")}`);

  // Flash the price green or red when a stream update moves it; the key block restarts the animation
  let flash = $state("");
  let flashes = $state(0);
  let lastPrice: number | null = null;
  $effect(() => {
    const price = Number(coin.current_price);
    if (!isNum(price) || !price) return;
    if (lastPrice !== null && price !== lastPrice) {
      flash = price > lastPrice ? "flash-green" : "flash-red";
      flashes++;
    }
    lastPrice = price;
  });
</script>

{#snippet empty(note: string)}
  <span class="text-[15px] val-muted">—</span>
  {#if note}<div class="{SUB} text-dim">{note}</div>{/if}
{/snippet}

<article class={["coin-row", open && "is-open"]} aria-label={pair.base}>
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions (the chevron button is the keyboard path) -->
  <div class="row-grid row-main" onclick={() => toggleExpanded(symbol)}>
    <div class="c-pair">
      <div class="flex min-w-0 items-baseline gap-1">
        <a
          class="text-[19px] font-bold tracking-[0.04em] no-underline hover:text-accent-text hover:underline hover:underline-offset-3"
          href={coindcxUrl(symbol)}
          target="_blank"
          rel="noopener noreferrer"
          onclick={(e) => e.stopPropagation()}
          title="Open {symbol} on CoinDCX"
        >{pair.base}</a>
        {#if pair.quote}<span class="text-[13px] font-semibold text-faint">{pair.quote}</span>{/if}
      </div>
      <div class="{SUB} text-muted">
        {#if isBusy(coin)}
          <span class="text-warn">{loadingStage(coin)}</span>
        {:else if coin.status === "failed"}
          <span class="text-neg" data-tooltip-title="Download failed" data-tooltip={coin.error || ""}>Failed</span>
        {:else}
          {sectorLabel(coin.sector)}{#if coin.leverage}<span class="hidden max-md:inline">{` · ${coin.leverage}×`}</span>{/if}
        {/if}
      </div>
    </div>

    <div class="c-price text-right tabular-nums">
      {#key flashes}
        <span class="-mr-1 px-1 py-0.5 text-base font-[550] {flashes ? flash : ''}">{formatPrice(coin.current_price)}</span>
      {/key}
    </div>

    <div class="c-call" data-tooltip-title={call ? `${call.label} · ${name}` : null} data-tooltip={call && w ? `${call.why}\n\n${callRecordText(w, call, name)}`.trim() : null}>
      {#if call}
        <span class="call call-{call.call}">{call.label}</span>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div
      class="c-tier max-md:hidden"
      data-tooltip-title="Tier and leverage"
      data-tooltip={overridden
        ? `Set by you (automatic: ${tierLabel(coin.tier_auto)}). Change it in the open row.`
        : "Automatic from the coin and its 24h volume. Change it in the open row."}
    >
      <span class={VAL}>{coin.leverage ? `${coin.leverage}×` : "—"}{#if overridden}<span class="text-accent-text">*</span>{/if}</span>
      <div class="{SUB} text-muted">{tierLabel(coin.tier)}</div>
    </div>

    <div class="c-long group-start text-right">
      {#if w}<RiskCell {coin} {w} side="long" {name} />{:else}{@render empty(waiting)}{/if}
    </div>

    <div class="c-short text-right">
      {#if w}<RiskCell {coin} {w} side="short" {name} />{:else}{@render empty(waiting)}{/if}
    </div>

    <div class="c-mom group-start max-md:hidden" data-tooltip-title="Momentum · {name}" data-tooltip={w ? momentumText(name) : null}>
      {#if w && isNum(w.momentum) && w.momentum_label}
        {@const m = w.momentum}
        <div class="flex items-baseline gap-2">
          <span class="{VAL} {w.momentum_label === 'bullish' ? 'val-pos' : w.momentum_label === 'bearish' ? 'val-neg' : 'val-flat'}">{MOMENTUM_LABELS[w.momentum_label]}</span>
          <span class="text-[13px] text-dim tabular-nums">{m > 0 ? "+" : m < 0 ? "−" : ""}{Math.abs(Math.round(m))}</span>
        </div>
        <div class="mom-bar mt-1.5">
          <span
            class={m >= 0 ? "bg-pos-fill" : "bg-neg-fill"}
            style:left={m >= 0 ? "50%" : `${50 + m / 2}%`}
            style:width="{Math.abs(m) / 2}%"
          ></span>
        </div>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div class="c-state max-md:hidden" data-tooltip-title="Market state · {name}" data-tooltip={w ? stateText(w, name) : null}>
      {#if w?.state}
        <span class="{VAL} {w.state === 'wild' ? 'val-warn' : 'val-flat'}">{STATE_LABELS[w.state]}</span>
        <div class="{SUB} text-dim">eff {isNum(w.efficiency) ? Math.round(w.efficiency * 100) : "—"}%</div>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div class="c-vol max-md:hidden" data-tooltip-title="Volatility · {name}" data-tooltip={w ? volatilityText(w, name) : null}>
      {#if w?.volatility}
        <span class="{VAL} {w.volatility === 'wild' ? 'val-warn' : 'val-flat'}">{VOL_LABELS[w.volatility]}</span>
        <div class="{SUB} text-dim">{pctile(w.vol_pct)} pct</div>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div
      class="c-fund max-lg:hidden"
      data-tooltip-title="Funding"
      data-tooltip={funding
        ? `The latest settled funding rate (longs pay shorts when positive), ${pctile(funding.percentile)} percentile of the last ${funding.days} days. Mean ${sp(funding.mean_pct, 4)}.`
        : null}
    >
      {#if funding}
        {@const p = funding.percentile}
        <span class="{VAL} val-flat tabular-nums">{sp(funding.rate_pct, 4)}</span>
        <div class="{SUB} {isNum(p) && (p >= 90 || p <= 10) ? 'val-warn' : 'text-dim'}">
          {isNum(p) && p >= 90 ? "longs crowded" : isNum(p) && p <= 10 ? "shorts crowded" : `${pctile(p)} pct`}
        </div>
      {:else}
        {@render empty(coin.has_data ? "no data" : waiting)}
      {/if}
    </div>

    <div class="c-actions flex justify-end gap-0.5">
      <button
        class="btn-remove icon-btn hover:bg-neg-soft! hover:text-neg!"
        onclick={(e) => { e.stopPropagation(); removeCoin(symbol); }}
        aria-label="Remove {pair.base}"
        title="Remove"
      ><Icon name="close" /></button>
      <button
        class="icon-btn"
        onclick={(e) => { e.stopPropagation(); toggleExpanded(symbol); }}
        aria-expanded={open}
        aria-controls={detailsId}
        aria-label="{open ? 'Hide' : 'Show'} details for {pair.base}"
      ><Icon name="chevron" class="transition-transform duration-200 {open ? 'rotate-180' : ''}" /></button>
    </div>
  </div>

  {#if open}
    <div class="row-details" id={detailsId}>
      <CoinDetails {coin} {symbol} />
    </div>
  {/if}
</article>

<style>
  .coin-row {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 0;
    overflow: clip;
    transition: margin 0.15s ease, box-shadow 0.15s ease;
  }

  .coin-row.is-open {
    margin: 14px 0;
    background: var(--open-bg);
    border: 1px solid var(--rule);
    box-shadow: var(--shadow-md);
  }

  .coin-row.is-open:last-child {
    margin-bottom: 0;
  }

  /* The row after an open card starts a new group, so it needs its own top edge */
  .coin-row.is-open + .coin-row:not(.is-open) {
    border-top: 1px solid var(--border);
  }

  /* HUD corner brackets on the open card */
  .coin-row.is-open::before,
  .coin-row.is-open::after {
    content: "";
    position: absolute;
    z-index: 1;
    width: 14px;
    height: 14px;
    pointer-events: none;
    border: 0 solid var(--accent);
  }

  .coin-row.is-open::before {
    top: 0;
    left: 0;
    border-top-width: 2px;
    border-left-width: 2px;
  }

  .coin-row.is-open::after {
    right: 0;
    bottom: 0;
    border-right-width: 2px;
    border-bottom-width: 2px;
  }

  .row-main {
    min-height: 64px;
    padding-top: 10px;
    padding-bottom: 10px;
    cursor: pointer;
    transition: background 0.12s;
  }

  .row-main:hover {
    background: linear-gradient(90deg, var(--accent-soft), transparent 55%), var(--surface-hover);
    box-shadow: inset 2px 0 0 var(--accent);
  }

  .is-open .row-main:hover {
    background: linear-gradient(90deg, var(--accent-soft), transparent 55%), var(--open-hover);
  }

  .btn-remove {
    opacity: 0;
  }

  .row-main:hover .btn-remove,
  .row-main:focus-within .btn-remove {
    opacity: 1;
  }

  @media (hover: none) {
    .btn-remove { opacity: 1; }
  }

  .row-details {
    padding: 4px 20px 20px;
    background: var(--open-well);
    border-top: 1px solid var(--rule);
  }

  /* Phones: pair, call and price on top, the two risks below */
  @media (width < 720px) {
    .row-main {
      grid-template-columns: 1fr auto auto auto;
      grid-template-areas:
        "pair call  price actions"
        "long long  short actions";
      column-gap: 12px;
      row-gap: 10px;
      padding: 12px 8px 12px 16px;
    }
    .c-pair { grid-area: pair; }
    .c-price { grid-area: price; }
    .c-call { grid-area: call; }
    .c-long { grid-area: long; box-shadow: none; padding-left: 0; }
    .c-short { grid-area: short; }
    .c-actions { grid-area: actions; flex-direction: column; justify-content: center; }
    .c-long::before { content: "Long risk"; }
    .c-short::before { content: "Short risk"; }
    .c-long::before,
    .c-short::before { display: block; font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .btn-remove { display: none; }

    .row-details { padding: 4px 12px 16px; }
  }
</style>
