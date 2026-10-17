<script lang="ts">
  import { app, removeCoin, toggleExpanded } from "../lib/live.svelte";
  import { coindcxUrl, formatNumber, formatPrice, formatUsdCompact, isNum, pct, pctile, pText, sectorLabel, signed, sp, splitSymbol, toneClass } from "../lib/format";
  import { afterText, CHARACTER, characterText, hasMove, horizonOf, isBusy, loadingStage, moveText, signalVerdict, significant, waitingNote } from "../lib/momentum";
  import type { Coin } from "../lib/types";
  import CoinDetails from "./CoinDetails.svelte";
  import Icon from "./Icon.svelte";

  let { symbol, coin }: { symbol: string; coin: Coin } = $props();

  // The small line under a cell's value, in the label red unless it leans one way
  const SUB = "mt-0.5 text-[13px] whitespace-nowrap";
  const VAL = "text-[15px] font-semibold tracking-[-0.01em] whitespace-nowrap";

  const tf = $derived(app.timeframe);
  const open = $derived(Boolean(app.expanded[symbol]));
  const pair = $derived(splitSymbol(symbol));
  const h = $derived(horizonOf(coin, tf));
  const after = $derived(h?.after);
  const persistence = $derived(h?.persistence);
  const flow = $derived(h?.flow);
  const book = $derived(coin.orderbook_depth || {});
  const verdict = $derived(signalVerdict(coin, tf));
  const waiting = $derived(waitingNote(coin));
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
  {#if note}<div class="{SUB} text-muted">{note}</div>{/if}
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
        {sectorLabel(coin.sector)}{#if isBusy(coin)}{coin.sector ? " · " : ""}<span class="text-warn">{loadingStage(coin)}</span>{/if}
      </div>
    </div>

    <div class="c-price text-right tabular-nums">
      {#key flashes}
        <span class="-mr-1 px-1 py-0.5 text-base font-[550] {flashes ? flash : ''}">{formatPrice(coin.current_price)}</span>
      {/key}
    </div>

    <div class="c-signal">
      {#if !coin.has_data}
        {#if isBusy(coin)}
          <span class="pill pill-warn">{loadingStage(coin)}</span>
        {:else if coin.status === "failed"}
          <span class="pill pill-neutral" data-tooltip-title="Download failed" data-tooltip={coin.error || ""}>Failed</span>
        {:else}
          <span class="pill pill-neutral">Loading</span>
        {/if}
      {:else}
        <span class="pill {verdict.pill}" data-tooltip-title="{tf} momentum · {verdict.label}" data-tooltip={verdict.why}>{verdict.label}</span>
      {/if}
    </div>

    <div class="c-move text-right tabular-nums" data-tooltip-title="Move · last {tf}" data-tooltip={hasMove(h) ? moveText(coin, h, tf) : null}>
      {#if hasMove(h)}
        <span class="{VAL} {Math.abs(h.z) >= 1 ? toneClass(h.return_pct) : 'val-flat'}">{sp(h.return_pct)}</span>
        <div class="{SUB} text-muted">z {signed(h.z, 1)} · {pctile(h.percentile)} pct</div>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div class="c-next text-right tabular-nums" data-tooltip-title="Next {tf} after similar moves" data-tooltip={after ? afterText(after, tf) : null}>
      {#if after}
        <span class="{VAL} {significant(after) ? toneClass(after.mean_pct, 2) : 'val-flat'}">{sp(after.mean_pct)}</span>
        <div class="{SUB} text-muted">{pct(after.hit_rate)} up · {significant(after) ? "significant" : "noise"}</div>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div class="c-char max-md:hidden" data-tooltip-title="Character · {tf}" data-tooltip={h && persistence?.character ? characterText(h, tf) : null}>
      {#if persistence?.character}
        <span class="text-[15px] font-semibold whitespace-nowrap {persistence.character === 'random' ? 'val-flat' : ''}">{CHARACTER[persistence.character].label}</span>
        <div class="{SUB} text-muted">VR {formatNumber(persistence.vr, 2)} · {pText(persistence.vr_p)}</div>
      {:else}
        {@render empty(waiting)}
      {/if}
    </div>

    <div
      class="c-flow"
      data-tooltip-title="Taker flow · last {tf}"
      data-tooltip={flow ? `Market buys took ${formatNumber(flow.buy_pct, 1)}% of the traded value over the last ${tf} (the rest were market sells): more buying than ${formatNumber(flow.percentile, 0)}% of past ${tf} windows.` : null}
    >
      {#if flow}
        <div class="book-bar"><span style:width="{Math.max(5, Math.min(95, flow.buy_pct))}%"></span><span></span></div>
        <div class="{SUB} {flow.percentile >= 80 ? 'val-pos' : flow.percentile <= 20 ? 'val-neg' : 'text-muted'}">{formatNumber(flow.buy_pct, 0)}% buys · {pctile(flow.percentile)} pct</div>
      {:else}
        <div class="book-bar is-empty"></div>
        <div class="{SUB} text-muted">{waiting || "No volume data"}</div>
      {/if}
    </div>

    <div
      class="c-book max-lg:hidden"
      data-tooltip-title="Order book · top 20 levels"
      data-tooltip={isNum(book.bid_pct) && book.available !== false
        ? `Bids ${formatUsdCompact(book.bid_depth_usd)} · ${book.bid_pct}%\nAsks ${formatUsdCompact(book.ask_depth_usd)} · ${book.ask_pct}%\nSpread ${formatNumber(book.spread_bps, 1)} bps`
        : null}
    >
      {#if isNum(book.bid_pct) && book.available !== false}
        {@const imbalance = book.book_imbalance_pct || 0}
        <div class="book-bar"><span style:width="{book.bid_pct}%"></span><span></span></div>
        <div class="{SUB} {imbalance >= 12 ? 'val-pos' : imbalance <= -12 ? 'val-neg' : 'text-muted'}">{book.bid_pct}% bids</div>
      {:else}
        <div class="book-bar is-empty"></div>
        <div class="{SUB} text-muted">No book data</div>
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

  /* Phones: pair and price on top, signal and the two momentum numbers below */
  @media (width < 720px) {
    .row-main {
      grid-template-columns: 1fr auto auto auto;
      grid-template-areas:
        "pair   pair  price  actions"
        "signal move  next   actions";
      column-gap: 14px;
      row-gap: 8px;
      padding: 12px 8px 12px 16px;
    }
    .c-pair { grid-area: pair; }
    .c-price { grid-area: price; }
    .c-signal { grid-area: signal; display: flex; align-items: center; }
    .c-move { grid-area: move; }
    .c-next { grid-area: next; }
    .c-actions { grid-area: actions; flex-direction: column; justify-content: center; }
    .c-move::before { content: "Move "; }
    .c-next::before { content: "Next "; }
    .c-move::before,
    .c-next::before { font-size: 12px; color: var(--text-muted); }
    .c-char,
    .c-flow,
    .c-book,
    .btn-remove { display: none; }

    .row-details { padding: 4px 12px 16px; }
  }
</style>
