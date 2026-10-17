<script lang="ts">
  import { app } from "../lib/live.svelte";
  import CoinRow from "./CoinRow.svelte";

  const h = $derived(app.timeframe);
</script>

<!-- The list's box is drawn by the head and each row rather than by the section, so an open
     row can step out of it as its own card, with the rows around it closing off their groups. -->
<section class="watchlist" aria-label="Tracked pairs">
  <div
    class="row-grid caps sticky top-(--header-h) z-10 h-[38px] border border-border bg-surface-muted text-[12.5px] font-semibold whitespace-nowrap text-muted max-sm:hidden"
    aria-hidden="true"
  >
    <span>Pair</span>
    <span class="text-right">Price</span>
    <span><span class="th-help" data-tooltip-title="Momentum signal" data-tooltip="Long or Short only when, after past moves like the latest one (same fifth of all moves of this length), the next window averaged more than a market round trip's fees in one direction, the 95% range of that average excludes zero, and it held in both halves of the history. Statistics, not a model: no edge is the usual answer.">Signal</span></span>
    <span class="text-right"><span class="th-help" data-tooltip-title="Last move" data-tooltip="Return over the last window, z (the move in units of a typical move of that length, from the last 14 days' volatility) and its percentile among every past move of that length.">Move · {h}</span></span>
    <span class="text-right"><span class="th-help" data-tooltip-title="After similar moves" data-tooltip="Mean return over the next window after past moves in the same fifth as the latest one, and how often it rose. Colored only when the 95% range of the mean excludes zero.">Next {h}</span></span>
    <span class="max-md:hidden"><span class="th-help" data-tooltip-title="Character" data-tooltip="Variance-ratio test at this horizon: Trending when moves keep their direction more than a random walk would (ratio above 1, p below 0.05), Reverting when they turn back (below 1), else Random.">Character</span></span>
    <span><span class="th-help" data-tooltip-title="Taker flow" data-tooltip="Share of the traded value bought with market orders over the last window, and its percentile among past windows. Aggressive buyers above 50%.">Taker flow · {h}</span></span>
    <span class="max-lg:hidden"><span class="th-help" data-tooltip-title="Order book" data-tooltip="Share of resting limit-order liquidity on the bid (buy) side across the top 20 levels.">Order book</span></span>
    <span></span>
  </div>

  {#each Object.entries(app.coins) as [symbol, coin] (symbol)}
    <CoinRow {symbol} {coin} />
  {:else}
    <div class="px-6 py-14 text-center">
      <h3 class="text-2xl font-semibold tracking-[0.04em] uppercase">Your watchlist is empty</h3>
      <p class="mx-auto mt-1.5 mb-[18px] max-w-[360px] text-[14.5px] text-dim">
        Add a CoinDCX futures pair to start watching its momentum statistics.
      </p>
      <button class="btn btn-primary mx-auto" onclick={() => (app.sidebarOpen = true)}>
        Add pair <kbd class="hotkey-hint">/</kbd>
      </button>
    </div>
  {/each}
</section>
