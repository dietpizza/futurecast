<!-- Plain-English glossary: a bar docked to the bottom of the window that opens upward -->
<footer class="plain-english-glossary sticky bottom-0 z-40 border-t border-border bg-header backdrop-blur-sm backdrop-saturate-180 has-open:bg-surface has-open:shadow-[0_-8px_24px_rgba(0,0,0,0.12)]">
  <details class="mx-auto max-w-(--page-w) px-6 max-sm:px-4">
    <summary class="relative flex cursor-pointer list-none flex-wrap items-baseline gap-x-3 gap-y-1 py-3 pl-[22px] select-none [&::-webkit-details-marker]:hidden">
      <span class="text-[15px] font-bold tracking-[0.08em] uppercase transition-colors">The "Not a Finance-Bro" Cheat Sheet</span>
      <span class="text-sm text-muted max-sm:hidden">Plain-English explanations of every number on this page</span>
    </summary>

    <div class="grid max-h-[min(60dvh,560px)] grid-cols-3 gap-7 overflow-y-auto overscroll-contain border-t border-border pb-5 max-md:grid-cols-1 max-md:gap-1">
      <div class="min-w-0">
        <h3 class="bullet mt-4 mb-3 flex items-center gap-[9px] text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">If you open now</h3>
        <dl class="flex flex-col gap-3.5">
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Risk</span> <span class="u-fwd">Long and short risk</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Open a long (or a short) now and hold it for the window: the chance in % that price touches your stop before the window ends. <strong class="font-semibold text-fg-2">Green</strong> under 25, <strong class="font-semibold text-fg-2">yellow</strong> 25–49, <strong class="font-semibold text-fg-2">red</strong> 50 and up, when the stop is hit more often than not. Hover it for the main reason behind the number.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Same?</span> Why long and short look alike</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">They are two separate questions about the same window: will price dip to a long's stop, and will it rise to a short's? Price wobbles both ways within a window, so both can happen, and the size of the wobble is what can be forecast; its direction can't. The two split apart only when moves have been lopsided, for example when funding is crowded on one side.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Bad</span> <span class="u-fwd">Bad case</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">A 1-in-10 bad stretch: how far price goes against the position within the window, wicks included, in 1 of every 10 similar windows. Shown as a price move and as a share of your margin.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Swing</span> <span class="u-fwd">Swing</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">The typical move expected over the next window, from the volatility forecast (in the open row's table). Everything in this column scales with it: a bigger swing means a higher chance of reaching your stop.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Liq</span> <span class="u-fwd">Liquidation</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">The chance the move reaches the liquidation price (about 1 / leverage, less a small maintenance margin). With a stop in place it matters only when a fast move skips past the stop.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Stop</span> Stop on margin</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Your stop as a loss of margin, the PnL % CoinDCX shows, set at the top of the page. Leverage turns it into a much smaller price move: <strong class="font-semibold text-fg-2">−5% at 20× is a 0.25% price move</strong>, at 3× it is 1.67%. That is why the same coin scores higher on a megacap tier.</dd>
          </div>
        </dl>
      </div>

      <div class="min-w-0">
        <h3 class="bullet mt-4 mb-3 flex items-center gap-[9px] text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">What just happened</h3>
        <dl class="flex flex-col gap-3.5">
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Mom</span> <span class="u-now">Momentum</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Which way price went over the last 3 windows (at 1h: the last 3 hours), from −100 to +100: the return and the slope of its average, measured in typical moves. Bullish above +25, bearish below −25. It describes the past; it does not predict the next move.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">State</span> <span class="u-now">Market state</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">How the last 3 windows behaved. <strong class="font-semibold text-fg-2">Wild</strong>: volatility in the top 10% of the last 30 days. <strong class="font-semibold text-fg-2">Trending</strong>: price moved in a fairly straight line with momentum (efficiency is the net move over the distance travelled). <strong class="font-semibold text-fg-2">Quiet</strong>: volatility in the bottom quarter. <strong class="font-semibold text-fg-2">Choppy</strong>: everything else, back and forth.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Vol</span> <span class="u-now">Volatility</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">How big the swings of the last 3 windows were compared with the last 30 days: calm (bottom quarter), normal, high (top quarter) or wild (top 10%), with the percentile.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Fund</span> <span class="u-now">Funding</span></dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">What longs pay shorts (or the reverse when negative) at the latest settlement, against the last 90 days. In the top 10%, longs are crowded and a sharp drop can flush them out; in the bottom 10%, shorts are crowded.</dd>
          </div>
        </dl>
      </div>

      <div class="min-w-0">
        <h3 class="bullet mt-4 mb-3 flex items-center gap-[9px] text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">Reading the page</h3>
        <dl class="flex flex-col gap-3.5">
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Color</span> The two head colors</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim"><span class="u-fwd font-semibold">Violet</span>: about the next window, if you open now. <span class="u-now font-semibold">Orange</span>: about the last few windows up to now. Each column head is written in its color, and the two groups of columns are split by a thin line.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Call</span> Long / Short / Wait</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">A rule, not a prediction. <strong class="font-semibold text-fg-2">Long</strong> when momentum is bullish in a trending market and a long would hit your stop less than half the time within the window; <strong class="font-semibold text-fg-2">Short</strong> the mirror; otherwise <strong class="font-semibold text-fg-2">Wait</strong>. It leans on what just happened for the direction, and hovering it shows how that has gone on the coin: so far the next window went the trend's way 41–51% of the time, no better than a coin flip once fees are paid.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Win</span> Window</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">How long you plan to hold, 5 minutes to 8 hours, picked at the top. Risk numbers cover the next window; momentum, state and volatility the last 3. 5m to 1h use 5-minute candles (90 days of history), 2–4h 15-minute candles (180 days), 8h hourly candles (2 years).</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Tier</span> Tier and leverage</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Megacap (BTC, ETH, SOL, BNB, XRP) 20×, shitcoin (meme coins, or under $50M traded in 24h) 3×, altcoin 10×. Open a row to set another tier or leverage; a * marks your own setting.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Model</span> Where the numbers come from</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Swings come in clusters: a wild hour is usually followed by another. A volatility forecast (HAR) scales thousands of past windows of the coin to today's conditions. A LightGBM model is used instead only where it beat that forecast by 2% on data it had not seen. Models retrain once a day.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Check</span> The honesty check</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Each model is tested on the last 40% of the history, which it was not trained on. The 1-in-10 bad case should be exceeded about 10% of the time; outside <strong class="font-semibold text-fg-2">7–13%</strong> the number is hidden and shows —.</dd>
          </div>
        </dl>
      </div>
    </div>
  </details>
</footer>

<style>
  /* Chevron points up while closed (the sheet opens upward) and down while open */
  summary::before {
    content: "";
    position: absolute;
    left: 2px;
    top: 50%;
    width: 7px;
    height: 7px;
    margin-top: -2px;
    border-right: 1.75px solid var(--text-muted);
    border-bottom: 1.75px solid var(--text-muted);
    transform: rotate(-135deg);
    transition: transform 0.2s, margin-top 0.2s;
  }

  details[open] summary::before {
    transform: rotate(45deg);
    margin-top: -6px;
  }

  summary:hover::before {
    border-color: var(--accent-text);
  }

  summary:hover > :first-child {
    color: var(--accent-text);
  }

  /* Boxed like the menu's key hints */
  .term-badge {
    display: inline-grid;
    place-items: center;
    min-width: 34px;
    height: 20px;
    padding: 0 6px;
    border: 1px solid var(--rule);
    background: var(--accent-soft);
    color: var(--accent-text);
    font-size: 12.5px;
    font-weight: 700;
    text-transform: uppercase;
  }
</style>
