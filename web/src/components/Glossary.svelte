<!-- Plain-English glossary: a bar docked to the bottom of the window that opens upward -->
<footer class="plain-english-glossary sticky bottom-0 z-40 border-t border-border bg-header backdrop-blur-sm backdrop-saturate-180 has-open:bg-surface has-open:shadow-[0_-8px_24px_rgba(0,0,0,0.12)]">
  <details class="mx-auto max-w-(--page-w) px-6 max-sm:px-4">
    <summary class="relative flex cursor-pointer list-none flex-wrap items-baseline gap-x-3 gap-y-1 py-3 pl-[22px] select-none [&::-webkit-details-marker]:hidden">
      <span class="text-[15px] font-bold tracking-[0.08em] uppercase transition-colors">The "Not a Finance-Bro" Cheat Sheet</span>
      <span class="text-sm text-muted max-sm:hidden">Plain-English explanations of every number on this page</span>
    </summary>

    <div class="grid max-h-[min(60dvh,560px)] grid-cols-3 gap-7 overflow-y-auto overscroll-contain border-t border-border pb-5 max-md:grid-cols-1 max-md:gap-1">
      <div class="min-w-0">
        <h3 class="bullet mt-4 mb-3 flex items-center gap-[9px] text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">Momentum</h3>
        <dl class="flex flex-col gap-3.5">
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Move</span> Last move and z</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">The return over the last window (1h, 4h, 1d or 3d), divided by the size of a typical move that long: <strong class="font-semibold text-fg-2">z</strong>. A typical move comes from the last 14 days' volatility, so z 2 means twice the usual move, however wild the coin is. The percentile ranks it against every past move of that length.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Next</span> After similar moves</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">All past moves are split into fifths, from the hardest falls to the hardest rises. For the fifth the latest move belongs to: what the next window did on average, and how often it rose. Compared with every window, which shows the coin's plain drift.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">95%</span> Range of the mean</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Where the true average plausibly lies. Windows overlap (a 4h window starting every 15 minutes), so the range is widened for that (Newey–West). If it straddles zero, the average is indistinguishable from luck.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Signal</span> Long / Short</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Only when the latest move is outside the middle fifth, the average after similar moves beats a market round trip in fees (taker in and out, CoinDCX USDT-M futures with GST), its 95% range excludes zero, it shows in both the older and newer half of the history, and at least 30 non-overlapping windows stand behind it. A continuation follows the move; a reversal fades it.</dd>
          </div>
        </dl>
      </div>

      <div class="min-w-0">
        <h3 class="bullet mt-4 mb-3 flex items-center gap-[9px] text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">Does momentum persist?</h3>
        <dl class="flex flex-col gap-3.5">
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">VR</span> Variance ratio</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">The variance of moves of the horizon's length over what a random walk of one-candle moves would give (Lo–MacKinlay). <strong class="font-semibold text-fg-2">Above 1</strong>: moves keep their direction (trending). <strong class="font-semibold text-fg-2">Below 1</strong>: they reverse (mean-reverting). The p-value says how surprising the ratio would be if price were a random walk; under 0.05 counts.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">IC</span> Past → next correlation</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Correlation between each move and the move of the same length that followed it, over the whole history. Positive means momentum, negative means reversal. Values of a few hundredths are normal for markets and can still matter.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Halves</span> Older / newer</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">The same average computed on the older and the newer half of the history separately. An effect that shows up in only one half is more likely a fluke of that stretch.</dd>
          </div>
        </dl>
      </div>

      <div class="min-w-0">
        <h3 class="bullet mt-4 mb-3 flex items-center gap-[9px] text-[12.5px] font-semibold tracking-[0.08em] text-muted uppercase">Signals</h3>
        <dl class="flex flex-col gap-3.5">
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Flow</span> Taker flow</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Share of traded value bought with market orders. Above 50%, buyers were the aggressive side; the percentile compares it with past windows.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">RSI</span> Relative strength index</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Average gains against average losses over 14 candles, 0–100. Above 70 price has run up hard, below 30 down hard.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">EMA</span> Moving-average stack</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Price against its 20, 50 and 200-hour exponential moving averages. An up stack (price above all three, faster above slower) is an established uptrend.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Fund</span> Funding rate</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">What longs pay shorts (or the reverse when negative) at each settlement. A high percentile means crowded longs.</dd>
          </div>
          <div>
            <dt class="flex items-center gap-2 text-[15px] font-semibold text-fg"><span class="term-badge">Book</span> Order book imbalance</dt>
            <dd class="mt-[3px] text-sm leading-[1.55] text-dim">Buy versus sell limit orders waiting in the book. Heavy bids can cushion dips; heavy asks can cap rallies. Display only.</dd>
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
