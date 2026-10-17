<script lang="ts">
  // An open card: the selected horizon's verdict and its numbers first, then every horizon side by side,
  // then the supporting signals and the history footer.
  import { app, reloadCoin, removeCoin } from "../lib/live.svelte";
  import { coindcxUrl, formatNumber, formatPrice, formatUsdCompact, isNum, pct, pctile, pText, signed, sp, spanText, splitSymbol, toneClass, whenText } from "../lib/format";
  import { afterText, CHARACTER, characterText, hasMove, horizonOf, HORIZONS, isBusy, loadingDetail, moveText, rangeText, signalVerdict, significant } from "../lib/momentum";
  import type { Coin } from "../lib/types";
  import Icon from "./Icon.svelte";

  let { coin, symbol }: { coin: Coin; symbol: string } = $props();

  const tf = $derived(app.timeframe);
  const h = $derived(horizonOf(coin, tf));
  const verdict = $derived(signalVerdict(coin, tf));
  const m = $derived(coin.momentum || {});
  const horizons = $derived(m.horizons || {});
  const signals = $derived(coin.signals || {});
  const book = $derived(coin.orderbook_depth || {});

  const historyNote = $derived.by(() => {
    const hist = coin.history || {};
    const parts = Object.keys(hist).filter((iv) => hist[iv]?.candles).map((iv) => `${spanText(hist[iv].from)} of ${iv}`);
    return parts.length ? `Statistics from ${parts.join(" and ")} candles` : "";
  });

  // Section heads: a square bullet, the label in red caps, then a quiet note
  const H3 = "caps bullet mb-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 font-semibold";
  const NOTE = "mr-auto text-[13px] font-medium tracking-normal text-faint normal-case";
  const TH = "caps border-b border-border px-4 py-2 text-left text-xs font-semibold whitespace-nowrap text-muted";
</script>

<!-- The selected horizon's numbers stay at full contrast whatever they say: the panel's color tells long from short -->
{#snippet stat(label: string, value: string, sub: string, tip: { title: string; body: string } | null)}
  <div class="flex min-w-0 flex-col items-start gap-[3px]" data-tooltip-title={tip?.title} data-tooltip={tip?.body}>
    <span class="caps text-xs font-semibold text-muted">{label}</span>
    <span class="text-[28px] font-semibold tracking-[-0.01em] whitespace-nowrap text-fg tabular-nums max-sm:text-[23px]">{value}</span>
    {#if sub}<span class="text-[13.5px] text-fg-2 tabular-nums">{sub}</span>{/if}
  </div>
{/snippet}

{#snippet rsi(v: number | null | undefined)}
  {#if isNum(v)}
    <strong class={v >= 70 || v <= 30 ? "val-warn" : ""}>{formatNumber(v, 0)}</strong>
  {:else}
    <span class="val-muted">—</span>
  {/if}
{/snippet}

{#snippet signalLine(label: string, tip: string)}
  <span class="min-w-24 font-bold text-fg-2" data-tooltip-title={label} data-tooltip={tip}>{label}</span>
{/snippet}

<div class="grid gap-3 pt-4">
  {#if coin.has_data && h}
    {@const a = h.after}
    {@const vs = h.versus}
    {@const flow = h.flow}
    {@const base = vs ? splitSymbol(vs.symbol).base : ""}
    <!-- The numbers you act on, at full contrast. The side's color says whether it is a trade. -->
    <section class="min-w-0 border border-rule bg-surface px-4 py-3.5 shadow-[inset_0_2px_0_var(--accent)]">
      <h3 class="{H3} text-[13px] text-accent-text">
        Momentum · {tf}
        <span class={NOTE}>as of {whenText(h.as_of)} · {h.candles} × {h.interval} candles · fees {formatNumber(m.round_trip_pct, 2)}% round trip</span>
      </h3>
      <div
        class={[
          "min-w-0 border px-4 py-3.5",
          verdict.state === "go-long" && "border-pos-fill bg-pos-soft shadow-[inset_3px_0_0_var(--pos-fill)]",
          verdict.state === "go-short" && "border-neg-fill bg-neg-soft shadow-[inset_3px_0_0_var(--neg-fill)]",
          verdict.state === "idle" && "border-border-strong bg-surface-raised",
        ]}
      >
        <div class="flex items-center justify-between gap-2 text-xl font-bold tracking-[0.08em] text-fg uppercase">
          <span>{tf} momentum</span><span class="pill {verdict.pill}">{verdict.label}</span>
        </div>
        <p class="mt-1.5 text-[15px] leading-[1.45] font-medium text-fg">{verdict.why}</p>
        {#if hasMove(h)}
          <div class="mt-3.5 grid grid-cols-5 gap-x-[18px] gap-y-3.5 max-lg:grid-cols-3 max-sm:grid-cols-2">
            {@render stat(`Move · last ${tf}`, sp(h.return_pct), `z ${signed(h.z, 1)} · ${pctile(h.percentile)} pct`,
              { title: "Move", body: moveText(coin, h, tf) })}
            {@render stat(`Next ${tf} after similar`, a ? sp(a.mean_pct) : "—", a ? `95%: ${rangeText(a)}` : "",
              a ? { title: "After similar moves", body: afterText(a, tf) } : null)}
            {@render stat("Rose afterwards", a ? pct(a.hit_rate) : "—", a ? `all windows ${pct(a.all_hit_rate)}` : "",
              { title: "Share up", body: `How often the next ${tf} closed higher after moves like this, against every ${tf} window.` })}
            {@render stat(vs ? `vs ${base} · beta-adjusted` : "vs benchmark", vs ? sp(vs.excess_pct) : "—",
              vs ? `${base} ${sp(vs.return_pct)} · β ${formatNumber(vs.beta, 2)}` : "",
              vs ? { title: `Against ${base}`, body: `The coin's last ${tf} return minus ${formatNumber(vs.beta, 2)} × ${base}'s (${sp(vs.return_pct)}): how much it moved on its own. Beta is from ${h.interval} returns over the last ${formatNumber(m.vol_days, 0)} days.` } : null)}
            {@render stat(`Taker buys · last ${tf}`, flow ? `${formatNumber(flow.buy_pct, 1)}%` : "—", flow ? `${pctile(flow.percentile)} pct of past ${tf}` : "",
              { title: "Taker flow", body: "Share of the traded value bought with market orders. 50% is balanced." })}
          </div>
        {/if}
      </div>
    </section>
  {/if}

  {#if coin.has_data && Object.keys(horizons).length}
    <section class="min-w-0 overflow-x-auto border border-border bg-surface pt-1.5 pb-1">
      <h3 class="{H3} px-4 pt-2 pb-1 text-[12.5px] text-muted">Every horizon <span class={NOTE}>click a row to select it</span></h3>
      <table class="w-full border-collapse text-sm tabular-nums">
        <thead>
          <tr>
            <th class={TH}>Horizon</th>
            <th class="{TH} text-right" data-tooltip-title="Move" data-tooltip="Return over the last horizon, its z (the move in units of a typical move of that length) and its percentile among past moves.">Last move</th>
            <th class="{TH} text-right" data-tooltip-title="Next, after similar moves" data-tooltip="Mean return over the next horizon after past moves in the same fifth as the latest one, with the 95% range of that mean.">Next, after similar</th>
            <th class="{TH} text-right" data-tooltip-title="Rose / all" data-tooltip="How often the next horizon closed higher after similar moves, and after every window.">Rose / all</th>
            <th class="{TH} text-right" data-tooltip-title="Older / newer" data-tooltip="The same mean in the older and the newer half of the history. A real effect should show in both.">Older / newer</th>
            <th class="{TH} text-right" data-tooltip-title="Past → next correlation" data-tooltip="Correlation of each move with the move that followed it, over the whole history (information coefficient). Positive: momentum. Negative: reversal.">Past → next</th>
            <th class="{TH} text-right" data-tooltip-title="Variance ratio" data-tooltip="Lo–MacKinlay variance ratio: variance of moves this long over the variance a random walk would give. Above 1 trending, below 1 reverting.">Var. ratio</th>
            <th class={TH}>Signal</th>
          </tr>
        </thead>
        <tbody>
          {#each HORIZONS as name (name)}
            {@const hz = horizonOf(coin, name)}
            <tr
              class={["hz-row cursor-pointer", name === tf && "is-current"]}
              onclick={(e) => { e.stopPropagation(); app.timeframe = name; }}
            >
              {#if !hasMove(hz)}
                <td><strong>{name}</strong></td>
                <td colspan="7" class="val-muted">Not enough history</td>
              {:else}
                {@const a = hz.after}
                {@const p = hz.persistence || {}}
                {@const v = signalVerdict(coin, name)}
                <td><strong>{name}</strong> <span class="val-muted">{hz.interval}</span></td>
                <td class="text-right">
                  <span class={Math.abs(hz.z) >= 1 ? toneClass(hz.return_pct) : ""}>{sp(hz.return_pct)}</span>
                  <span class="val-muted">z {signed(hz.z, 1)} · {pctile(hz.percentile)}</span>
                </td>
                <td class="text-right" data-tooltip-title="After similar moves" data-tooltip={a ? afterText(a, name) : null}>
                  {#if a}
                    <strong class={significant(a) ? toneClass(a.mean_pct) : ""}>{sp(a.mean_pct)}</strong>
                    <span class="val-muted">{rangeText(a)}</span>
                  {:else}—{/if}
                </td>
                <td class="text-right">{#if a}{pct(a.hit_rate)} <span class="val-muted">/ {pct(a.all_hit_rate)}</span>{:else}—{/if}</td>
                <td class="text-right">{#if a}{sp(a.older_mean_pct)} <span class="val-muted">/</span> {sp(a.newer_mean_pct)}{:else}—{/if}</td>
                <td class="text-right">{signed(p.ic, 3)} <span class="val-muted">{pText(p.ic_p)}</span></td>
                <td class="text-right" data-tooltip-title="Character" data-tooltip={characterText(hz, name)}>
                  {formatNumber(p.vr, 2)} <span class="val-muted">{p.character ? CHARACTER[p.character].label : ""}</span>
                </td>
                <td><span class="pill pill-sm {v.pill}" data-tooltip-title="{name} · {v.label}" data-tooltip={v.why}>{v.label}</span></td>
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}

  {#if coin.has_data && Object.keys(signals).length}
    {@const ranges = Object.entries(signals.ranges || {}).filter(([, r]) => r) as [string, { position: number; low: number; high: number }][]}
    <!-- Fine print: supporting detail that doesn't change a trade -->
    <section class="min-w-0 border border-border px-4 py-3.5">
      <h3 class="{H3} text-[12.5px] text-muted">Signals</h3>
      <ul class="grid gap-1.5 text-sm tabular-nums [&>li]:flex [&>li]:flex-wrap [&>li]:items-baseline [&>li]:gap-x-3.5 [&>li]:gap-y-0.5">
        <li>
          {@render signalLine("RSI 14", "Wilder's relative strength index on closed candles. Above 70 is stretched up, below 30 stretched down.")}
          <span class="flex gap-4">
            {#each Object.entries(signals.rsi || {}) as [iv, v] (iv)}<span>{iv} {@render rsi(v)}</span>{/each}
          </span>
        </li>
        {#if signals.ema}
          {@const e = signals.ema.ema || {}}
          {@const stack = ({ up: ["Up stack", "val-pos"], down: ["Down stack", "val-neg"], mixed: ["Mixed", ""] } as Record<string, string[]>)[signals.ema.stack] || ["—", ""]}
          <li>
            {@render signalLine(`Trend ${signals.ema.interval}`, `Price against its 20, 50 and 200-candle exponential moving averages (${formatPrice(e["20"])}, ${formatPrice(e["50"])}, ${formatPrice(e["200"])}). Up stack: price > EMA20 > EMA50 > EMA200.`)}
            <span><strong class={stack[1]}>{stack[0]}</strong> <span class="val-muted">· price {sp(signals.ema.vs_slowest_pct, 1)} vs EMA 200</span></span>
          </li>
        {/if}
        {#if ranges.length}
          <li>
            {@render signalLine("Range", `Where price sits between the low (0%) and high (100%) of each window.\n${ranges.map(([k, r]) => `${k}: ${formatPrice(r.low)} – ${formatPrice(r.high)}`).join("\n")}`)}
            <span class="flex gap-4">
              {#each ranges as [k, r] (k)}<span>{k} <strong>{pct(r.position)}</strong></span>{/each}
            </span>
          </li>
        {/if}
        {#if signals.volatility}
          <li>
            {@render signalLine("Volatility", "Realized volatility of the last 24 hours of 15m candles, scaled to a day, and its percentile among every earlier day on hand.")}
            <span><strong>{formatNumber(signals.volatility.daily_pct, 2)}%</strong> a day <span class="val-muted">· {pctile(signals.volatility.percentile)} pct</span></span>
          </li>
        {/if}
        {#if signals.volume}
          <li>
            {@render signalLine("Volume", `Traded value of the last 24 hours against the median of the 30 days before (${formatUsdCompact(signals.volume.typical_24h_usd)}).`)}
            <span><strong>{formatUsdCompact(signals.volume.last_24h_usd)}</strong> in 24h <span class="val-muted">· {formatNumber(signals.volume.ratio, 2)}× the median day</span></span>
          </li>
        {/if}
        {#if signals.funding}
          <li>
            {@render signalLine("Funding", `The latest settled funding rate (per settlement; longs pay shorts when positive). Binance's baseline is +0.0100%. Mean over ${signals.funding.days} days ${sp(signals.funding.mean_pct, 4)}.`)}
            <span><strong class={toneClass(signals.funding.rate_pct, 4)}>{sp(signals.funding.rate_pct, 4)}</strong> <span class="val-muted">· {pctile(signals.funding.percentile)} pct of {signals.funding.days} days</span></span>
          </li>
        {/if}
        {#if book.available}
          <li>
            {@render signalLine("Order book", "Resting limit orders on the bid side across the top 20 levels, the bid/ask imbalance of the top 5, and the spread. Display only.")}
            <span><strong>{book.bid_pct}%</strong> bids <span class="val-muted">· top 5 {signed(book.near_imbalance_pct, 0)}% · spread {formatNumber(book.spread_bps, 2)} bps</span></span>
          </li>
        {/if}
      </ul>
    </section>
  {/if}
</div>

<div class="mt-3.5 flex flex-wrap items-center gap-x-5 gap-y-2 text-[13.5px] text-muted">
  {#if isBusy(coin)}
    {@const p = coin.progress}
    <span class="text-warn">{loadingDetail(coin)}</span>
    {#if p && isNum(p.fraction) && coin.status !== "queued"}
      <span class="inline-block h-1 w-[120px] overflow-hidden bg-border" aria-hidden="true">
        <span class="block h-full bg-warn transition-[width] duration-400" style:width="{Math.round(p.fraction * 100)}%"></span>
      </span>
    {/if}
  {:else}
    <button class="btn btn-sm" onclick={() => reloadCoin(symbol)}>{coin.status === "failed" ? "Retry download" : "Reload history"}</button>
    {#if coin.status === "failed" && coin.error}<span class="val-neg">{coin.error}</span>{/if}
    {#if historyNote}<span class="val-muted">{historyNote}</span>{/if}
  {/if}
  <span class="flex-1"></span>
  <a class="inline-flex items-center gap-1 font-medium text-accent-text no-underline hover:underline" href={coindcxUrl(symbol)} target="_blank" rel="noopener noreferrer">
    Open on CoinDCX <Icon name="external" class="size-[13px]" />
  </a>
  <button class="btn btn-sm btn-ghost-danger" onclick={() => removeCoin(symbol)}>Remove</button>
</div>

<style>
  td {
    padding: 9px 16px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  tr:last-child td {
    border-bottom: 0;
  }

  /* Rows select the horizon; the selected one is marked like a selection */
  .hz-row:hover td {
    background: var(--surface-hover);
  }

  .hz-row.is-current td {
    background: var(--accent-soft);
  }

  .hz-row.is-current td:first-child {
    box-shadow: inset 2px 0 0 var(--accent);
  }
</style>
