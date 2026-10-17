<script lang="ts">
  // An open row: the coin's tier and leverage (changeable), what the stop means in price, then every window
  // side by side (clicking a row selects that window), then the history footer.
  import { app, reloadCoin, removeCoin, selectWindow, updateCoin } from "../lib/live.svelte";
  import { coindcxUrl, formatNumber, isNum, pctile, sp, spanText, whenText } from "../lib/format";
  import { isBusy, loadingDetail, MOMENTUM_LABELS, momentumText, STATE_LABELS, stateText, TIERS, tierLabel, VOL_LABELS, volatilityText, WINDOWS, windowOf } from "../lib/risk";
  import type { Coin, Tier } from "../lib/types";
  import Dropdown from "./Dropdown.svelte";
  import Icon from "./Icon.svelte";
  import RiskCell from "./RiskCell.svelte";

  let { coin, symbol }: { coin: Coin; symbol: string } = $props();

  const leverage = $derived(coin.leverage || 1);
  const autoLeverage = $derived(TIERS.find((t) => t.value === coin.tier)?.leverage);
  const stopMove = $derived(app.stopPct / leverage);
  const liquidation = $derived(isNum(coin.risk?.mmr) ? 100 / leverage - 100 * coin.risk.mmr : null);
  const windows = $derived(coin.risk?.windows || {});
  const funding = $derived(coin.funding);

  const historyNote = $derived.by(() => {
    const hist = coin.history || {};
    const parts = Object.keys(hist).filter((iv) => hist[iv]?.candles).map((iv) => `${spanText(hist[iv].from)} of ${iv}`);
    return parts.length ? `Models from ${parts.join(", ")} candles` : "";
  });

  const tierOptions = $derived([
    { value: "", label: `Auto · ${tierLabel(coin.tier_auto)}` },
    ...TIERS.map((t) => ({ value: t.value, label: `${t.label} · ${t.leverage}×` })),
  ]);

  function setTier(value: string): void {
    updateCoin(symbol, { tier: (value || null) as Tier | null });
  }

  function setLeverage(e: Event & { currentTarget: HTMLInputElement }): void {
    const raw = e.currentTarget.value.trim();
    updateCoin(symbol, { leverage: raw ? Number(raw) : null });
  }

  const H3 = "caps bullet mb-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 font-semibold";
  const NOTE = "mr-auto text-[13px] font-medium tracking-normal text-faint normal-case";
  const TH = "caps border-b border-border px-4 py-1 text-left text-xs font-semibold whitespace-nowrap text-muted";
  // Text values in the settings strip match the inputs' height, so labels and values each share a line
  const VALUE = "block h-[30px] leading-[30px] whitespace-nowrap";
  const FIELD = "h-[30px] border border-border-strong bg-surface px-2 text-[14px] font-semibold text-fg focus:border-accent";
</script>

<div class="grid gap-3 pt-4">
  <section class="flex min-w-0 flex-wrap items-start gap-x-6 gap-y-3 border border-rule bg-surface px-4 py-3.5 shadow-[inset_0_2px_0_var(--accent)]">
    <div class="flex flex-col gap-1">
      <span class="caps text-xs font-semibold text-muted">Tier</span>
      <Dropdown class="w-[168px]" label="Tier" options={tierOptions} value={coin.overrides?.tier ?? ""} onchange={setTier} />
    </div>
    <label class="flex flex-col gap-1">
      <span class="caps text-xs font-semibold text-muted">Leverage</span>
      <input
        class="{FIELD} w-[110px] tabular-nums"
        type="number"
        min="1"
        max="125"
        step="1"
        placeholder="Auto · {autoLeverage}×"
        value={coin.overrides?.leverage ?? ""}
        onchange={setLeverage}
      />
    </label>
    <div class="flex flex-col gap-1 text-[14px] text-fg-2 tabular-nums">
      <span class="caps text-xs font-semibold text-muted">Your stop</span>
      <span class={VALUE}><strong class="text-fg">−{app.stopPct}%</strong> of margin at {leverage}× = <strong class="text-fg">{formatNumber(stopMove, 2)}%</strong> price</span>
    </div>
    <div
      class="flex flex-col gap-1 text-[14px] text-fg-2 tabular-nums"
      data-tooltip-title="Liquidation"
      data-tooltip="Isolated margin: about 1 / leverage minus the maintenance margin rate ({formatNumber(100 * (coin.risk?.mmr ?? 0), 1)}%, a fixed estimate; Binance's exact brackets need an account)."
    >
      <span class="caps text-xs font-semibold text-muted">Liquidation</span>
      <span class={VALUE}>≈ <strong class="text-fg">{formatNumber(liquidation, 2)}%</strong> price</span>
    </div>
    {#if funding}
      <div class="flex flex-col gap-1 text-[14px] text-fg-2 tabular-nums">
        <span class="caps th-now text-xs font-semibold">Funding</span>
        <span class={VALUE}><strong class="text-fg">{sp(funding.rate_pct, 4)}</strong> · {pctile(funding.percentile)} pct of {funding.days} days</span>
      </div>
    {/if}
  </section>

  {#if Object.keys(windows).length}
    <section class="min-w-0 overflow-x-auto border border-border bg-surface pt-1.5 pb-1">
      <h3 class="{H3} px-4 pt-2 pb-1 text-[12.5px] text-muted">
        Every window
        <span class={NOTE}>risk = chance the −{app.stopPct}% stop is hit within the window · click a row to select it{#if coin.risk?.trained_at}{` · models trained ${whenText(coin.risk.trained_at)}`}{/if}</span>
      </h3>
      <table class="w-full border-collapse text-sm tabular-nums">
        <thead>
          <tr>
            <th class={TH}>Window</th>
            <th class="{TH} group-start text-right"><span class="th-help th-fwd" data-tooltip-title="Long risk" data-tooltip="Chance a long opened now hits your stop within the window, and its 1-in-10 bad case.">Long risk</span></th>
            <th class="{TH} text-right"><span class="th-help th-fwd" data-tooltip-title="Short risk" data-tooltip="Chance a short opened now hits your stop within the window, and its 1-in-10 bad case.">Short risk</span></th>
            <th class="{TH} text-right"><span class="th-help th-fwd" data-tooltip-title="Swing" data-tooltip="The typical move expected over the next window: the volatility forecast the risk numbers are scaled by.">Swing</span></th>
            <th class="{TH} group-start"><span class="th-help th-now" data-tooltip-title="Momentum" data-tooltip="What price did over the last few windows, −100 to +100.">Momentum</span></th>
            <th class={TH}><span class="th-help th-now" data-tooltip-title="Market state" data-tooltip="Whether the last few windows were trending, choppy, quiet or wild.">State</span></th>
            <th class={TH}><span class="th-help th-now" data-tooltip-title="Volatility" data-tooltip="How wild the last few windows were against the last 30 days.">Volatility</span></th>
          </tr>
        </thead>
        <tbody>
          {#each WINDOWS as name (name)}
            {@const w = windowOf(coin, name)}
            <tr class={["win-row cursor-pointer", name === app.window && "is-current"]} onclick={(e) => { e.stopPropagation(); selectWindow(name); }}>
              <td><strong>{name}</strong></td>
              {#if !w}
                <td colspan="6" class="group-start val-muted">Not enough history</td>
              {:else}
                <td class="group-start text-right"><RiskCell {coin} {w} side="long" {name} compact /></td>
                <td class="text-right"><RiskCell {coin} {w} side="short" {name} compact /></td>
                <td class="text-right" data-tooltip-title="Swing · next {name}" data-tooltip="The typical move expected over the next {name}, from the volatility forecast.">
                  <span>±{formatNumber(w.swing_pct, 2)}%</span>
                </td>
                <td class="group-start" data-tooltip-title="Momentum · {name}" data-tooltip={momentumText(name)}>
                  {#if w.momentum_label && isNum(w.momentum)}
                    <span class={w.momentum_label === "bullish" ? "val-pos" : w.momentum_label === "bearish" ? "val-neg" : ""}>{MOMENTUM_LABELS[w.momentum_label]}</span>
                    <span class="val-muted">{w.momentum > 0 ? "+" : w.momentum < 0 ? "−" : ""}{Math.abs(Math.round(w.momentum))}</span>
                  {:else}—{/if}
                </td>
                <td data-tooltip-title="Market state · {name}" data-tooltip={stateText(w, name)}>
                  <span class={w.state === "wild" ? "val-warn" : ""}>{w.state ? STATE_LABELS[w.state] : "—"}</span>
                </td>
                <td data-tooltip-title="Volatility · {name}" data-tooltip={volatilityText(w, name)}>
                  <span class={w.volatility === "wild" ? "val-warn" : ""}>{w.volatility ? VOL_LABELS[w.volatility] : "—"}</span>
                  <span class="val-muted">{pctile(w.vol_pct)} pct</span>
                </td>
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
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
    padding: 2px 16px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  tr:last-child td {
    border-bottom: 0;
  }

  /* Rows select the window; the selected one is marked like a selection */
  .win-row:hover td {
    background: var(--surface-hover);
  }

  .win-row.is-current td {
    background: var(--accent-soft);
  }

  .win-row.is-current td:first-child {
    box-shadow: inset 2px 0 0 var(--accent);
  }
</style>
