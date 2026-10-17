<script lang="ts">
  // One side's risk: the stop-hit chance for the page's stop as a score, a bar, and the bad case under it.
  // A reading that failed the walk-forward check shows "—" with the reason. `compact` puts it on one line
  // (the all-windows table).
  import { app } from "../lib/live.svelte";
  import { formatNumber } from "../lib/format";
  import { reading, riskText, riskTone, uncheckedText } from "../lib/risk";
  import type { Coin, Side, WindowName, WindowRisk } from "../lib/types";

  let { coin, w, side, name, align = "right", compact = false }:
    { coin: Coin; w: WindowRisk; side: Side; name: WindowName; align?: "left" | "right"; compact?: boolean } = $props();

  const r = $derived(reading(coin, w, side, app.stopPct));
  const tone = $derived(riskTone(r.score));
  const title = $derived(`${side === "long" ? "Long" : "Short"} risk · ${name} · −${app.stopPct}% stop`);
</script>

{#if compact}
  <div
    class={["flex items-center gap-2 tabular-nums whitespace-nowrap", align === "right" ? "justify-end" : "justify-start"]}
    data-tooltip-title={title}
    data-tooltip={r.checked ? riskText(coin, w, r, name) : uncheckedText(r, name)}
  >
    {#if r.checked}
      <span class="text-[13px] text-dim">bad −{formatNumber(r.risk.bad_case_pct, 2)}%</span>
      <div class="risk-bar risk-{tone} w-12"><span style:width="{Math.max(2, r.score)}%"></span></div>
      <span class="risk-{tone} min-w-[38px] text-right text-[15px] font-bold">{r.score}<span class="text-[12px] font-semibold">%</span></span>
    {:else}
      <span class="text-[13px] text-dim">failed check</span>
      <span class="min-w-[38px] text-right text-[15px] font-semibold text-dim">—</span>
    {/if}
  </div>
{:else}
  <div
    class={["flex flex-col gap-1 tabular-nums", align === "right" ? "items-end" : "items-start"]}
    data-tooltip-title={title}
    data-tooltip={r.checked ? riskText(coin, w, r, name) : uncheckedText(r, name)}
  >
    {#if r.checked}
      <span class="risk-{tone} text-[18px] leading-tight font-bold whitespace-nowrap">{r.score}<span class="text-[13px] font-semibold">%</span></span>
      <div class="risk-bar risk-{tone} w-full"><span style:width="{Math.max(2, r.score)}%"></span></div>
      <span class="text-[13px] whitespace-nowrap text-dim">bad −{formatNumber(r.risk.bad_case_pct, 2)}%</span>
    {:else}
      <span class="text-[18px] leading-none font-semibold text-dim">—</span>
      <span class="text-[13px] whitespace-nowrap text-dim">failed check</span>
    {/if}
  </div>
{/if}
