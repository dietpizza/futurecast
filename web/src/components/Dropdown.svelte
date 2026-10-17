<script lang="ts">
  // A dropdown in the page's own style instead of the browser's: a square field with a chevron, and a
  // menu whose current option sits on a soft cyan wash. Closes on a choice, a click
  // outside, Escape or Tab. Keys: Enter / Space / ↓ open; ↑ ↓ move; Enter picks; Home / End jump.
  import { tick } from "svelte";
  import Icon from "./Icon.svelte";

  type Option = { value: string; label: string };
  let { options, value, onchange, label, class: cls = "" }:
    { options: Option[]; value: string; onchange: (value: string) => void; label: string; class?: string } = $props();

  let open = $state(false);
  let active = $state(0); // the option the keyboard is on
  let root: HTMLDivElement;
  let menu: HTMLUListElement | undefined = $state();
  const id = `dd-${Math.random().toString(36).slice(2, 8)}`;
  const current = $derived(options.find((o) => o.value === value) ?? options[0]);

  async function show(): Promise<void> {
    active = Math.max(0, options.findIndex((o) => o.value === value));
    open = true;
    await tick();
    menu?.focus();
  }

  function hide(refocus = true): void {
    open = false;
    if (refocus) root.querySelector<HTMLButtonElement>("button")?.focus();
  }

  function choose(i: number): void {
    const option = options[i];
    hide();
    if (option && option.value !== value) onchange(option.value);
  }

  function onButtonKey(e: KeyboardEvent): void {
    if (["ArrowDown", "ArrowUp"].includes(e.key)) {
      e.preventDefault();
      show();
    }
  }

  function onMenuKey(e: KeyboardEvent): void {
    const last = options.length - 1;
    const moves: Record<string, () => number> = {
      ArrowDown: () => Math.min(last, active + 1),
      ArrowUp: () => Math.max(0, active - 1),
      Home: () => 0,
      End: () => last,
    };
    if (moves[e.key]) {
      e.preventDefault();
      active = moves[e.key]();
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      choose(active);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation(); // the page's own Escape (closing the add-pair drawer) shouldn't fire too
      hide();
    } else if (e.key === "Tab") {
      hide(false);
    }
  }

  // Any press outside the dropdown closes it
  function onWindowPointer(e: PointerEvent): void {
    if (open && !root.contains(e.target as Node)) hide(false);
  }
</script>

<svelte:window onpointerdown={onWindowPointer} />

<div class="relative {cls}" bind:this={root}>
  <button
    type="button"
    class={["dd-field", open && "is-open"]}
    aria-haspopup="listbox"
    aria-expanded={open}
    aria-controls={id}
    aria-label="{label}: {current?.label}"
    onclick={() => (open ? hide() : show())}
    onkeydown={onButtonKey}
  >
    <span class="truncate">{current?.label}</span>
    <Icon name="chevron" class="size-3.5 transition-transform duration-150 {open ? 'rotate-180' : ''}" />
  </button>

  {#if open}
    <ul
      bind:this={menu}
      id={id}
      class="dd-menu"
      role="listbox"
      tabindex="-1"
      aria-label={label}
      aria-activedescendant="{id}-{active}"
      onkeydown={onMenuKey}
    >
      {#each options as option, i (option.value)}
        <li
          id="{id}-{i}"
          role="option"
          aria-selected={option.value === value}
          class={["dd-option", option.value === value && "is-selected", i === active && "is-active"]}
          onpointerenter={() => (active = i)}
          onclick={() => choose(i)}
          onkeydown={() => {}}
        >{option.label}</li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .dd-field {
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    width: 100%;
    height: 30px;
    padding: 0 8px 0 10px;
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--text);
    font-size: 14px;
    font-weight: 600;
    text-align: left;
    transition: border-color 0.15s, background 0.15s;
  }

  .dd-field:hover,
  .dd-field.is-open {
    border-color: var(--accent);
    background: var(--accent-soft);
  }

  .dd-field :global(.icon) {
    color: var(--text-muted);
  }

  .dd-field.is-open :global(.icon),
  .dd-field:hover :global(.icon) {
    color: var(--accent-text);
  }

  /* The menu: a framed panel with the HUD's top accent line, options like the window selector's */
  .dd-menu {
    position: absolute;
    z-index: 30;
    top: calc(100% + 4px);
    left: 0;
    min-width: 100%;
    border: 1px solid var(--rule);
    background: var(--surface-raised);
    box-shadow: var(--shadow-lg), inset 0 2px 0 var(--accent);
    outline: none;
  }

  .dd-option {
    display: flex;
    align-items: center;
    height: 28px;
    padding: 0 12px 0 10px;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    color: var(--text-2);
    cursor: pointer;
  }

  .dd-option.is-active {
    background: var(--accent-soft);
    color: var(--accent-text);
  }

  /* The current option: a quieter cyan wash than a full fill, stronger than the hover one */
  .dd-option.is-selected {
    background: color-mix(in srgb, var(--accent) 22%, transparent);
    color: var(--accent-text);
    font-weight: 700;
  }
</style>
