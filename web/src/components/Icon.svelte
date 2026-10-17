<script lang="ts">
  // The page's line icons, drawn on a 16-unit grid with the current text color.
  const PATHS = {
    plus: "M8 3v10M3 8h10",
    close: "M4.5 4.5l7 7M11.5 4.5l-7 7",
    "close-lg": "M4 4l8 8M12 4l-8 8",
    "close-sm": "M5 5l6 6M11 5l-6 6",
    chevron: "M4 6l4 4 4-4",
    external: "M6 3H3.5v9.5H13V10M9 3h4v4M13 3L7.5 8.5",
    moon: "M13.2 10.1A5.6 5.6 0 0 1 5.9 2.8a5.6 5.6 0 1 0 7.3 7.3z",
    sun: "M8 1.5v1.25M8 13.25v1.25M1.5 8h1.25M13.25 8h1.25M3.4 3.4l.9.9M11.7 11.7l.9.9M3.4 12.6l.9-.9M11.7 4.3l.9-.9",
    search: "M10.5 10.5L14 14",
  };

  let { name, class: cls = "" }: { name: keyof typeof PATHS; class?: string } = $props();
</script>

<svg class="icon {cls}" viewBox="0 0 16 16" aria-hidden="true">
  {#if name === "sun"}<circle cx="8" cy="8" r="2.75" />{/if}
  {#if name === "search"}<circle cx="7" cy="7" r="4.5" />{/if}
  <path d={PATHS[name]} />
</svg>
