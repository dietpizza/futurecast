import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

// In development the page is served here and talks to the FastAPI backend through this proxy;
// `mise dev` starts both. In production the backend serves the built files from web/dist.
const api = `127.0.0.1:${process.env.DELTARISK_API_PORT || "5001"}`;

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
  server: {
    port: Number(process.env.DELTARISK_WEB_PORT || 5000),
    strictPort: true,
    proxy: {
      "/api": `http://${api}`,
      "/ws": { target: `ws://${api}`, ws: true },
    },
  },
});
