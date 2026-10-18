import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

// In development the page is served here (127.0.0.1:5000) and talks to the Go backend (127.0.0.1:5001)
// through this proxy; `mise dev` starts both, and the backend's / redirects here. In production the
// backend serves the built page itself (compiled into bin/deltarisk by `mise run build`).
const api = `127.0.0.1:${process.env.DELTARISK_API_PORT || "5001"}`;

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
  server: {
    // 127.0.0.1, not "localhost": Node resolves that to ::1 only, and http://127.0.0.1:5000 then refuses
    host: "127.0.0.1",
    port: Number(process.env.DELTARISK_WEB_PORT || 5000),
    strictPort: true,
    proxy: {
      "/api": `http://${api}`,
      "/ws": { target: `ws://${api}`, ws: true },
    },
  },
});
