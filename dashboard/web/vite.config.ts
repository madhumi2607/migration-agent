import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Vite server proxies API calls to FastAPI (uvicorn on port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
