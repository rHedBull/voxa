import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const BACKEND = process.env.VOXA_BACKEND || 'http://127.0.0.1:8765';

export default defineConfig({
  root: 'frontend',
  plugins: [react()],
  server: {
    port: Number(process.env.VOXA_FRONTEND_PORT || 5173),
    strictPort: true,
    // Vite serves the dev UI; the FastAPI backend handles /api/* (and is also
    // the production server for the built frontend).
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
    },
    // Use polling — avoids hitting the system-wide inotify watcher limit when
    // many other projects are also being watched. Slightly higher CPU but
    // works in any environment.
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  test: {
    include: ['src/**/*.test.{js,jsx}'],
    environment: 'node',
  },
});
