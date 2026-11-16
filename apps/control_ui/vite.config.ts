import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  define: {'process.env.NODE_ENV': JSON.stringify('production')},
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    cssCodeSplit: false,
    target: 'es2022',
    lib: {entry: 'src/main.tsx', name: 'SensoryControlUI', formats: ['iife'], fileName: () => 'rate_control.js'},
    rollupOptions: {output: {assetFileNames: (asset) => asset.name?.endsWith('.css') ? 'rate_control.css' : '[name][extname]'}},
  },
});
