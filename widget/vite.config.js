import { defineConfig } from "vite";
import anywidget from "@anywidget/vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [anywidget(), react()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    outDir: "src/quantem/widget/static",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        show2d: "js/show2d/index.tsx",
        show4dstem: "js/show4dstem/index.tsx",
      },
      output: {
        entryFileNames: "[name].js",
        assetFileNames: "[name][extname]",
        format: "es",
        inlineDynamicImports: false,
      },
    },
  },
});
