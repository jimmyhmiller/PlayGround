import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const ReactCompilerConfig = {
  /* React Compiler config options */
};

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler', ReactCompilerConfig]],
      },
    }),
  ],
  root: '.',
  build: {
    outDir: 'dist/renderer',
  },
});
