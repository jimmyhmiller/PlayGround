// markpad — Tauri markdown editor (CodeMirror + preview). Both sides get the
// SAME minimal __TAURI_INTERNALS__ stub (from the earlier triage harness) so
// the desktop app boots in a plain browser; the session is stubbed to a single
// empty untitled document so the typing step starts from identical state.
import { clickText } from "../harness.mjs";

const APP =
  "/tmp/claude-1000/-home-jimmyhmiller-Documents-Code-Playground/19134627-eb88-4448-8dc7-b23ce19dea39/scratchpad/oss-triage/markpad";

const TAURI_STUB = `
  (() => {
    let __cb = 0;
    window.__TAURI_INTERNALS__ = {
      transformCallback(cb) {
        const id = ++__cb;
        window["_" + id] = cb;
        return id;
      },
      invoke(cmd) {
        if (cmd === "get_pending_files") return Promise.resolve([]);
        if (cmd === "load_session")
          return Promise.resolve({
            version: 2,
            items: [
              {
                kind: "untitled",
                path: null,
                name: "Untitled-1",
                dirty: false,
                text: "",
                saved_text: "",
                language: null,
              },
            ],
            active_index: 0,
          });
        return Promise.resolve(null);
      },
      metadata: {
        currentWindow: { label: "main" },
        currentWebview: { windowLabel: "main", label: "main" },
      },
      plugins: {},
    };
  })();
`;

export default {
  name: "markpad",
  appDir: APP,
  base: "/",
  initScript: TAURI_STUB,
  notes: [
    "Tauri IPC stubbed identically on both sides (same stub as the triage harness); session = one empty untitled doc.",
  ],
  steps: [
    {
      name: "initial-load",
      settle: 700,
      run: async (page, ctx) => {
        await page.goto(ctx.url, { waitUntil: "networkidle0", timeout: 30000 });
        await page.waitForSelector(".cm-content", { timeout: 15000 });
      },
    },
    {
      name: "type-markdown",
      settle: 900, // preview render is debounced; give it time to settle
      run: async (page) => {
        await page.click(".cm-content");
        await page.keyboard.type("# Hello *world*", { delay: 15 });
      },
      probe: async (page) =>
        page.evaluate(() => {
          const preview = document.querySelector(
            ".markdown-body, [class*=preview i]"
          );
          return {
            editorText: document.querySelector(".cm-content")?.textContent ?? null,
            previewHtml: preview
              ? preview.innerHTML.replace(/\s+/g, " ").trim().slice(0, 500)
              : null,
          };
        }),
    },
    {
      name: "mode-preview",
      settle: 700,
      run: async (page) => clickText(page, "Preview"),
    },
    {
      name: "mode-editor",
      settle: 700,
      run: async (page) => clickText(page, "Editor"),
    },
    {
      name: "mode-split",
      settle: 700,
      run: async (page) => clickText(page, "Split"),
    },
    {
      name: "toggle-dark-theme",
      settle: 700,
      run: async (page) => page.click('[aria-label="Switch to dark theme"]'),
    },
  ],
};
