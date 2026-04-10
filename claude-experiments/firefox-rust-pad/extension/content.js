(() => {
  const EVAL_SERVER = "http://localhost:7483";

  function getMonacoContent() {
    // Monaco stores its model globally; we can access it via the editor instance
    // Rustpad exposes the Monaco editor on the page
    const editorEl = document.querySelector(".monaco-editor");
    if (!editorEl) return null;

    // Monaco attaches the editor instance to the DOM element's parent
    // We can use the Monaco API that's available on the page
    // Inject a script to access the Monaco editor model
    return new Promise((resolve) => {
      const script = document.createElement("script");
      script.textContent = `
        (function() {
          try {
            // Monaco editor instances are tracked globally
            const editors = window.monaco?.editor?.getEditors?.() || [];
            if (editors.length > 0) {
              const content = editors[0].getValue();
              document.dispatchEvent(new CustomEvent('rustpad-eval-content', { detail: content }));
              return;
            }
            // Fallback: try to get from the model
            const models = window.monaco?.editor?.getModels?.() || [];
            if (models.length > 0) {
              const content = models[0].getValue();
              document.dispatchEvent(new CustomEvent('rustpad-eval-content', { detail: content }));
              return;
            }
            document.dispatchEvent(new CustomEvent('rustpad-eval-content', { detail: null }));
          } catch(e) {
            document.dispatchEvent(new CustomEvent('rustpad-eval-content', { detail: null }));
          }
        })();
      `;

      const handler = (e) => {
        document.removeEventListener("rustpad-eval-content", handler);
        resolve(e.detail);
      };
      document.addEventListener("rustpad-eval-content", handler);
      document.head.appendChild(script);
      script.remove();
    });
  }

  function detectLanguage(code) {
    // Simple heuristics
    if (/^\s*\(/.test(code) || /defn |defmacro |ns |let \[/.test(code)) return "clojure";
    if (/fn\s+\w+|let\s+mut\s|use\s+std::|impl\s|struct\s+\w+\s*\{|println!\(/.test(code)) return "rust";
    return "javascript";
  }

  function createUI() {
    // Bottom bar
    const bar = document.createElement("div");
    bar.id = "rustpad-eval-bar";
    bar.innerHTML = `
      <select id="rustpad-lang-select">
        <option value="auto">Auto-detect</option>
        <option value="javascript">JavaScript</option>
        <option value="rust">Rust</option>
        <option value="clojure">Clojure</option>
      </select>
      <button id="rustpad-eval-btn">Run</button>
      <span class="status" id="rustpad-eval-status"></span>
    `;

    // Output panel
    const output = document.createElement("div");
    output.id = "rustpad-eval-output";
    output.innerHTML = `
      <div id="rustpad-eval-output-header">
        <span>Output</span>
        <button id="rustpad-eval-close">\u00d7</button>
      </div>
      <pre id="rustpad-eval-output-pre"></pre>
    `;

    // Invisible hover zone at bottom of screen
    const hoverZone = document.createElement("div");
    hoverZone.id = "rustpad-eval-hover-zone";

    document.body.appendChild(output);
    document.body.appendChild(bar);
    document.body.appendChild(hoverZone);

    document.getElementById("rustpad-eval-close").addEventListener("click", () => {
      output.classList.remove("visible");
    });

    document.getElementById("rustpad-eval-btn").addEventListener("click", runEval);

    // Show bar after 3s hover or on first eval
    let locked = false;
    let hoverTimer;
    function lockBar() {
      locked = true;
      bar.classList.add("visible");
    }

    hoverZone.addEventListener("mouseenter", () => {
      if (!locked) {
        hoverTimer = setTimeout(lockBar, 3000);
      }
    });
    hoverZone.addEventListener("mouseleave", () => {
      clearTimeout(hoverTimer);
    });

    // Keyboard shortcut: Cmd/Ctrl+Enter — use capture phase to beat Monaco
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        lockBar();
        runEval();
      }
    }, true);
  }

  async function runEval() {
    const status = document.getElementById("rustpad-eval-status");
    const outputPanel = document.getElementById("rustpad-eval-output");
    const outputPre = document.getElementById("rustpad-eval-output-pre");
    const langSelect = document.getElementById("rustpad-lang-select");

    status.textContent = "Getting code...";

    const code = await getMonacoContent();
    if (code === null) {
      status.textContent = "Could not read editor content";
      return;
    }

    const lang = langSelect.value === "auto" ? detectLanguage(code) : langSelect.value;
    status.textContent = `Running (${lang})...`;

    try {
      const resp = await fetch(`${EVAL_SERVER}/eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, language: lang }),
      });

      const result = await resp.json();

      outputPre.className = result.success ? "success" : "error";
      outputPre.textContent = result.output || "(no output)";
      outputPanel.classList.add("visible");
      status.textContent = result.success ? "Done" : "Error";
    } catch (e) {
      outputPre.className = "error";
      outputPre.textContent = `Failed to connect to eval server at ${EVAL_SERVER}\n\nMake sure the server is running:\n  cd firefox-rust-pad/server && node server.js`;
      outputPanel.classList.add("visible");
      status.textContent = "Server unreachable";
    }
  }

  // Wait for Monaco to load
  function waitForEditor() {
    if (document.querySelector(".monaco-editor")) {
      createUI();
    } else {
      setTimeout(waitForEditor, 500);
    }
  }

  waitForEditor();
})();
