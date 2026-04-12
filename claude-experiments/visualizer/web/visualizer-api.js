/**
 * Visualizer API
 *
 * Usage:
 *   import { createVisualizer } from './visualizer-api.js';
 *
 *   const viz = await createVisualizer(document.getElementById('container'));
 *   viz.load('(scene my-viz (rect :x 100 :y 100 :w 50 :h 50))');
 *   viz.step();
 *   viz.stepBack();
 *   viz.setPresent(true);
 *   viz.setControls(false);
 */

export async function createVisualizer(container, options = {}) {
    const {
        present = false,
        controls = true,
        editor = false,
        scene = null,
        theme = null,
    } = options;

    // Build DOM structure
    container.innerHTML = '';
    container.style.position = container.style.position || 'relative';

    const app = document.createElement('div');
    app.id = 'app';
    app.style.cssText = 'display:flex; height:100%; width:100%;';
    container.appendChild(app);

    // Editor pane (hidden by default in embed)
    const editorPane = document.createElement('div');
    editorPane.id = 'editor-pane';
    app.appendChild(editorPane);

    const editorHeader = document.createElement('div');
    editorHeader.id = 'editor-header';
    editorPane.appendChild(editorHeader);

    const sceneLabel = document.createElement('label');
    sceneLabel.textContent = 'scene';
    editorHeader.appendChild(sceneLabel);

    const sceneSelect = document.createElement('select');
    sceneSelect.id = 'scene-select';
    editorHeader.appendChild(sceneSelect);

    const editorContainer = document.createElement('div');
    editorContainer.id = 'editor-container';
    editorPane.appendChild(editorContainer);

    const resizeHandle = document.createElement('div');
    resizeHandle.id = 'resize-handle';
    app.appendChild(resizeHandle);

    // Canvas pane
    const canvasPane = document.createElement('div');
    canvasPane.id = 'canvas-pane';
    canvasPane.style.cssText = 'flex:1; position:relative; min-width:200px;';
    app.appendChild(canvasPane);

    // Controls
    const controlsDiv = document.createElement('div');
    controlsDiv.id = 'controls';
    controlsDiv.innerHTML = `
        <button id="btn-back" title="Back (←)">← back</button>
        <button id="btn-step" title="Step (→ or space)">step →</button>
    `;
    canvasPane.appendChild(controlsDiv);

    // Load WASM
    const wasm = await import('./visualizer.js');
    await wasm.default();

    // Wait a frame for WASM start() to register its window callbacks
    await new Promise(r => requestAnimationFrame(r));

    // Wire up button clicks
    controlsDiv.querySelector('#btn-back').addEventListener('click', () => {
        if (window.__stepBack) window.__stepBack();
    });
    controlsDiv.querySelector('#btn-step').addEventListener('click', () => {
        if (window.__step) window.__step();
    });

    // Apply initial options
    if (!editor) {
        editorPane.style.display = 'none';
        resizeHandle.style.display = 'none';
    }
    if (!controls) {
        controlsDiv.style.display = 'none';
    }
    if (theme) {
        wasm.theme_set_preset(theme);
    }
    if (scene) {
        if (window.__onCodeChange) window.__onCodeChange(scene);
    }

    // Public API
    return {
        load(code) {
            if (window.__onCodeChange) window.__onCodeChange(code);
        },

        step() {
            if (window.__step) window.__step();
        },

        stepBack() {
            if (window.__stepBack) window.__stepBack();
        },

        setPresent(on) {
            editorPane.style.display = on ? 'none' : '';
            resizeHandle.style.display = on ? 'none' : '';
        },

        setControls(on) {
            controlsDiv.style.display = on ? '' : 'none';
        },

        setEditor(on) {
            editorPane.style.display = on ? '' : 'none';
            resizeHandle.style.display = on ? '' : 'none';
        },

        setTheme(name) {
            return wasm.theme_set_preset(name);
        },

        setThemeField(key, value) {
            return wasm.theme_set_field(key, value);
        },

        getTheme() {
            return JSON.parse(wasm.theme_get_json());
        },

        get container() { return container; },
        get wasm() { return wasm; },
    };
}
