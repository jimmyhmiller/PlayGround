// WASM HTML generator - creates standalone HTML files with embedded WASM

use crate::compilers::ion::schema::IonJSON;
use std::fs;
use std::io;
use std::path::Path;

/// Helper function to serialize IonJSON to a string
fn serialize_ion_json(ion_json: &IonJSON) -> Result<String, io::Error> {
    #[cfg(feature = "serde")]
    {
        serde_json::to_string(ion_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
    #[cfg(not(feature = "serde"))]
    {
        use crate::json::ToJson;
        Ok(ion_json.to_json().to_string())
    }
}

/// Generate a standalone WASM viewer (no embedded JSON, supports drag-and-drop)
pub fn generate_wasm_viewer(output_path: &str) -> io::Result<()> {
    // Read WASM binary and base64 encode it
    let wasm_path = "pkg/iongraph_rust_redux_bg.wasm";
    if !Path::new(wasm_path).exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "WASM binary not found at '{}'. Please run 'wasm-pack build --target web --out-dir pkg' first.",
                wasm_path
            ),
        ));
    }

    let wasm_binary = fs::read(wasm_path)?;
    let wasm_base64 = base64_encode(&wasm_binary);

    // Read WASM JS glue code
    let wasm_js_path = "pkg/iongraph_rust_redux.js";
    if !Path::new(wasm_js_path).exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "WASM JS glue code not found at '{}'. Please run 'wasm-pack build --target web --out-dir pkg' first.",
                wasm_js_path
            ),
        ));
    }

    let wasm_js = fs::read_to_string(wasm_js_path)?;

    // Escape backticks in the WASM JS to avoid breaking template literals
    let wasm_js = wasm_js.replace("`", "\\`").replace("${", "\\${");

    // Generate the HTML
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IonGraph WASM Viewer</title>
  <style>
{css}

/* Drop zone styles */
.drop-zone {{
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  border: 3px dashed #ccc;
  border-radius: 10px;
  z-index: 1000;
}}

.drop-zone.drag-over {{
  background: #e3f2fd;
  border-color: #2196F3;
}}

.drop-zone h2 {{
  color: #666;
  margin: 20px 0;
}}

.drop-zone p {{
  color: #999;
  margin: 10px 0;
}}

.file-input-btn {{
  margin: 20px;
  padding: 12px 24px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
}}

.file-input-btn:hover {{
  background: #1976D2;
}}

.hidden {{
  display: none;
}}

.loading {{
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  z-index: 2000;
}}
  </style>
</head>
<body>
  <div id="drop-zone" class="drop-zone">
    <h2>🎯 IonGraph WASM Viewer</h2>
    <p>Drag and drop an Ion JSON file here</p>
    <p>or</p>
    <button class="file-input-btn" onclick="document.getElementById('file-input').click()">
      Choose File
    </button>
    <input type="file" id="file-input" accept=".json" class="hidden">
    <div style="margin-top: 30px; color: #999; font-size: 12px;">
      <p>Supported: Ion JSON files from SpiderMonkey JIT compiler</p>
    </div>
  </div>

  <div id="app" class="hidden">
    <div class="ig-sidebar">
      <div style="padding: 10px; border-bottom: 1px solid #ddd;">
        <button class="file-input-btn" style="margin: 0; padding: 8px 16px; font-size: 14px;" onclick="document.getElementById('file-input').click()">
          Load Different File
        </button>
      </div>
      <select id="function-selector" class="ig-function-selector"></select>
      <div id="pass-sidebar" class="ig-pass-sidebar"></div>
    </div>
    <div class="ig-viewport" id="viewport"></div>
  </div>

  <div id="loading" class="loading hidden">
    <div>Loading and rendering...</div>
  </div>

  <script>
    // Embedded WASM binary (base64)
    const WASM_BASE64 = "{wasm_base64}";

    // Current loaded data
    let ION_DATA = null;

    // Decode base64 to bytes
    function base64ToBytes(base64) {{
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
      }}
      return bytes;
    }}

    // File handling
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const app = document.getElementById('app');
    const loading = document.getElementById('loading');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {{
      dropZone.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    }});

    function preventDefaults(e) {{
      e.preventDefault();
      e.stopPropagation();
    }}

    // Highlight drop zone when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {{
      dropZone.addEventListener(eventName, () => {{
        dropZone.classList.add('drag-over');
      }}, false);
    }});

    ['dragleave', 'drop'].forEach(eventName => {{
      dropZone.addEventListener(eventName, () => {{
        dropZone.classList.remove('drag-over');
      }}, false);
    }});

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {{
      const dt = e.dataTransfer;
      const files = dt.files;
      handleFiles(files);
    }}, false);

    // Handle file input
    fileInput.addEventListener('change', (e) => {{
      handleFiles(e.target.files);
    }});

    function handleFiles(files) {{
      if (files.length === 0) return;

      const file = files[0];
      if (!file.name.endsWith('.json')) {{
        alert('Please select a JSON file');
        return;
      }}

      loading.classList.remove('hidden');

      const reader = new FileReader();
      reader.onload = (e) => {{
        try {{
          ION_DATA = JSON.parse(e.target.result);
          initializeViewer();
        }} catch (err) {{
          alert('Error parsing JSON: ' + err.message);
          loading.classList.add('hidden');
        }}
      }};
      reader.readAsText(file);
    }}

    function initializeViewer() {{
      dropZone.classList.add('hidden');
      app.classList.remove('hidden');

      // Will be initialized after WASM loads
      if (window.wasmReady) {{
        setupUI();
      }}
    }}
  </script>

  <script type="module">
    // Inline WASM glue code
    {wasm_js}

    // Initialize WASM using default export
    const wasmBytes = base64ToBytes(WASM_BASE64);
    await __wbg_init(wasmBytes);

    // Make IonGraphViewer class globally available
    window.IonGraphViewer = IonGraphViewer;

    // Mark WASM as ready
    window.wasmReady = true;

    // If data was loaded before WASM initialized, set up UI now
    if (ION_DATA) {{
      setupUI();
    }}

    // Interactive JavaScript
    window.setupUI = function() {{
      {interactive_js}

      loading.classList.add('hidden');
    }}
  </script>
</body>
</html>"#,
        css = include_str!("../assets/iongraph.css"),
        wasm_base64 = wasm_base64,
        wasm_js = wasm_js,
        interactive_js = generate_viewer_js()
    );

    fs::write(output_path, html)?;
    Ok(())
}

/// Generate a standalone HTML file with embedded WASM for client-side rendering
///
/// This generates a much smaller HTML file (~1-2MB) compared to pre-rendered HTML (~18MB)
/// by embedding the WASM binary and rendering graphs on-demand in the browser.
pub fn generate_wasm_html(ion_json: &IonJSON, output_path: &str) -> io::Result<()> {
    // Serialize IonJSON data
    let json_data = serialize_ion_json(ion_json)?;

    // Read WASM binary and base64 encode it
    let wasm_path = "pkg/iongraph_rust_redux_bg.wasm";
    if !Path::new(wasm_path).exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "WASM binary not found at '{}'. Please run 'wasm-pack build --target web --out-dir pkg' first.",
                wasm_path
            ),
        ));
    }

    let wasm_binary = fs::read(wasm_path)?;
    let wasm_base64 = base64_encode(&wasm_binary);

    // Read WASM JS glue code
    let wasm_js_path = "pkg/iongraph_rust_redux.js";
    if !Path::new(wasm_js_path).exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "WASM JS glue code not found at '{}'. Please run 'wasm-pack build --target web --out-dir pkg' first.",
                wasm_js_path
            ),
        ));
    }

    let wasm_js = fs::read_to_string(wasm_js_path)?;

    // Escape backticks in the WASM JS to avoid breaking template literals
    let wasm_js = wasm_js.replace("`", "\\`").replace("${", "\\${");

    // Generate the HTML
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IonGraph WASM Viewer</title>
  <style>
{css}
  </style>
</head>
<body>
  <div id="app">
    <div class="ig-sidebar">
      <select id="function-selector" class="ig-function-selector"></select>
      <div id="pass-sidebar" class="ig-pass-sidebar"></div>
    </div>
    <div class="ig-viewport" id="viewport"></div>
  </div>

  <script>
    // Embedded IonJSON data
    const ION_DATA = {json_data};

    // Embedded WASM binary (base64)
    const WASM_BASE64 = "{wasm_base64}";

    // Decode base64 to bytes
    function base64ToBytes(base64) {{
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
      }}
      return bytes;
    }}
  </script>

  <script type="module">
    // Inline WASM glue code
    {wasm_js}

    // Initialize WASM using default export
    const wasmBytes = base64ToBytes(WASM_BASE64);
    await __wbg_init(wasmBytes);

    // Make IonGraphViewer class globally available
    window.IonGraphViewer = IonGraphViewer;

    // Interactive JavaScript
    {interactive_js}
  </script>
</body>
</html>"#,
        css = include_str!("../assets/iongraph.css"),
        json_data = json_data,
        wasm_base64 = wasm_base64,
        wasm_js = wasm_js,
        interactive_js = generate_interactive_js()
    );

    fs::write(output_path, html)?;
    Ok(())
}

/// Generate the interactive JavaScript code for the standalone viewer (drag-and-drop)
fn generate_viewer_js() -> String {
    r#"
let currentFunction = 0;
let currentPass = 0;
let svgCache = new Map(); // Cache rendered SVGs
let viewer = null; // Stateful WASM viewer instance

// Pan/Zoom state
let zoom = 1.0;
let translation = { x: 0, y: 0 };
let isPanning = false;
let panStart = { x: 0, y: 0 };

const viewport = document.getElementById('viewport');
const funcSelector = document.getElementById('function-selector');
const passSidebar = document.getElementById('pass-sidebar');

// Clear previous state
svgCache.clear();
funcSelector.innerHTML = '';
passSidebar.innerHTML = '';

// Initialize viewer with IonJSON data (parses once, caches in WASM)
const ionJsonStr = JSON.stringify(ION_DATA);
viewer = new IonGraphViewer(ionJsonStr);

const numFuncs = viewer.get_function_count();

for (let i = 0; i < numFuncs; i++) {
  const opt = document.createElement('option');
  opt.value = i;
  const funcName = viewer.get_function_name(i);
  opt.textContent = `${i}: ${funcName}`;
  funcSelector.appendChild(opt);
}

// Pan/Zoom functions
function updateTransform() {
  const svgElement = viewport.querySelector('svg');
  if (svgElement) {
    svgElement.style.transform = `translate(${translation.x}px, ${translation.y}px) scale(${zoom})`;
    svgElement.style.transformOrigin = '0 0';
  }
}

function zoomAt(delta, focalX, focalY) {
  const rect = viewport.getBoundingClientRect();
  const focal = { x: focalX - rect.left, y: focalY - rect.top };
  const worldFocal = {
    x: (focal.x - translation.x) / zoom,
    y: (focal.y - translation.y) / zoom
  };

  const newZoom = Math.max(0.1, Math.min(10.0, zoom * delta));
  zoom = newZoom;

  translation.x = focal.x - worldFocal.x * zoom;
  translation.y = focal.y - worldFocal.y * zoom;

  updateTransform();
}

function pan(dx, dy) {
  translation.x += dx;
  translation.y += dy;
  updateTransform();
}

// Event handlers for pan/zoom
viewport.addEventListener('wheel', (e) => {
  e.preventDefault();

  if (e.ctrlKey || e.metaKey) {
    // Zoom
    const delta = Math.pow(1.5, -e.deltaY * 0.01);
    zoomAt(delta, e.clientX, e.clientY);
  } else {
    // Pan
    pan(-e.deltaX, -e.deltaY);
  }
}, { passive: false });

viewport.addEventListener('mousedown', (e) => {
  if (e.button === 0) { // Left mouse button
    isPanning = true;
    panStart = { x: e.clientX, y: e.clientY };
    viewport.style.cursor = 'grabbing';
    e.preventDefault();
  }
});

document.addEventListener('mousemove', (e) => {
  if (isPanning) {
    const dx = e.clientX - panStart.x;
    const dy = e.clientY - panStart.y;
    pan(dx, dy);
    panStart = { x: e.clientX, y: e.clientY };
  }
});

document.addEventListener('mouseup', (e) => {
  if (e.button === 0) {
    isPanning = false;
    viewport.style.cursor = 'default';
  }
});

// Render a specific pass
async function renderPass(funcIdx, passIdx) {
  const key = `${funcIdx}-${passIdx}`;

  // Check cache
  if (svgCache.has(key)) {
    viewport.innerHTML = svgCache.get(key);
    updateTransform(); // Apply current pan/zoom to cached SVG
    return;
  }

  // Show loading
  viewport.innerHTML = '<div style="padding: 20px; color: #666;">Rendering...</div>';

  try {
    // Call WASM to generate SVG (no JSON parsing - uses cached data!)
    const svg = viewer.render_pass(funcIdx, passIdx);

    // Cache and display
    svgCache.set(key, svg);
    viewport.innerHTML = svg;
    updateTransform(); // Apply current pan/zoom to new SVG
  } catch (err) {
    viewport.innerHTML = `<div style="color: red; padding: 20px;">Error: ${err}</div>`;
    console.error('Rendering error:', err);
  }
}

// Function switching
async function switchToFunction(funcIdx) {
  currentFunction = funcIdx;
  currentPass = 0;

  // Update pass sidebar
  const numPasses = viewer.get_pass_count(funcIdx);
  passSidebar.innerHTML = '';

  for (let i = 0; i < numPasses; i++) {
    const passDiv = document.createElement('div');
    passDiv.className = 'ig-pass' + (i === 0 ? ' ig-active' : '');
    const passName = viewer.get_pass_name(funcIdx, i);
    passDiv.textContent = `${i}: ${passName}`;
    passDiv.onclick = () => switchToPass(i);
    passSidebar.appendChild(passDiv);
  }

  await renderPass(funcIdx, 0);
}

async function switchToPass(passIdx) {
  currentPass = passIdx;

  // Update active state
  const passes = passSidebar.querySelectorAll('.ig-pass');
  passes.forEach((p, i) => {
    p.classList.toggle('ig-active', i === passIdx);
  });

  await renderPass(currentFunction, passIdx);
}

// Event handlers
funcSelector.addEventListener('change', e => switchToFunction(parseInt(e.target.value)));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;

  const numPasses = viewer.get_pass_count(currentFunction);

  switch(e.key) {
    case 'ArrowRight':
    case 'f':
      if (currentPass < numPasses - 1) {
        switchToPass(currentPass + 1);
      }
      break;
    case 'ArrowLeft':
    case 'r':
      if (currentPass > 0) {
        switchToPass(currentPass - 1);
      }
      break;
    case 'ArrowUp':
      if (currentFunction > 0) {
        funcSelector.value = currentFunction - 1;
        switchToFunction(currentFunction - 1);
      }
      break;
    case 'ArrowDown':
      if (currentFunction < numFuncs - 1) {
        funcSelector.value = currentFunction + 1;
        switchToFunction(currentFunction + 1);
      }
      break;
  }
});

// Initialize
switchToFunction(0);
"#
    .to_string()
}

/// Generate the interactive JavaScript code for the WASM viewer (embedded JSON)
fn generate_interactive_js() -> String {
    r#"
let currentFunction = 0;
let currentPass = 0;
let svgCache = new Map(); // Cache rendered SVGs
let viewer = null; // Stateful WASM viewer instance

const viewport = document.getElementById('viewport');
const funcSelector = document.getElementById('function-selector');
const passSidebar = document.getElementById('pass-sidebar');

// Initialize viewer with IonJSON data (parses once, caches in WASM)
const ionJsonStr = JSON.stringify(ION_DATA);
viewer = new IonGraphViewer(ionJsonStr);

const numFuncs = viewer.get_function_count();

for (let i = 0; i < numFuncs; i++) {
  const opt = document.createElement('option');
  opt.value = i;
  const funcName = viewer.get_function_name(i);
  opt.textContent = `${i}: ${funcName}`;
  funcSelector.appendChild(opt);
}

// Render a specific pass
async function renderPass(funcIdx, passIdx) {
  const key = `${funcIdx}-${passIdx}`;

  // Check cache
  if (svgCache.has(key)) {
    viewport.innerHTML = svgCache.get(key);
    updateTransform(); // Apply current pan/zoom to cached SVG
    return;
  }

  // Show loading
  viewport.innerHTML = '<div style="padding: 20px; color: #666;">Rendering...</div>';

  try {
    // Call WASM to generate SVG (no JSON parsing - uses cached data!)
    const svg = viewer.render_pass(funcIdx, passIdx);

    // Cache and display
    svgCache.set(key, svg);
    viewport.innerHTML = svg;
    updateTransform(); // Apply current pan/zoom to new SVG
  } catch (err) {
    viewport.innerHTML = `<div style="color: red; padding: 20px;">Error: ${err}</div>`;
    console.error('Rendering error:', err);
  }
}

// Function switching
async function switchToFunction(funcIdx) {
  currentFunction = funcIdx;
  currentPass = 0;

  // Update pass sidebar
  const numPasses = viewer.get_pass_count(funcIdx);
  passSidebar.innerHTML = '';

  for (let i = 0; i < numPasses; i++) {
    const passDiv = document.createElement('div');
    passDiv.className = 'ig-pass' + (i === 0 ? ' ig-active' : '');
    const passName = viewer.get_pass_name(funcIdx, i);
    passDiv.textContent = `${i}: ${passName}`;
    passDiv.onclick = () => switchToPass(i);
    passSidebar.appendChild(passDiv);
  }

  await renderPass(funcIdx, 0);
}

async function switchToPass(passIdx) {
  currentPass = passIdx;

  // Update active state
  const passes = passSidebar.querySelectorAll('.ig-pass');
  passes.forEach((p, i) => {
    p.classList.toggle('ig-active', i === passIdx);
  });

  await renderPass(currentFunction, passIdx);
}

// Event handlers
funcSelector.addEventListener('change', e => switchToFunction(parseInt(e.target.value)));

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;

  const numPasses = viewer.get_pass_count(currentFunction);

  switch(e.key) {
    case 'ArrowRight':
    case 'f':
      if (currentPass < numPasses - 1) {
        switchToPass(currentPass + 1);
      }
      break;
    case 'ArrowLeft':
    case 'r':
      if (currentPass > 0) {
        switchToPass(currentPass - 1);
      }
      break;
    case 'ArrowUp':
      if (currentFunction > 0) {
        funcSelector.value = currentFunction - 1;
        switchToFunction(currentFunction - 1);
      }
      break;
    case 'ArrowDown':
      if (currentFunction < numFuncs - 1) {
        funcSelector.value = currentFunction + 1;
        switchToFunction(currentFunction + 1);
      }
      break;
  }
});

// Initialize
switchToFunction(0);
"#
    .to_string()
}

/// Simple base64 encoding function
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    let mut i = 0;

    while i + 2 < data.len() {
        let b1 = data[i];
        let b2 = data[i + 1];
        let b3 = data[i + 2];

        result.push(CHARS[(b1 >> 2) as usize] as char);
        result.push(CHARS[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize] as char);
        result.push(CHARS[(((b2 & 0x0f) << 2) | (b3 >> 6)) as usize] as char);
        result.push(CHARS[(b3 & 0x3f) as usize] as char);

        i += 3;
    }

    // Handle remaining bytes
    if i < data.len() {
        let b1 = data[i];
        result.push(CHARS[(b1 >> 2) as usize] as char);

        if i + 1 < data.len() {
            let b2 = data[i + 1];
            result.push(CHARS[(((b1 & 0x03) << 4) | (b2 >> 4)) as usize] as char);
            result.push(CHARS[((b2 & 0x0f) << 2) as usize] as char);
            result.push('=');
        } else {
            result.push(CHARS[((b1 & 0x03) << 4) as usize] as char);
            result.push('=');
            result.push('=');
        }
    }

    result
}
