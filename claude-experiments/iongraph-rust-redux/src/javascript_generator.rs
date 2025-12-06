// JavaScript generator for interactive HTML

pub struct JavaScriptGenerator {
    pub num_passes: usize,
    pub key_passes: [Option<usize>; 4], // First MIR, last MIR, first LIR, last LIR
    pub redundant_passes: Vec<usize>,
    pub num_functions: Option<usize>, // If Some, multi-function mode
}

impl JavaScriptGenerator {
    pub fn new(num_passes: usize) -> Self {
        JavaScriptGenerator {
            num_passes,
            key_passes: [None, None, None, None],
            redundant_passes: Vec::new(),
            num_functions: None,
        }
    }

    pub fn with_key_passes(mut self, key_passes: [Option<usize>; 4]) -> Self {
        self.key_passes = key_passes;
        self
    }

    pub fn with_redundant_passes(mut self, redundant_passes: Vec<usize>) -> Self {
        self.redundant_passes = redundant_passes;
        self
    }

    pub fn with_multi_function_support(mut self, num_functions: usize) -> Self {
        self.num_functions = Some(num_functions);
        self
    }

    pub fn generate(&self) -> String {
        if let Some(num_functions) = self.num_functions {
            self.generate_multi_function(num_functions)
        } else {
            self.generate_single_function()
        }
    }

    fn generate_single_function(&self) -> String {
        let key_passes_str = format!(
            "[{}, {}, {}, {}]",
            self.key_passes[0]
                .map(|n| n.to_string())
                .unwrap_or("null".to_string()),
            self.key_passes[1]
                .map(|n| n.to_string())
                .unwrap_or("null".to_string()),
            self.key_passes[2]
                .map(|n| n.to_string())
                .unwrap_or("null".to_string()),
            self.key_passes[3]
                .map(|n| n.to_string())
                .unwrap_or("null".to_string())
        );

        let redundant_passes_str = format!(
            "[{}]",
            self.redundant_passes
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        format!(
            r#"
(function() {{
  'use strict';

  // Constants
  const NUM_PASSES = {};
  const KEY_PASSES = {};
  const REDUNDANT_PASSES = {};

  // State
  let currentPass = 0;
  let zoom = 1.0;
  let translation = {{ x: 0, y: 0 }};
  let selectedBlock = null;
  let isPanning = false;
  let panStart = {{ x: 0, y: 0 }};

  // DOM references
  const graphs = Array.from(document.querySelectorAll('.ig-graph'));
  const viewport = document.querySelector('.ig-viewport');
  const passDivs = Array.from(document.querySelectorAll('.ig-pass'));

  // Pass switching
  function switchToPass(passIndex) {{
    if (passIndex < 0 || passIndex >= NUM_PASSES) return;
    if (passIndex === currentPass) return;

    // Hide current graph and deactivate sidebar
    graphs[currentPass].classList.add('ig-hidden');
    passDivs[currentPass].classList.remove('ig-active');

    // Show new graph and activate sidebar
    currentPass = passIndex;
    graphs[currentPass].classList.remove('ig-hidden');
    passDivs[currentPass].classList.add('ig-active');

    updateTransform();
  }}

  function nextPass() {{
    for (let i = currentPass + 1; i < NUM_PASSES; i++) {{
      if (!REDUNDANT_PASSES.includes(i)) {{
        switchToPass(i);
        return;
      }}
    }}
  }}

  function prevPass() {{
    for (let i = currentPass - 1; i >= 0; i--) {{
      if (!REDUNDANT_PASSES.includes(i)) {{
        switchToPass(i);
        return;
      }}
    }}
  }}

  // Pan/Zoom transform
  function updateTransform() {{
    const graph = graphs[currentPass];
    graph.style.transform = `translate(${{translation.x}}px, ${{translation.y}}px) scale(${{zoom}})`;
  }}

  function screenToWorld(screenX, screenY) {{
    const rect = viewport.getBoundingClientRect();
    return {{
      x: (screenX - rect.left - translation.x) / zoom,
      y: (screenY - rect.top - translation.y) / zoom
    }};
  }}

  function worldToScreen(worldX, worldY) {{
    return {{
      x: worldX * zoom + translation.x,
      y: worldY * zoom + translation.y
    }};
  }}

  function zoomAt(delta, focalX, focalY) {{
    const rect = viewport.getBoundingClientRect();
    const focal = {{ x: focalX - rect.left, y: focalY - rect.top }};
    const worldFocal = {{
      x: (focal.x - translation.x) / zoom,
      y: (focal.y - translation.y) / zoom
    }};

    const newZoom = Math.max(0.1, Math.min(1.0, zoom * delta));
    zoom = newZoom;

    translation.x = focal.x - worldFocal.x * zoom;
    translation.y = focal.y - worldFocal.y * zoom;

    updateTransform();
  }}

  function pan(dx, dy) {{
    translation.x += dx;
    translation.y += dy;
    updateTransform();
  }}

  // Block selection
  function selectBlock(blockId) {{
    // Remove previous selection
    const selected = graphs[currentPass].querySelector('.ig-block.ig-selected');
    if (selected) {{
      selected.classList.remove('ig-selected', 'ig-last-selected');
    }}

    // Add new selection
    const block = graphs[currentPass].querySelector(`[data-block-id="${{blockId}}"]`);
    if (block) {{
      block.classList.add('ig-block', 'ig-selected', 'ig-last-selected');
      selectedBlock = blockId;
    }}
  }}

  function centerOnSelectedBlock() {{
    if (!selectedBlock) return;

    const block = graphs[currentPass].querySelector(`[data-block-id="${{selectedBlock}}"]`);
    if (!block) return;

    const rect = block.getBoundingClientRect();
    const viewportRect = viewport.getBoundingClientRect();

    // Get block position in world coordinates
    const blockStyle = block.style;
    const blockX = parseFloat(blockStyle.left) || 0;
    const blockY = parseFloat(blockStyle.top) || 0;
    const blockWidth = rect.width / zoom;
    const blockHeight = rect.height / zoom;

    // Center of block in world coords
    const blockCenterX = blockX + blockWidth / 2;
    const blockCenterY = blockY + blockHeight / 2;

    // Center of viewport in screen coords
    const viewportCenterX = viewportRect.width / 2;
    const viewportCenterY = viewportRect.height / 2;

    // Set zoom to 1 and translate to center
    zoom = 1.0;
    translation.x = viewportCenterX - blockCenterX * zoom;
    translation.y = viewportCenterY - blockCenterY * zoom;

    updateTransform();
  }}

  // Event handlers
  viewport.addEventListener('wheel', (e) => {{
    e.preventDefault();

    if (e.ctrlKey || e.metaKey) {{
      // Zoom
      const delta = Math.pow(1.5, -e.deltaY * 0.01);
      zoomAt(delta, e.clientX, e.clientY);
    }} else {{
      // Pan
      pan(-e.deltaX, -e.deltaY);
    }}
  }}, {{ passive: false }});

  viewport.addEventListener('mousedown', (e) => {{
    if (e.button === 0) {{ // Left mouse button
      isPanning = true;
      panStart = {{ x: e.clientX, y: e.clientY }};
      viewport.style.cursor = 'grabbing';
      e.preventDefault();
    }}
  }});

  document.addEventListener('mousemove', (e) => {{
    if (isPanning) {{
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      pan(dx, dy);
      panStart = {{ x: e.clientX, y: e.clientY }};
    }}
  }});

  document.addEventListener('mouseup', (e) => {{
    if (e.button === 0) {{
      isPanning = false;
      viewport.style.cursor = 'default';
    }}
  }});

  // Block click handler
  graphs.forEach((graph) => {{
    const blocks = graph.querySelectorAll('.ig-block');
    blocks.forEach((block) => {{
      block.addEventListener('click', (e) => {{
        const blockId = block.getAttribute('data-block-id');
        if (blockId) {{
          selectBlock(blockId);
        }}
        e.stopPropagation();
      }});
    }});
  }});

  // Sidebar click handlers
  passDivs.forEach((passDiv, index) => {{
    passDiv.addEventListener('click', () => {{
      switchToPass(index);
    }});
  }});

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {{
    // Ignore if typing in input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch(e.key) {{
      case 'f':
        nextPass();
        e.preventDefault();
        break;
      case 'r':
        prevPass();
        e.preventDefault();
        break;
      case '1':
        if (KEY_PASSES[0] !== null) switchToPass(KEY_PASSES[0]);
        e.preventDefault();
        break;
      case '2':
        if (KEY_PASSES[1] !== null) switchToPass(KEY_PASSES[1]);
        e.preventDefault();
        break;
      case '3':
        if (KEY_PASSES[2] !== null) switchToPass(KEY_PASSES[2]);
        e.preventDefault();
        break;
      case '4':
        if (KEY_PASSES[3] !== null) switchToPass(KEY_PASSES[3]);
        e.preventDefault();
        break;
      case 'c':
        centerOnSelectedBlock();
        e.preventDefault();
        break;
      case 'w':
      case 's':
      case 'a':
      case 'd':
        // TODO: Implement block navigation
        e.preventDefault();
        break;
    }}
  }});

  // Initialize
  switchToPass(0);
}})();
"#,
            self.num_passes, key_passes_str, redundant_passes_str
        )
    }

    fn generate_multi_function(&self, num_functions: usize) -> String {
        format!(
            r#"
(function() {{
  'use strict';

  // Constants
  const NUM_FUNCTIONS = {};

  // State
  let currentFunction = 0;
  let currentPass = 0;
  let zoom = 1.0;
  let translation = {{ x: 0, y: 0 }};
  let selectedBlock = null;
  let isPanning = false;
  let panStart = {{ x: 0, y: 0 }};

  // DOM references
  const functionSelector = document.getElementById('function-selector');
  const viewport = document.querySelector('.ig-viewport');
  const passSidebars = Array.from(document.querySelectorAll('.ig-pass-sidebar'));

  // Get all graphs and passes for current function
  function getCurrentGraphs() {{
    return Array.from(document.querySelectorAll(`.ig-graph[data-function="${{currentFunction}}"]`));
  }}

  function getCurrentPasses() {{
    return Array.from(document.querySelectorAll(`.ig-pass[data-function="${{currentFunction}}"]`));
  }}

  // Function switching
  function switchToFunction(funcIndex) {{
    if (funcIndex < 0 || funcIndex >= NUM_FUNCTIONS) return;
    if (funcIndex === currentFunction) return;

    // Hide all graphs for current function
    getCurrentGraphs().forEach(g => g.classList.add('ig-hidden'));

    // Hide current pass sidebar
    passSidebars[currentFunction].classList.add('ig-hidden');

    // Switch function
    currentFunction = funcIndex;
    currentPass = 0;

    // Show new pass sidebar
    passSidebars[currentFunction].classList.remove('ig-hidden');

    // Show first pass of new function
    const newGraphs = getCurrentGraphs();
    if (newGraphs.length > 0) {{
      newGraphs[0].classList.remove('ig-hidden');
    }}

    // Update pass sidebar active state
    const passes = getCurrentPasses();
    passes.forEach((p, i) => {{
      if (i === 0) {{
        p.classList.add('ig-active');
      }} else {{
        p.classList.remove('ig-active');
      }}
    }});

    updateTransform();
  }}

  // Pass switching
  function switchToPass(passIndex) {{
    const graphs = getCurrentGraphs();
    const passes = getCurrentPasses();

    if (passIndex < 0 || passIndex >= graphs.length) return;
    if (passIndex === currentPass) return;

    // Hide current graph
    graphs[currentPass].classList.add('ig-hidden');
    passes[currentPass].classList.remove('ig-active');

    // Show new graph
    currentPass = passIndex;
    graphs[currentPass].classList.remove('ig-hidden');
    passes[currentPass].classList.add('ig-active');

    updateTransform();
  }}

  function nextPass() {{
    const graphs = getCurrentGraphs();
    if (currentPass + 1 < graphs.length) {{
      switchToPass(currentPass + 1);
    }}
  }}

  function prevPass() {{
    if (currentPass > 0) {{
      switchToPass(currentPass - 1);
    }}
  }}

  // Pan/Zoom transform
  function updateTransform() {{
    const graphs = getCurrentGraphs();
    if (graphs[currentPass]) {{
      graphs[currentPass].style.transform = `translate(${{translation.x}}px, ${{translation.y}}px) scale(${{zoom}})`;
    }}
  }}

  function zoomAt(delta, focalX, focalY) {{
    const rect = viewport.getBoundingClientRect();
    const focal = {{ x: focalX - rect.left, y: focalY - rect.top }};
    const worldFocal = {{
      x: (focal.x - translation.x) / zoom,
      y: (focal.y - translation.y) / zoom
    }};

    const newZoom = Math.max(0.1, Math.min(1.0, zoom * delta));
    zoom = newZoom;

    translation.x = focal.x - worldFocal.x * zoom;
    translation.y = focal.y - worldFocal.y * zoom;

    updateTransform();
  }}

  function pan(dx, dy) {{
    translation.x += dx;
    translation.y += dy;
    updateTransform();
  }}

  // Block selection
  function selectBlock(blockId) {{
    const graphs = getCurrentGraphs();
    const selected = graphs[currentPass].querySelector('.ig-block.ig-selected');
    if (selected) {{
      selected.classList.remove('ig-selected', 'ig-last-selected');
    }}

    const block = graphs[currentPass].querySelector(`[data-block-id="${{blockId}}"]`);
    if (block) {{
      block.classList.add('ig-selected', 'ig-last-selected');
      selectedBlock = blockId;
    }}
  }}

  // Event handlers
  functionSelector.addEventListener('change', (e) => {{
    switchToFunction(parseInt(e.target.value));
  }});

  viewport.addEventListener('wheel', (e) => {{
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {{
      const delta = Math.pow(1.5, -e.deltaY * 0.01);
      zoomAt(delta, e.clientX, e.clientY);
    }} else {{
      pan(-e.deltaX, -e.deltaY);
    }}
  }}, {{ passive: false }});

  viewport.addEventListener('mousedown', (e) => {{
    if (e.button === 0) {{
      isPanning = true;
      panStart = {{ x: e.clientX, y: e.clientY }};
      viewport.style.cursor = 'grabbing';
      e.preventDefault();
    }}
  }});

  document.addEventListener('mousemove', (e) => {{
    if (isPanning) {{
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      pan(dx, dy);
      panStart = {{ x: e.clientX, y: e.clientY }};
    }}
  }});

  document.addEventListener('mouseup', (e) => {{
    if (e.button === 0) {{
      isPanning = false;
      viewport.style.cursor = 'default';
    }}
  }});

  // Block click handlers (delegate for all functions)
  viewport.addEventListener('click', (e) => {{
    const block = e.target.closest('.ig-block');
    if (block) {{
      const blockId = block.getAttribute('data-block-id');
      if (blockId) {{
        selectBlock(blockId);
      }}
    }}
  }});

  // Pass click handlers
  document.addEventListener('click', (e) => {{
    if (e.target.classList.contains('ig-pass')) {{
      const passIndex = parseInt(e.target.getAttribute('data-pass'));
      switchToPass(passIndex);
    }}
  }});

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    switch(e.key) {{
      case 'f':
        nextPass();
        e.preventDefault();
        break;
      case 'r':
        prevPass();
        e.preventDefault();
        break;
      case 'ArrowUp':
        if (currentFunction > 0) switchToFunction(currentFunction - 1);
        e.preventDefault();
        break;
      case 'ArrowDown':
        if (currentFunction < NUM_FUNCTIONS - 1) switchToFunction(currentFunction + 1);
        e.preventDefault();
        break;
    }}
  }});

  // Initialize
  switchToFunction(0);
}})();
"#,
            num_functions
        )
    }
}
