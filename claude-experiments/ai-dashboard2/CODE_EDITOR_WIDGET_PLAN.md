# Code Editor Widget - Planning Document

## Widget Overview
**Type:** `codeEditor`
**Purpose:** Allow users to type code, select a language, and execute it with output displayed below

## Key Requirements (from user)
- Text area for typing code
- Ability to run code with custom commands
- Output display below the code input
- Language selection (can be baked into component or selectable)

## Design Questions to Answer

### Q1: Code Execution Method ✅ ANSWERED
**User Response:** Shell commands

**Implementation:** Run as shell commands (e.g., `node script.js`, `python script.py`) - similar to commandRunner

### Q2: Language Selection Interface
**Options:**
- A) Dropdown selector in the widget UI
- B) Only configured via JSON (baked into the widget)
- C) Both - JSON default with optional UI override

**Status:** ⏳ PENDING

### Q3: Code Persistence
**Options:**
- A) Saved in the widget configuration (persists in dashboard.json)
- B) Linked to an external file (edit existing files)
- C) Session-only (lost on refresh)
- D) Multiple options available

**Status:** ⏳ PENDING

### Q4: Editor Complexity
**Options:**
- A) Simple textarea with syntax highlighting
- B) Full Monaco Editor (VS Code-like with autocomplete, multi-cursor, etc.)
- C) Something in between

**Status:** ⏳ PENDING

### Q5: Output Display
**Questions:**
- Should it show stdout/stderr separately or combined?
- Should it support ANSI color codes (like commandRunner)?
- Should it be clearable or show execution history?
- Should it show execution time/status?

**Status:** ⏳ PENDING

### Q6: Supported Languages
**Baseline:** JavaScript/Node.js, Python, Bash/Shell

**Additional to consider:** Ruby, Go, Rust, TypeScript, etc.

**Status:** ⏳ PENDING

---

## Technical Implementation Plan

### Architecture Pattern
Based on **CommandRunner widget** (lines 1024-1229 in `src/App.jsx`)

### Key Components Needed

1. **Widget Component Function**
   - Location: `src/App.jsx`
   - State management: code, language, output, isRunning, error
   - Global state persistence (like CommandRunner)

2. **IPC Communication**
   - Use existing `window.commandAPI.startStreaming(widgetId, command, cwd)`
   - Event listeners for output streaming
   - Handle exit and error events

3. **Widget Registration**
   - Add to `widgetComponents` object
   - Add default dimensions to `defaultWidgetDimensions`

4. **Configuration Format**
   ```json
   {
     "id": "code-editor-1",
     "type": "codeEditor",
     "label": "Code Editor",
     "language": "javascript",
     "command": "node {file}",
     "showLineNumbers": true,
     "autoRun": false,
     "width": 500,
     "height": 400,
     "x": 0,
     "y": 0
   }
   ```

### Language Command Mappings
```javascript
const languageCommands = {
  javascript: 'node {file}',
  python: 'python {file}',
  python3: 'python3 {file}',
  bash: 'bash {file}',
  sh: 'sh {file}',
  ruby: 'ruby {file}',
  go: 'go run {file}',
  rust: 'rustc {file} && ./{basename}',
  // ... extensible
};
```

### Key Features
- ✅ Code input area (textarea or Monaco editor)
- ✅ Language selector (UI and/or config)
- ✅ Run/Stop buttons
- ✅ Output display with syntax highlighting
- ✅ ANSI color support (using existing parseAnsiToReact)
- ✅ State persistence across dashboard switches
- ✅ Error handling and display
- ✅ Execution status indicator (running/idle)
- ⏳ Optional: Copy code/output buttons
- ⏳ Optional: Clear output button
- ⏳ Optional: Execution timer
- ⏳ Optional: Save code to file
- ⏳ Optional: Load code from file

### File Structure
```
/tmp/
  └── dashboard-code-{widgetId}-{timestamp}.{ext}  # Temporary execution files
```

### State Management Pattern
```javascript
// Global state map (outside component)
const globalCodeEditorStates = new Map();

// Inside component
const [code, setCode] = useState(() =>
  globalCodeEditorStates.get(widgetId)?.code || config.defaultCode || ''
);

const [language, setLanguage] = useState(() =>
  globalCodeEditorStates.get(widgetId)?.language || config.language || 'javascript'
);

// Persist on change
useEffect(() => {
  globalCodeEditorStates.set(widgetId, { code, language });
}, [code, language, widgetId]);
```

---

## Implementation Steps

### Phase 1: Basic Implementation
1. Create CodeEditor component function
2. Add textarea for code input
3. Add run button
4. Write code to temp file
5. Execute using commandAPI.startStreaming
6. Display output with ANSI parsing
7. Add to widgetComponents registry
8. Add default dimensions

### Phase 2: Language Support
1. Add language selector UI (if decided)
2. Implement language-to-command mapping
3. Add file extension detection
4. Test with multiple languages

### Phase 3: Enhanced Features
1. Add syntax highlighting to code input
2. Add clear output button
3. Add copy buttons
4. Add execution timer
5. Add stop button
6. Improve error display

### Phase 4: Advanced (Optional)
1. Integrate Monaco Editor (if decided)
2. Add file load/save capabilities
3. Add execution history
4. Add split stdout/stderr
5. Add custom command templates

---

## Dependencies

### Existing (already in project)
- `react` - Component framework
- `react-syntax-highlighter` - Output highlighting
- `ansi-to-react` - ANSI color parsing
- Electron IPC - Command execution

### Potential New (if Monaco chosen)
- `@monaco-editor/react` - Full-featured code editor

---

## Next Steps
1. ⏳ Answer remaining design questions (Q2-Q6)
2. ⏳ Decide on implementation approach
3. ⏳ Begin Phase 1 implementation
4. ⏳ Test with multiple languages
5. ⏳ Iterate based on feedback

---

## Notes
- CommandRunner widget is the perfect reference implementation
- Execution happens via Electron IPC (secure, sandboxed)
- Temp files needed for most language executions
- State persistence ensures code survives dashboard switches
- Theme integration ensures visual consistency
