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

### Q2: Language Selection Interface ✅ ANSWERED
**User Response:** Both - JSON default with optional UI override

**Implementation:** Widget will have a language dropdown that syncs with config.language. Users can change language via UI or JSON.

### Q3: Code Persistence ✅ ANSWERED
**User Response:** Saved in the widget configuration (persists in dashboard.json)

**Implementation:** Code will be stored in global state and persisted to dashboard.json using the same pattern as other widgets.

### Q4: Editor Complexity ✅ ANSWERED
**User Response:** Full Monaco Editor (VS Code-like with autocomplete, multi-cursor, etc.)

**Implementation:** Use @monaco-editor/react for full VS Code editing experience with IntelliSense, multi-cursor support, and advanced features.

### Q5: Output Display ✅ ANSWERED
**User Response:** Combined stdout/stderr, execution history, ANSI colors, execution time/status

**Implementation:**
- Combined stdout/stderr with ANSI color parsing (reuse parseAnsiToReact)
- Show execution history with collapsible runs
- Display execution time and exit status for each run
- Include clear history button

### Q6: Supported Languages ✅ ANSWERED
**User Response:** As many as possible, including Clojure

**Implementation:** Support 25+ languages with default command mappings:
- JavaScript, Python, Ruby, PHP, Perl, Lua, Bash, Shell
- Go, Rust, C, C++, Java, Kotlin, Swift
- TypeScript, Deno, Bun
- Clojure, R, PowerShell
- **CRITICAL: Must support custom command override in JSON config for custom languages**
- Command template uses `{file}` placeholder for temp file path

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
