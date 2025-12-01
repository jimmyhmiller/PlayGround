import { FC, useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import { globalCodeEditorStates, LANGUAGE_CONFIG, parseAnsiToReact } from './helpers';

interface CodeEditorConfig {
  id: string;
  type: 'codeEditor' | 'code-editor';
  label?: string;
  content?: string;
  language?: string;
  command?: string;
  cwd?: string;
  minimap?: boolean;
  showLineNumbers?: boolean;
  readOnly?: boolean;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const CodeEditor: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboard } = props;
  const editorConfig = config as CodeEditorConfig;
  const widgetId = config.id;
  const projectRoot = (dashboard as any)?._projectRoot;

  // Initialize from global state
  const globalState = globalCodeEditorStates.get(widgetId) || {};
  const [code, setCode] = useState(globalState.code || editorConfig.content || '// Write your code here\n');
  const [language, setLanguage] = useState(globalState.language || editorConfig.language || 'javascript');
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState('');
  const [exitCode, setExitCode] = useState<number | null>(null);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [monacoInstance, setMonacoInstance] = useState<any>(null);

  // Sync to global state
  useEffect(() => {
    globalCodeEditorStates.set(widgetId, {
      code,
      language
    });
  }, [code, language, widgetId]);

  // Get command for language (config.command overrides default)
  const getCommand = () => {
    if (editorConfig.command) {
      return editorConfig.command; // Custom command from config takes precedence
    }
    const langConfig = LANGUAGE_CONFIG[language];
    return langConfig ? langConfig.command : `${language} {file}`;
  };

  // Get file extension for language
  const getExtension = () => {
    const langConfig = LANGUAGE_CONFIG[language];
    return langConfig ? langConfig.extension : 'txt';
  };

  // Run code
  const runCode = async () => {
    if (!code.trim() || !(window as any).commandAPI) return;

    setIsRunning(true);
    setOutput('');
    setExitCode(null);
    setExecutionTime(null);
    setStartTime(Date.now());

    try {
      // Create temp file path
      const timestamp = Date.now();
      const extension = getExtension();
      const tempFile = `/tmp/dashboard-code-${widgetId}-${timestamp}.${extension}`;
      const tempDir = '/tmp';

      // Write code to temp file
      const writeResult = await (window as any).dashboardAPI?.writeCodeFile?.(tempFile, code);
      if (!writeResult?.success) {
        throw new Error('Failed to write code to temp file');
      }

      // Prepare command with placeholders replaced
      let command = getCommand()
        .replace(/\{file\}/g, tempFile)
        .replace(/\{tempdir\}/g, tempDir)
        .replace(/\{basename\}/g, `dashboard-code-${widgetId}-${timestamp}`);

      // Execute command
      const cwd = editorConfig.cwd || projectRoot || tempDir;
      const result = await (window as any).commandAPI.startStreaming(widgetId, command, cwd);

      if (!result.success) {
        setOutput(`Error: ${result.error}`);
        setIsRunning(false);
        setExitCode(1);
        const currentTime = Date.now();
        const duration = startTime ? currentTime - startTime : 0;
        setExecutionTime(duration);
      }
    } catch (err: any) {
      setOutput(`Error: ${err.message}`);
      setIsRunning(false);
      setExitCode(1);
      const currentTime = Date.now();
      const duration = startTime ? currentTime - startTime : 0;
      setExecutionTime(duration);
    }
  };

  // Stop execution
  const stopExecution = async () => {
    if (!(window as any).commandAPI) return;
    try {
      await (window as any).commandAPI.stopStreaming(widgetId);
      setIsRunning(false);
    } catch (err) {
      console.error('Failed to stop execution:', err);
    }
  };

  // Listen for command output
  useEffect(() => {
    if (!(window as any).commandAPI) return;

    const handleOutput = ({ widgetId: eventWidgetId, output: text }: any) => {
      if (eventWidgetId === widgetId) {
        setOutput(prev => prev + text);
      }
    };

    const handleExit = ({ widgetId: eventWidgetId, code: code }: any) => {
      if (eventWidgetId === widgetId) {
        setIsRunning(false);
        setExitCode(code);
        const duration = startTime ? Date.now() - startTime : 0;
        setExecutionTime(duration);
      }
    };

    const handleError = ({ widgetId: eventWidgetId, error: errorMsg }: any) => {
      if (eventWidgetId === widgetId) {
        setOutput(prev => prev + `\nError: ${errorMsg}`);
        setIsRunning(false);
        setExitCode(1);
        const duration = startTime ? Date.now() - startTime : 0;
        setExecutionTime(duration);
      }
    };

    const outputHandler = (window as any).commandAPI.onOutput(handleOutput);
    const exitHandler = (window as any).commandAPI.onExit(handleExit);
    const errorHandler = (window as any).commandAPI.onError(handleError);

    return () => {
      (window as any).commandAPI.offOutput(outputHandler);
      (window as any).commandAPI.offExit(exitHandler);
      (window as any).commandAPI.offError(errorHandler);
    };
  }, [widgetId, startTime]);

  // Configure Monaco theme on mount
  useEffect(() => {
    if (monacoInstance) {
      monacoInstance.editor.defineTheme('dashboard-dark', {
        base: 'vs-dark',
        inherit: true,
        rules: [],
        colors: {
          'editor.background': theme.bgApp || '#0d1117',
          'editor.foreground': theme.textColor || '#c9d1d9',
          'editorLineNumber.foreground': theme.textColor ? `${theme.textColor}60` : '#c9d1d960',
          'editor.selectionBackground': theme.accent ? `${theme.accent}40` : '#58a6ff40',
          'editor.inactiveSelectionBackground': theme.accent ? `${theme.accent}20` : '#58a6ff20',
        }
      });
      monacoInstance.editor.setTheme('dashboard-dark');
    }
  }, [monacoInstance, theme]);

  // Get Monaco language mode
  const getMonacoLanguage = () => {
    const langConfig = LANGUAGE_CONFIG[language];
    return langConfig ? langConfig.monacoLang : language;
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header with controls */}
      <div className="widget-label" style={{
        fontFamily: theme.textBody,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexShrink: 0,
        gap: 8
      }}>
        <span>{editorConfig.label || 'Code Editor'}</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Language selector */}
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: `1px solid ${theme.accent}44`,
              borderRadius: 4,
              padding: '4px 8px',
              color: theme.textColor,
              fontSize: '0.7rem',
              cursor: 'pointer'
            }}
          >
            {Object.keys(LANGUAGE_CONFIG).sort().map(lang => (
              <option key={lang} value={lang}>{lang}</option>
            ))}
          </select>

          {/* Run/Stop button */}
          <button
            onClick={isRunning ? stopExecution : runCode}
            style={{
              background: isRunning ? theme.negative : theme.positive,
              border: 'none',
              borderRadius: 4,
              padding: '4px 12px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.7rem',
              fontWeight: 600
            }}
          >
            {isRunning ? 'Stop' : 'Run'}
          </button>
        </div>
      </div>

      {/* Monaco Editor */}
      <div style={{ flex: '1 1 50%', minHeight: 0, overflow: 'hidden' }}>
        <Editor
          height="100%"
          language={getMonacoLanguage()}
          value={code}
          onChange={(value) => setCode(value || '')}
          theme="vs-dark"
          onMount={(editor, monaco) => {
            setMonacoInstance(monaco);

            // Add Command+Enter (Mac) or Ctrl+Enter (Windows/Linux) to run code
            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
              runCode();
            });
          }}
          loading=""
          options={{
            minimap: { enabled: editorConfig.minimap !== false },
            fontSize: 13,
            lineNumbers: editorConfig.showLineNumbers !== false ? 'on' : 'off',
            readOnly: editorConfig.readOnly || false,
            automaticLayout: true,
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            tabSize: 2,
            renderValidationDecorations: 'off',
            quickSuggestions: false,
          }}
        />
      </div>

      {/* Output area */}
      <div style={{
        flex: '1 1 50%',
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        borderTop: `1px solid ${theme.accent}44`,
        marginTop: 8,
        paddingTop: 8,
        overflow: 'hidden'
      }}>
        <div style={{
          fontSize: '0.75rem',
          fontWeight: 600,
          marginBottom: 6,
          color: theme.textColor,
          opacity: 0.8
        }}>
          Output
        </div>

        <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
          {/* Output display */}
          {(isRunning || output) ? (
            <div style={{
              fontFamily: 'monospace',
              fontSize: '0.7rem',
              color: theme.textColor,
              background: 'rgba(0,0,0,0.3)',
              padding: 8,
              borderRadius: 4,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              border: exitCode !== null ? `1px solid ${exitCode === 0 ? theme.positive : theme.negative}44` : 'none'
            }}>
              {/* Status header */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: 4,
                fontSize: '0.65rem',
                opacity: 0.8
              }}>
                <span>
                  {isRunning ? (
                    <span style={{ color: theme.accent }}>▶ Running...</span>
                  ) : exitCode !== null ? (
                    exitCode === 0 ? (
                      <span style={{ color: theme.positive }}>✓ Success</span>
                    ) : (
                      <span style={{ color: theme.negative }}>✗ Exit {exitCode}</span>
                    )
                  ) : null}
                </span>
                {executionTime !== null && (
                  <span>{executionTime}ms</span>
                )}
              </div>
              {/* Output content */}
              {parseAnsiToReact(output || '')}
            </div>
          ) : (
            <div style={{
              color: theme.textColor,
              opacity: 0.5,
              fontSize: '0.75rem',
              textAlign: 'center',
              padding: 20
            }}>
              Click "Run" to execute your code
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
