import { FC, useState, useEffect } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import { globalCommandOutputs, parseAnsiToReact } from './helpers';

interface CommandRunnerConfig {
  id: string;
  type: 'commandRunner' | 'command-runner';
  label: string;
  command?: string;
  cwd?: string;
  autoRun?: boolean;
  showOutput?: boolean;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const CommandRunner: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboard } = props;
  const runnerConfig = config as CommandRunnerConfig;
  const widgetId = config.id;
  // Initialize output from global state if available
  const [output, setOutput] = useState(() => globalCommandOutputs.get(widgetId) || '');
  const [isRunning, setIsRunning] = useState(false);
  const [showOutput, setShowOutput] = useState(runnerConfig.showOutput !== false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const copyOutput = () => {
    // Remove ANSI codes for clean copy
    const cleanText = (error || output).replace(/\x1b\[[0-9;]+m/g, '');
    navigator.clipboard.writeText(cleanText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const startCommand = async () => {
    if (!runnerConfig.command || !(window as any).commandAPI) return;

    setIsRunning(true);
    setError(null);
    setOutput('');
    globalCommandOutputs.set(widgetId, ''); // Clear global state too

    try {
      const cwd = runnerConfig.cwd || (dashboard as any)?._projectRoot;
      const result = await (window as any).commandAPI.startStreaming(widgetId, runnerConfig.command, cwd);
      if (!result.success) {
        setError(result.error);
        setIsRunning(false);
      }
    } catch (err: any) {
      setError(err.message);
      setIsRunning(false);
    }
  };

  const stopCommand = async () => {
    if (!(window as any).commandAPI) return;

    try {
      await (window as any).commandAPI.stopStreaming(widgetId);
      setIsRunning(false);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Check if already running on mount
  useEffect(() => {
    const checkRunning = async () => {
      if ((window as any).commandAPI) {
        const result = await (window as any).commandAPI.isRunning(widgetId);
        setIsRunning(result.running);
      }
    };
    checkRunning();
  }, [widgetId]);

  // Sync output with global state when it changes
  useEffect(() => {
    if (widgetId) {
      globalCommandOutputs.set(widgetId, output);
    }
  }, [output, widgetId]);

  // Listen for command output
  useEffect(() => {
    if (!(window as any).commandAPI) return;

    const handleOutput = ({ widgetId: eventWidgetId, output: text }: any) => {
      if (eventWidgetId === widgetId) {
        setOutput(prev => {
          const newOutput = prev + text;
          globalCommandOutputs.set(widgetId, newOutput);
          return newOutput;
        });
      }
    };

    const handleExit = ({ widgetId: eventWidgetId, code }: any) => {
      if (eventWidgetId === widgetId) {
        setIsRunning(false);
        if (code !== 0) {
          setOutput(prev => {
            const newOutput = prev + `\n\n[Process exited with code ${code}]`;
            globalCommandOutputs.set(widgetId, newOutput);
            return newOutput;
          });
        }
      }
    };

    const handleError = ({ widgetId: eventWidgetId, error: errorMsg }: any) => {
      if (eventWidgetId === widgetId) {
        setError(errorMsg);
        setIsRunning(false);
      }
    };

    const outputHandler = (window as any).commandAPI.onOutput(handleOutput);
    const exitHandler = (window as any).commandAPI.onExit(handleExit);
    const errorHandler = (window as any).commandAPI.onError(handleError);

    return () => {
      // Call cleanup functions directly (they already remove the listeners)
      outputHandler();
      exitHandler();
      errorHandler();
    };
  }, [widgetId]);

  // Auto-run if configured
  useEffect(() => {
    if (runnerConfig.autoRun && !isRunning) {
      startCommand();
    }
  }, [runnerConfig.command, runnerConfig.autoRun]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="widget-label" style={{ fontFamily: theme.textBody, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
        <span>{runnerConfig.label}</span>
        <div style={{ display: 'flex', gap: 8 }}>
          {runnerConfig.command && (output || error) && (
            <button
              onClick={copyOutput}
              style={{
                background: copied ? theme.positive : 'rgba(255,255,255,0.1)',
                border: `1px solid ${theme.accent}44`,
                borderRadius: 4,
                padding: '4px 8px',
                color: theme.textColor,
                cursor: 'pointer',
                fontSize: '0.65rem'
              }}
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
          )}
          {runnerConfig.command && (
            <button
              onClick={() => setShowOutput(!showOutput)}
              style={{
                background: 'rgba(255,255,255,0.1)',
                border: `1px solid ${theme.accent}44`,
                borderRadius: 4,
                padding: '4px 8px',
                color: theme.textColor,
                cursor: 'pointer',
                fontSize: '0.65rem'
              }}
            >
              {showOutput ? 'Hide' : 'Show'}
            </button>
          )}
          <button
            onClick={isRunning ? stopCommand : startCommand}
            disabled={!runnerConfig.command}
            style={{
              background: isRunning ? theme.negative : theme.accent,
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
      {!runnerConfig.command && (
        <div style={{ fontFamily: theme.textBody, color: theme.negative, fontSize: '0.8rem', padding: '12px 0' }}>
          No command configured
        </div>
      )}
      {showOutput && (output || error) && (
        <div style={{
          fontFamily: 'monospace',
          fontSize: '0.7rem',
          color: theme.textColor,
          background: 'rgba(0,0,0,0.3)',
          padding: 8,
          borderRadius: 4,
          marginTop: 8,
          flex: 1,
          overflow: 'auto',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          minHeight: 0,
          userSelect: 'text',
          cursor: 'text'
        }}>
          {error ? (
            <span style={{ color: theme.negative }}>{error}</span>
          ) : (
            parseAnsiToReact(output)
          )}
        </div>
      )}
    </div>
  );
};
