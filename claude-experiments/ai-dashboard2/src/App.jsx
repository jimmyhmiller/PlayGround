import { useState, useRef, useEffect, useMemo, Suspense, Component } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Grid, GridItem } from './components';
import './styles.css';

// Error Boundary Component
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      const { theme } = this.props;

      return (
        <div style={{
          padding: '20px',
          fontFamily: theme?.textBody || 'sans-serif',
          color: theme?.negative || '#ff4757',
          backgroundColor: theme?.widgetBg || 'rgba(22, 27, 34, 0.85)',
          border: `1px solid ${theme?.negative || '#ff4757'}`,
          borderRadius: theme?.widgetRadius || '8px',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
            ⚠️ Widget Error
          </div>
          <div style={{
            fontSize: '0.9rem',
            opacity: 0.9,
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            overflow: 'auto',
            flex: 1
          }}>
            {this.state.error?.toString()}
            {this.state.errorInfo?.componentStack && (
              <details style={{ marginTop: '10px', fontSize: '0.8rem', opacity: 0.7 }}>
                <summary style={{ cursor: 'pointer' }}>Stack Trace</summary>
                <pre>{this.state.errorInfo.componentStack}</pre>
              </details>
            )}
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
            style={{
              padding: '8px 16px',
              backgroundColor: theme?.accent || '#00d9ff',
              color: theme?.bgApp || '#0d1117',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontFamily: theme?.textBody || 'sans-serif',
              fontSize: '0.9rem',
              fontWeight: 'bold'
            }}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

const icons = {
  botanist: (
    <svg viewBox="0 0 60 60">
      <path d="M10 10 Q30 5 50 10 Q55 30 50 50 Q30 55 10 50 Q5 30 10 10" />
      <path d="M30 50 Q 45 30 30 10 Q 15 30 30 50" />
    </svg>
  ),
  grid: (
    <svg viewBox="0 0 60 60">
      <path d="M10 5 L50 5 L55 15 L55 55 L15 55 L5 45 Z" />
      <rect x="20" y="20" width="20" height="20" />
    </svg>
  ),
  dream: (
    <svg viewBox="0 0 60 60">
      <rect x="5" y="5" width="50" height="50" rx="20" />
      <circle cx="30" cy="30" r="10" />
    </svg>
  ),
  console: (
    <svg viewBox="0 0 60 60">
      <rect x="5" y="10" width="50" height="40" rx="4" />
      <path d="M15 25 L25 32 L15 39" />
      <line x1="30" y1="39" x2="45" y2="39" />
    </svg>
  ),
  square: <svg viewBox="0 0 60 60"><rect x="10" y="10" width="40" height="40" /></svg>,
  circle: <svg viewBox="0 0 60 60"><circle cx="30" cy="30" r="20" /></svg>,
};

// Hook to load widget data from inline or file reference
function useWidgetData(config, reloadTrigger) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // If data is provided inline, use it directly
    if (config.data !== undefined) {
      setData(config.data);
      setLoading(false);
      setError(null);
      return;
    }

    // If dataSource is provided, load from file
    if (config.dataSource) {
      setLoading(true);
      setError(null);

      // Use the dashboard API to load the file
      if (window.dashboardAPI && window.dashboardAPI.loadDataFile) {
        window.dashboardAPI.loadDataFile(config.dataSource)
          .then(loadedData => {
            setData(loadedData);
            setLoading(false);
          })
          .catch(err => {
            console.error('[useWidgetData] Failed to load data from', config.dataSource, err);
            setError(err.message || 'Failed to load data');
            setLoading(false);
          });
      } else {
        // Fallback: try to fetch as a relative URL
        fetch(config.dataSource)
          .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
          })
          .then(loadedData => {
            setData(loadedData);
            setLoading(false);
          })
          .catch(err => {
            console.error('[useWidgetData] Failed to fetch data from', config.dataSource, err);
            setError(err.message || 'Failed to load data');
            setLoading(false);
          });
      }
    }
  }, [config.data, config.dataSource, reloadTrigger]);

  return { data, loading, error };
}

// Widget Components
function BarChart({ theme, config, reloadTrigger }) {
  const { data, loading, error } = useWidgetData(config, reloadTrigger);

  // Generate random data as fallback
  const randomBars = useMemo(() => Array.from({ length: 30 }, () => Math.floor(Math.random() * 80 + 20)), []);

  // Use loaded data if available, otherwise fall back to random data
  const bars = useMemo(() => {
    if (data) {
      // Support different data formats:
      // 1. Array of numbers: [20, 45, 60, ...]
      // 2. Array of objects: [{value: 20}, {value: 45}, ...]
      // 3. Object with values array: {values: [20, 45, 60, ...]}
      if (Array.isArray(data)) {
        return data.map(item => typeof item === 'number' ? item : item.value);
      } else if (data.values && Array.isArray(data.values)) {
        return data.values;
      }
    }
    return randomBars;
  }, [data, randomBars]);

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        <span>{Array.isArray(config.label) ? config.label[0] : config.label}</span>
        {Array.isArray(config.label) && config.label[1] && <span>{config.label[1]}</span>}
      </div>
      {loading && (
        <div style={{ fontFamily: theme.textBody, color: theme.textColor, padding: '20px' }}>
          Loading data...
        </div>
      )}
      {error && (
        <div style={{ fontFamily: theme.textBody, color: theme.negative, padding: '20px' }}>
          Error: {error}
        </div>
      )}
      {!loading && !error && (
        <div className="chart-container">
          {bars.map((h, i) => (
            <div key={i} className="bar" style={{ height: `${h}%`, backgroundColor: theme.accent, borderRadius: theme.chartRadius }} />
          ))}
        </div>
      )}
    </>
  );
}

function DiffList({ theme, config }) {
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      {config.items.map(([file, added, removed], i) => (
        <div key={i} className="diff-item" style={{ fontFamily: theme.textBody, color: theme.textColor }}>
          <span className="diff-file">{file}</span>
          <span className="diff-stats">
            <span style={{ color: theme.positive }}>+{added}</span>
            <span style={{ color: theme.negative }}>-{removed}</span>
          </span>
        </div>
      ))}
    </>
  );
}

function TestResults({ theme, config, dashboardId }) {
  const [hiddenStatuses, setHiddenStatuses] = useState(new Set());
  const [isRegenerating, setIsRegenerating] = useState(false);

  const toggleStatus = (status) => {
    setHiddenStatuses(prev => {
      const next = new Set(prev);
      if (next.has(status)) {
        next.delete(status);
      } else {
        next.add(status);
      }
      return next;
    });
  };

  const handleRegenerate = async () => {
    if (!window.dashboardAPI || !dashboardId || !config.id) return;

    setIsRegenerating(true);
    try {
      const result = await window.dashboardAPI.regenerateWidget(dashboardId, config.id);
      if (!result.success) {
        console.error('Regenerate failed:', result.error);
      }
    } catch (error) {
      console.error('Regenerate error:', error);
    } finally {
      setIsRegenerating(false);
    }
  };

  const hasRegenerate = config.regenerateCommand || config.regenerateScript || config.regenerate || config.testRunner;

  // Validation
  const validateTests = () => {
    // If using a built-in test runner, tests array is optional (will be populated on run)
    if (!config.tests && !config.testRunner) {
      return {
        valid: false,
        message: 'Missing "tests" array',
        expected: 'The testResults widget requires a "tests" array property.\n\nExpected format:\n{\n  "type": "testResults",\n  "label": "Test Suite Results",\n  "tests": [\n    {\n      "name": "Test name",\n      "status": "passed" | "failed" | "skipped",\n      "duration": 123,  // optional, in milliseconds\n      "error": "Error message"  // optional, for failed tests\n    }\n  ]\n}\n\nAlternatively, use a built-in test runner:\n{\n  "type": "testResults",\n  "label": "Test Suite",\n  "testRunner": "cargo" | "jest" | "pytest",\n  "autoRun": false\n}'
      };
    }

    // If no tests yet (using testRunner), validation passes
    if (!config.tests) {
      return { valid: true };
    }

    if (!Array.isArray(config.tests)) {
      return {
        valid: false,
        message: '"tests" must be an array',
        expected: `Expected "tests" to be an array, but got ${typeof config.tests}.\n\nExpected format:\n"tests": [\n  {\n    "name": "Test name",\n    "status": "passed",\n    "duration": 123\n  }\n]`
      };
    }

    // Validate each test
    for (let i = 0; i < config.tests.length; i++) {
      const test = config.tests[i];

      if (!test || typeof test !== 'object') {
        return {
          valid: false,
          message: `Test at index ${i} is not an object`,
          expected: `Each test must be an object with "name" and "status" properties.\n\nExpected format for test:\n{\n  "name": "Test name",\n  "status": "passed" | "failed" | "skipped",\n  "duration": 123,  // optional\n  "error": "Error message"  // optional\n}`
        };
      }

      if (!test.name || typeof test.name !== 'string') {
        return {
          valid: false,
          message: `Test at index ${i} missing or invalid "name" property`,
          expected: `Each test must have a "name" property (string).\n\nCurrent test: ${JSON.stringify(test, null, 2)}\n\nExpected: { "name": "Test name", "status": "passed", ... }`
        };
      }

      if (!test.status || !['passed', 'failed', 'skipped'].includes(test.status)) {
        return {
          valid: false,
          message: `Test "${test.name}" has invalid "status" property`,
          expected: `The "status" property must be one of: "passed", "failed", or "skipped".\n\nCurrent value: "${test.status}"\n\nExpected format:\n{\n  "name": "${test.name}",\n  "status": "passed" | "failed" | "skipped"\n}`
        };
      }

      if (test.duration !== undefined && typeof test.duration !== 'number') {
        return {
          valid: false,
          message: `Test "${test.name}" has invalid "duration" property`,
          expected: `The "duration" property must be a number (milliseconds).\n\nCurrent value: ${JSON.stringify(test.duration)} (${typeof test.duration})\n\nExpected format:\n{\n  "name": "${test.name}",\n  "status": "${test.status}",\n  "duration": 123\n}`
        };
      }

      if (test.error !== undefined && typeof test.error !== 'string') {
        return {
          valid: false,
          message: `Test "${test.name}" has invalid "error" property`,
          expected: `The "error" property must be a string.\n\nCurrent value: ${JSON.stringify(test.error)} (${typeof test.error})\n\nExpected format:\n{\n  "name": "${test.name}",\n  "status": "failed",\n  "error": "Error message here"\n}`
        };
      }
    }

    return { valid: true };
  };

  const validation = validateTests();

  if (!validation.valid) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label || 'Test Results'}</div>
        <div style={{
          fontFamily: theme.textBody,
          fontSize: '0.8rem',
          padding: '16px',
          color: theme.negative,
          backgroundColor: `${theme.negative}15`,
          border: `1px solid ${theme.negative}40`,
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>
            ⚠️ Invalid configuration: {validation.message}
          </div>
          <div style={{
            fontFamily: theme.textCode || 'monospace',
            fontSize: '0.7rem',
            backgroundColor: `${theme.negative}10`,
            padding: '12px',
            borderRadius: '6px',
            whiteSpace: 'pre-wrap',
            color: theme.textColor,
            lineHeight: 1.5,
            userSelect: 'text',
            cursor: 'text'
          }}>
            {validation.expected}
          </div>
          <div style={{
            fontSize: '0.7rem',
            opacity: 0.8,
            borderTop: `1px solid ${theme.negative}20`,
            paddingTop: '8px',
            marginTop: '4px'
          }}>
            💡 Tip: Copy the expected format above and provide it to an AI to generate the correct configuration.
          </div>
        </div>
      </>
    );
  }

  const tests = config.tests || [];
  const passedCount = tests.filter(t => t.status === 'passed').length;
  const failedCount = tests.filter(t => t.status === 'failed').length;
  const skippedCount = tests.filter(t => t.status === 'skipped').length;
  const totalCount = tests.length;

  return (
    <>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: hasRegenerate ? '8px' : '0'
      }}>
        <div className="widget-label" style={{ fontFamily: theme.textBody, marginBottom: 0 }}>
          {config.label || 'Test Results'}
        </div>
        {hasRegenerate && (
          <button
            onClick={handleRegenerate}
            disabled={isRegenerating}
            style={{
              background: 'transparent',
              border: `1px solid ${theme.accent}40`,
              borderRadius: '4px',
              color: theme.accent,
              cursor: isRegenerating ? 'wait' : 'pointer',
              padding: '4px 8px',
              fontSize: '0.75rem',
              fontFamily: theme.textBody,
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              opacity: isRegenerating ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              if (!isRegenerating) {
                e.target.style.backgroundColor = `${theme.accent}15`;
                e.target.style.borderColor = `${theme.accent}80`;
              }
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'transparent';
              e.target.style.borderColor = `${theme.accent}40`;
            }}
          >
            <span style={{ fontSize: '0.9rem' }}>{isRegenerating ? '⟳' : '▶'}</span>
            {isRegenerating ? 'Running...' : 'Run Tests'}
          </button>
        )}
      </div>

      {/* Summary */}
      <div style={{
        fontFamily: theme.textBody,
        fontSize: '0.85rem',
        padding: '12px 0',
        borderBottom: `1px solid ${theme.accent}22`,
        marginBottom: '12px',
        display: 'flex',
        gap: '16px',
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: theme.textColor, opacity: 0.7 }}>Total:</span>
          <span style={{ color: theme.accent, fontWeight: 600 }}>{totalCount}</span>
        </div>
        {passedCount > 0 && (
          <div
            onClick={() => toggleStatus('passed')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              padding: '2px 6px',
              margin: '-2px -6px',
              borderRadius: '4px',
              backgroundColor: hiddenStatuses.has('passed') ? `${theme.positive}10` : 'transparent',
              opacity: hiddenStatuses.has('passed') ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none'
            }}
          >
            <span style={{ color: theme.positive }}>✓</span>
            <span style={{ color: theme.positive, fontWeight: 600 }}>{passedCount} passed</span>
          </div>
        )}
        {failedCount > 0 && (
          <div
            onClick={() => toggleStatus('failed')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              padding: '2px 6px',
              margin: '-2px -6px',
              borderRadius: '4px',
              backgroundColor: hiddenStatuses.has('failed') ? `${theme.negative}10` : 'transparent',
              opacity: hiddenStatuses.has('failed') ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none'
            }}
          >
            <span style={{ color: theme.negative }}>✗</span>
            <span style={{ color: theme.negative, fontWeight: 600 }}>{failedCount} failed</span>
          </div>
        )}
        {skippedCount > 0 && (
          <div
            onClick={() => toggleStatus('skipped')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              padding: '2px 6px',
              margin: '-2px -6px',
              borderRadius: '4px',
              backgroundColor: hiddenStatuses.has('skipped') ? `${theme.textColor}10` : 'transparent',
              opacity: hiddenStatuses.has('skipped') ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none'
            }}
          >
            <span style={{ color: theme.textColor, opacity: 0.5 }}>○</span>
            <span style={{ color: theme.textColor, opacity: 0.7 }}>{skippedCount} skipped</span>
          </div>
        )}
      </div>

      {/* Test List */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        overflow: 'auto',
        maxHeight: '100%'
      }}>
        {tests.length === 0 && config.testRunner ? (
          <div style={{
            fontFamily: theme.textBody,
            fontSize: '0.85rem',
            padding: '24px',
            textAlign: 'center',
            color: theme.textColor,
            opacity: 0.6
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📋</div>
            <div>No test results yet</div>
            <div style={{ fontSize: '0.75rem', marginTop: '8px' }}>
              Click "Run Tests" to execute {config.testRunner} tests
            </div>
          </div>
        ) : tests.length === 0 ? (
          <div style={{
            fontFamily: theme.textBody,
            fontSize: '0.85rem',
            padding: '24px',
            textAlign: 'center',
            color: theme.textColor,
            opacity: 0.6
          }}>
            No tests configured
          </div>
        ) : null}
        {tests.filter(test => !hiddenStatuses.has(test.status)).map((test, i) => (
          <div
            key={i}
            style={{
              fontFamily: theme.textBody,
              fontSize: '0.75rem',
              padding: '8px 12px',
              borderRadius: '6px',
              backgroundColor: test.status === 'passed'
                ? `${theme.positive}15`
                : test.status === 'failed'
                ? `${theme.negative}15`
                : `${theme.textColor}08`,
              border: `1px solid ${
                test.status === 'passed'
                  ? `${theme.positive}40`
                  : test.status === 'failed'
                  ? `${theme.negative}40`
                  : `${theme.textColor}20`
              }`,
              display: 'flex',
              flexDirection: 'column',
              gap: '4px'
            }}
          >
            {/* Test name with status icon */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
              <span style={{
                color: test.status === 'passed'
                  ? theme.positive
                  : test.status === 'failed'
                  ? theme.negative
                  : theme.textColor,
                opacity: test.status === 'skipped' ? 0.5 : 1,
                fontSize: '0.85rem',
                flexShrink: 0
              }}>
                {test.status === 'passed' ? '✓' : test.status === 'failed' ? '✗' : '○'}
              </span>
              <div style={{ flex: 1 }}>
                <div style={{
                  color: theme.textColor,
                  fontWeight: 500,
                  wordBreak: 'break-word'
                }}>
                  {test.name}
                </div>
                {test.duration && (
                  <div style={{
                    color: theme.textColor,
                    opacity: 0.5,
                    fontSize: '0.7rem',
                    marginTop: '2px'
                  }}>
                    {test.duration}ms
                  </div>
                )}
              </div>
            </div>

            {/* Error message for failed tests */}
            {test.status === 'failed' && test.error && (
              <div style={{
                marginTop: '4px',
                padding: '6px 8px',
                backgroundColor: `${theme.negative}10`,
                borderLeft: `2px solid ${theme.negative}`,
                borderRadius: '4px',
                color: theme.negative,
                fontSize: '0.7rem',
                fontFamily: theme.textCode || 'monospace',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word'
              }}>
                {test.error}
              </div>
            )}
          </div>
        ))}
      </div>
    </>
  );
}

function Stat({ theme, config }) {
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div className="big-stat" style={{ fontFamily: theme.textHead, color: theme.textColor }}>{config.value}</div>
    </>
  );
}

function Progress({ theme, config }) {
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div style={{ marginTop: 10, fontFamily: theme.textBody, fontSize: '0.9rem', color: theme.textColor }}>{config.text}</div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${config.value}%`, backgroundColor: theme.accent }} />
      </div>
    </>
  );
}

function FileList({ theme, config }) {
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div className="file-list">
        {config.files.map((file, i) => (
          <div key={i} className="file-item" style={{ fontFamily: theme.textBody, color: theme.textColor }}>
            <span className="file-name">{file.name}</span>
            <span className="file-status" style={{ color: file.status === 'created' ? theme.positive : theme.accent }}>
              {file.status}
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

function TodoList({ theme, config }) {
  const items = config.items || [];
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div className="todo-list">
        {items.map((item, i) => (
          <div key={i} className="todo-item" style={{ fontFamily: theme.textBody, color: theme.textColor }}>
            <span className="todo-check" style={{ color: item.done ? theme.positive : theme.accent }}>
              {item.done ? '✓' : '○'}
            </span>
            <span className={item.done ? 'todo-done' : ''}>{item.text}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function ClaudeTodoList({ theme, config, dashboardId, widgetConversations }) {
  const [todos, setTodos] = useState([]);

  // Resolve conversation ID from linked chat widget
  // Support both chatWidgetKey (full key) and chatWidgetId (partial, needs dashboardId)
  let conversationId = null;

  if (config.chatWidgetKey) {
    // Use full key directly
    conversationId = widgetConversations[config.chatWidgetKey];
  } else if (config.chatWidgetId) {
    // First try using chatWidgetId directly as a full key
    conversationId = widgetConversations[config.chatWidgetId];

    // If not found, try with dashboard prefix
    if (!conversationId) {
      const widgetKey = `${dashboardId}-${config.chatWidgetId}`;
      conversationId = widgetConversations[widgetKey];
    }

    // If still not found, search all widget conversations for a key ending with the chatWidgetId
    if (!conversationId) {
      const matchingKey = Object.keys(widgetConversations).find(key =>
        key.endsWith(config.chatWidgetId) || key.endsWith(`-${config.chatWidgetId}`)
      );
      if (matchingKey) {
        conversationId = widgetConversations[matchingKey];
        console.log('[ClaudeTodoList] Found matching key via search:', matchingKey);
      }
    }
  }

  console.log('[ClaudeTodoList] Dashboard ID:', dashboardId);
  console.log('[ClaudeTodoList] Config chatWidgetId:', config.chatWidgetId);
  console.log('[ClaudeTodoList] Config chatWidgetKey:', config.chatWidgetKey);
  console.log('[ClaudeTodoList] All widget conversation keys:', Object.keys(widgetConversations));
  console.log('[ClaudeTodoList] Resolved conversation ID:', conversationId);

  // Load initial todos and listen for updates
  useEffect(() => {
    console.log('[ClaudeTodoList] useEffect running!', { conversationId, hasAPI: !!window.claudeAPI });

    if (!conversationId || !window.claudeAPI) {
      console.log('[ClaudeTodoList] Missing conversationId or claudeAPI:', { conversationId, hasAPI: !!window.claudeAPI });
      return;
    }

    console.log('[ClaudeTodoList] Setting up listeners for conversation:', conversationId);

    // Load initial todos
    window.claudeAPI.getTodos(conversationId).then(result => {
      console.log('[ClaudeTodoList] Initial todos loaded:', result);
      if (result.success) {
        setTodos(result.todos || []);
      }
    });

    // Listen for real-time updates
    const handler = window.claudeAPI.onTodoUpdate((data) => {
      console.log('[ClaudeTodoList] Received todo update:', data);
      console.log('[ClaudeTodoList] Comparing chatId:', data.chatId, 'with conversationId:', conversationId);
      if (data.chatId === conversationId) {
        console.log('[ClaudeTodoList] Match! Updating todos:', data.todos);
        setTodos(data.todos || []);
      } else {
        console.log('[ClaudeTodoList] No match, ignoring update');
      }
    });

    return () => {
      console.log('[ClaudeTodoList] Cleaning up listeners');
      window.claudeAPI.offTodoUpdate(handler);
    };
  }, [conversationId]);

  // Helper to get status icon and color
  const getStatusDisplay = (status) => {
    switch (status) {
      case 'completed':
        return { icon: '✓', color: theme.positive };
      case 'in_progress':
        return { icon: '⋯', color: theme.accent };
      default: // pending
        return { icon: '○', color: theme.textColor };
    }
  };

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {config.label || 'Agent Tasks'}
      </div>
      <div className="todo-list">
        {todos.length === 0 ? (
          <div style={{
            fontFamily: theme.textBody,
            color: theme.textColor,
            opacity: 0.5,
            fontSize: '0.85rem',
            padding: '8px 0'
          }}>
            No active tasks
          </div>
        ) : (
          todos.map((todo, i) => {
            const { icon, color } = getStatusDisplay(todo.status);
            const displayText = todo.status === 'in_progress' && todo.activeForm
              ? todo.activeForm
              : todo.content;

            return (
              <div key={i} className="todo-item" style={{ fontFamily: theme.textBody, color: theme.textColor }}>
                <span className="todo-check" style={{ color }}>
                  {icon}
                </span>
                <span className={todo.status === 'completed' ? 'todo-done' : ''}>
                  {displayText}
                </span>
              </div>
            );
          })
        )}
      </div>
    </>
  );
}

function KeyValue({ theme, config }) {
  const items = Array.isArray(config.items) ? config.items : [];

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div className="kv-list">
        {items.length > 0 ? (
          items.map((item, i) => (
            <div key={i} className="kv-item" style={{ fontFamily: theme.textBody }}>
              <span className="kv-key" style={{ color: theme.textColor }}>{item.key}</span>
              <span className="kv-value" style={{ color: theme.accent }}>{item.value}</span>
            </div>
          ))
        ) : (
          <div style={{ fontFamily: theme.textBody, color: theme.neutral, fontSize: '0.8rem', padding: '12px 0' }}>
            No items
          </div>
        )}
      </div>
    </>
  );
}

function JsonViewer({ theme, config }) {
  const [collapsed, setCollapsed] = useState(new Set());

  const data = config.data || {};

  const toggleCollapse = (path) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const renderValue = (value, path = '', depth = 0) => {
    const indent = depth * 16;

    if (value === null) {
      return <span style={{ color: theme.accent, opacity: 0.6 }}>null</span>;
    }

    if (value === undefined) {
      return <span style={{ color: theme.accent, opacity: 0.6 }}>undefined</span>;
    }

    if (typeof value === 'boolean') {
      return <span style={{ color: theme.accent }}>{value.toString()}</span>;
    }

    if (typeof value === 'number') {
      return <span style={{ color: theme.positive }}>{value}</span>;
    }

    if (typeof value === 'string') {
      return <span style={{ color: theme.textColor }}>&quot;{value}&quot;</span>;
    }

    if (Array.isArray(value)) {
      const isCollapsed = collapsed.has(path);
      const isEmpty = value.length === 0;

      return (
        <div>
          <span
            onClick={() => !isEmpty && toggleCollapse(path)}
            style={{
              cursor: isEmpty ? 'default' : 'pointer',
              userSelect: 'none',
              color: theme.textColor,
              opacity: 0.7
            }}
          >
            {!isEmpty && (isCollapsed ? '▶ ' : '▼ ')}
            [
            {isEmpty && ']'}
            {!isEmpty && isCollapsed && ` ... ${value.length} items ]`}
          </span>
          {!isEmpty && !isCollapsed && (
            <div style={{ paddingLeft: 16 }}>
              {value.map((item, idx) => (
                <div key={idx} style={{ marginBottom: 4, fontFamily: theme.textBody }}>
                  <span style={{ color: theme.accent, opacity: 0.5 }}>{idx}: </span>
                  {renderValue(item, `${path}[${idx}]`, depth + 1)}
                  {idx < value.length - 1 && <span style={{ color: theme.textColor, opacity: 0.5 }}>,</span>}
                </div>
              ))}
              <span style={{ color: theme.textColor, opacity: 0.7 }}>]</span>
            </div>
          )}
        </div>
      );
    }

    if (typeof value === 'object') {
      const isCollapsed = collapsed.has(path);
      const keys = Object.keys(value);
      const isEmpty = keys.length === 0;

      return (
        <div>
          <span
            onClick={() => !isEmpty && toggleCollapse(path)}
            style={{
              cursor: isEmpty ? 'default' : 'pointer',
              userSelect: 'none',
              color: theme.textColor,
              opacity: 0.7
            }}
          >
            {!isEmpty && (isCollapsed ? '▶ ' : '▼ ')}
            {'{'}
            {isEmpty && '}'}
            {!isEmpty && isCollapsed && ` ... ${keys.length} keys }`}
          </span>
          {!isEmpty && !isCollapsed && (
            <div style={{ paddingLeft: 16 }}>
              {keys.map((key, idx) => (
                <div key={key} style={{ marginBottom: 4, fontFamily: theme.textBody }}>
                  <span style={{ color: theme.accent }}>&quot;{key}&quot;</span>
                  <span style={{ color: theme.textColor, opacity: 0.5 }}>: </span>
                  {renderValue(value[key], `${path}.${key}`, depth + 1)}
                  {idx < keys.length - 1 && <span style={{ color: theme.textColor, opacity: 0.5 }}>,</span>}
                </div>
              ))}
              <span style={{ color: theme.textColor, opacity: 0.7 }}>{'}'}</span>
            </div>
          )}
        </div>
      );
    }

    return <span style={{ color: theme.textColor }}>{String(value)}</span>;
  };

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label || 'JSON'}</div>
      <div style={{
        fontFamily: 'Monaco, Menlo, "Courier New", monospace',
        fontSize: '0.75rem',
        padding: '8px 12px',
        overflow: 'auto',
        color: theme.textColor,
        lineHeight: 1.6
      }}>
        {renderValue(data)}
      </div>
    </>
  );
}

function Markdown({ theme, config, dashboard }) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Extract only the project root to avoid re-rendering on every dashboard change
  const projectRoot = dashboard?._projectRoot;

  useEffect(() => {
    // If markdown content is provided inline, use it directly
    if (config.content !== undefined) {
      setContent(config.content);
      setLoading(false);
      setError(null);
      return;
    }

    // If filePath is provided, load from file
    if (config.filePath) {
      setLoading(true);
      setError(null);

      // Resolve relative paths based on project root
      let fullPath = config.filePath;
      if (!fullPath.startsWith('/') && projectRoot) {
        fullPath = `${projectRoot}/${config.filePath}`;
      }

      // Use the dashboard API to load the file as text
      if (window.dashboardAPI && window.dashboardAPI.loadTextFile) {
        console.log('[Markdown] Loading file via dashboardAPI:', fullPath);
        window.dashboardAPI.loadTextFile(fullPath)
          .then(text => {
            console.log('[Markdown] Loaded text, length:', text.length);
            setContent(text);
            setLoading(false);
          })
          .catch(err => {
            console.error('[Markdown] Failed to load file from', fullPath, err);
            setError(err.message || 'Failed to load file');
            setLoading(false);
          });
      } else {
        console.error('[Markdown] dashboardAPI.loadTextFile not available, this will not work correctly');
        setError('File loading API not available');
        setLoading(false);
      }
    }
  }, [config.content, config.filePath, projectRoot]);

  if (loading) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
        <div style={{
          fontFamily: theme.textBody,
          color: theme.textColor,
          opacity: 0.6,
          padding: '20px',
          textAlign: 'center'
        }}>
          Loading...
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
        <div style={{
          fontFamily: theme.textBody,
          color: theme.negative,
          padding: '20px'
        }}>
          ⚠️ Error: {error}
        </div>
      </>
    );
  }

  if (!content) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
        <div style={{
          fontFamily: theme.textBody,
          color: theme.textColor,
          opacity: 0.6,
          padding: '20px',
          textAlign: 'center'
        }}>
          No content provided. Add "content" or "filePath" to widget config.
        </div>
      </>
    );
  }

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div style={{
        flex: 1,
        overflow: 'auto',
        padding: '12px',
        fontFamily: theme.textBody,
        color: theme.textColor,
        lineHeight: 1.6,
        fontSize: '0.9rem'
      }}>
        <ReactMarkdown
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code
                  style={{
                    background: 'rgba(0,0,0,0.3)',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    fontFamily: 'monospace',
                    fontSize: '0.85em'
                  }}
                  className={className}
                  {...props}
                >
                  {children}
                </code>
              );
            },
            h1: ({ children }) => (
              <h1 style={{ color: theme.accent, marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</h1>
            ),
            h2: ({ children }) => (
              <h2 style={{ color: theme.accent, marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</h2>
            ),
            h3: ({ children }) => (
              <h3 style={{ color: theme.accent, marginTop: '0.5em', marginBottom: '0.5em' }}>{children}</h3>
            ),
            a: ({ children, href }) => (
              <a href={href} style={{ color: theme.accent }} target="_blank" rel="noopener noreferrer">{children}</a>
            ),
            ul: ({ children }) => (
              <ul style={{ paddingLeft: '1.5em' }}>{children}</ul>
            ),
            ol: ({ children }) => (
              <ol style={{ paddingLeft: '1.5em' }}>{children}</ol>
            ),
            blockquote: ({ children }) => (
              <blockquote style={{
                borderLeft: `3px solid ${theme.accent}`,
                paddingLeft: '1em',
                marginLeft: 0,
                opacity: 0.8
              }}>
                {children}
              </blockquote>
            ),
            pre: ({ children }) => (
              <pre style={{
                background: 'rgba(0,0,0,0.3)',
                padding: '12px',
                borderRadius: '6px',
                overflow: 'auto'
              }}>
                {children}
              </pre>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </>
  );
}

function ErrorTestWidget({ theme, config }) {
  const [shouldError, setShouldError] = useState(false);

  if (shouldError) {
    throw new Error('This is a test error thrown by the button!');
  }

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {config.label || 'Error Test'}
      </div>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '20px',
        padding: '40px',
        fontFamily: theme.textBody,
        color: theme.textColor,
        height: '100%'
      }}>
        <div style={{ fontSize: '3rem' }}>💣</div>
        <div style={{ textAlign: 'center', opacity: 0.8 }}>
          Click the button below to throw an error and test the error boundary
        </div>
        <button
          onClick={() => setShouldError(true)}
          style={{
            padding: '12px 24px',
            backgroundColor: theme.negative || '#ff4757',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontFamily: theme.textBody,
            fontSize: '1rem',
            fontWeight: 'bold',
            transition: 'transform 0.1s',
          }}
          onMouseEnter={(e) => e.target.style.transform = 'scale(1.05)'}
          onMouseLeave={(e) => e.target.style.transform = 'scale(1)'}
        >
          💥 Throw Error
        </button>
      </div>
    </>
  );
}

function LayoutSettings({ theme, config, dashboardId, layout }) {
  const [widgetGap, setWidgetGap] = useState(layout?.widgetGap ?? 10);
  const [buffer, setBuffer] = useState(layout?.buffer ?? 20);
  const [layoutMode, setLayoutMode] = useState(layout?.mode || 'single-pane');


  // Update local state when layout prop changes
  useEffect(() => {
    if (layout?.mode && layout.mode !== layoutMode) {
      setLayoutMode(layout.mode);
    }
  }, [layout?.mode]);

  const handleGapChange = (e) => {
    const value = parseInt(e.target.value);
    setWidgetGap(value);
    if (window.dashboardAPI && dashboardId) {
      window.dashboardAPI.updateLayoutSettings(dashboardId, { widgetGap: value });
    }
  };

  const handleBufferChange = (e) => {
    const value = parseInt(e.target.value);
    setBuffer(value);
    if (window.dashboardAPI && dashboardId) {
      window.dashboardAPI.updateLayoutSettings(dashboardId, { buffer: value });
    }
  };

  const handleModeChange = (mode) => {
    setLayoutMode(mode);
    if (window.dashboardAPI && dashboardId) {
      window.dashboardAPI.updateLayoutSettings(dashboardId, { mode });
    } else {
      console.error('[LayoutSettings] dashboardAPI not available or no dashboardId');
    }
  };

  const modes = [
    { value: 'single-pane', label: '📄 Single Pane', description: 'Fixed viewport, no scrolling' },
    { value: 'infinite-canvas', label: '🎨 Infinite Canvas', description: 'Hold Option/Alt to pan around' },
    { value: 'vertical-scroll', label: '↕️ Vertical Scroll', description: 'Scroll up and down' },
    { value: 'horizontal-scroll', label: '↔️ Horizontal Scroll', description: 'Scroll left and right' }
  ];

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>Layout Settings</div>
      <div style={{ fontFamily: theme.textBody, fontSize: '0.85rem', color: theme.textColor, padding: '12px' }}>
        {/* Layout Mode Selector */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ marginBottom: 8, fontWeight: 'bold', color: theme.accent }}>Layout Mode</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {modes.map(mode => (
              <div
                key={mode.value}
                onClick={() => handleModeChange(mode.value)}
                style={{
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: `2px solid ${layoutMode === mode.value ? theme.accent : 'rgba(255,255,255,0.1)'}`,
                  backgroundColor: layoutMode === mode.value ? `${theme.accent}22` : 'rgba(255,255,255,0.05)',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  if (layoutMode !== mode.value) {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.08)';
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (layoutMode !== mode.value) {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
                  }
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: 2 }}>{mode.label}</div>
                <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>{mode.description}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Widget Gap Slider */}
        <div style={{ marginBottom: 15 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
            <label>Widget Gap</label>
            <span style={{ color: theme.accent }}>{widgetGap}px</span>
          </div>
          <input
            type="range"
            min="0"
            max="40"
            value={widgetGap}
            onChange={handleGapChange}
            style={{ width: '100%', accentColor: theme.accent }}
          />
        </div>

        {/* Buffer Slider */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
            <label>Min Size Buffer</label>
            <span style={{ color: theme.accent }}>{buffer}px</span>
          </div>
          <input
            type="range"
            min="0"
            max="50"
            value={buffer}
            onChange={handleBufferChange}
            style={{ width: '100%', accentColor: theme.accent }}
          />
        </div>
      </div>
    </>
  );
}

// Helper to convert ANSI escape codes to React elements
function parseAnsiToReact(text) {
  const ansiRegex = /\x1b\[([0-9;]+)m/g;
  const elements = [];
  let lastIndex = 0;
  let currentStyle = {};

  const colorMap = {
    30: '#000', 31: '#e74c3c', 32: '#2ecc71', 33: '#f39c12',
    34: '#3498db', 35: '#9b59b6', 36: '#1abc9c', 37: '#ecf0f1',
    90: '#7f8c8d', 91: '#ff6b6b', 92: '#51cf66', 93: '#ffd43b',
    94: '#4dabf7', 95: '#da77f2', 96: '#3bc9db', 97: '#f8f9fa'
  };

  const applyCode = (code, style) => {
    const num = parseInt(code);
    if (num === 0) return {}; // Reset
    if (num === 1) return { ...style, fontWeight: 'bold' };
    if (num === 3) return { ...style, fontStyle: 'italic' };
    if (num === 4) return { ...style, textDecoration: 'underline' };
    if (colorMap[num]) return { ...style, color: colorMap[num] };
    return style;
  };

  let match;
  while ((match = ansiRegex.exec(text)) !== null) {
    // Add text before this code
    if (match.index > lastIndex) {
      const textContent = text.substring(lastIndex, match.index);
      elements.push(
        <span key={elements.length} style={currentStyle}>
          {textContent}
        </span>
      );
    }

    // Apply the ANSI code(s)
    const codes = match[1].split(';');
    for (const code of codes) {
      currentStyle = applyCode(code, currentStyle);
    }

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    elements.push(
      <span key={elements.length} style={currentStyle}>
        {text.substring(lastIndex)}
      </span>
    );
  }

  return elements.length > 0 ? elements : text;
}

function CommandRunner({ theme, config, dashboard }) {
  const widgetId = config.id;
  // Initialize output from global state if available
  const [output, setOutput] = useState(() => globalCommandOutputs.get(widgetId) || '');
  const [isRunning, setIsRunning] = useState(false);
  const [showOutput, setShowOutput] = useState(config.showOutput !== false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const copyOutput = () => {
    // Remove ANSI codes for clean copy
    const cleanText = (error || output).replace(/\x1b\[[0-9;]+m/g, '');
    navigator.clipboard.writeText(cleanText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const startCommand = async () => {
    if (!config.command || !window.commandAPI) return;

    setIsRunning(true);
    setError(null);
    setOutput('');
    globalCommandOutputs.set(widgetId, ''); // Clear global state too

    try {
      const cwd = config.cwd || dashboard?._projectRoot;
      const result = await window.commandAPI.startStreaming(widgetId, config.command, cwd);
      if (!result.success) {
        setError(result.error);
        setIsRunning(false);
      }
    } catch (err) {
      setError(err.message);
      setIsRunning(false);
    }
  };

  const stopCommand = async () => {
    if (!window.commandAPI) return;

    try {
      await window.commandAPI.stopStreaming(widgetId);
      setIsRunning(false);
    } catch (err) {
      setError(err.message);
    }
  };

  // Check if already running on mount
  useEffect(() => {
    const checkRunning = async () => {
      if (window.commandAPI) {
        const result = await window.commandAPI.isRunning(widgetId);
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
    if (!window.commandAPI) return;

    const handleOutput = ({ widgetId: eventWidgetId, output: text }) => {
      if (eventWidgetId === widgetId) {
        setOutput(prev => {
          const newOutput = prev + text;
          globalCommandOutputs.set(widgetId, newOutput);
          return newOutput;
        });
      }
    };

    const handleExit = ({ widgetId: eventWidgetId, code }) => {
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

    const handleError = ({ widgetId: eventWidgetId, error: errorMsg }) => {
      if (eventWidgetId === widgetId) {
        setError(errorMsg);
        setIsRunning(false);
      }
    };

    const outputHandler = window.commandAPI.onOutput(handleOutput);
    const exitHandler = window.commandAPI.onExit(handleExit);
    const errorHandler = window.commandAPI.onError(handleError);

    return () => {
      window.commandAPI.offOutput(outputHandler);
      window.commandAPI.offExit(exitHandler);
      window.commandAPI.offError(errorHandler);
    };
  }, [widgetId]);

  // Auto-run if configured
  useEffect(() => {
    if (config.autoRun && !isRunning) {
      startCommand();
    }
  }, [config.command, config.autoRun]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="widget-label" style={{ fontFamily: theme.textBody, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
        <span>{config.label}</span>
        <div style={{ display: 'flex', gap: 8 }}>
          {config.command && (output || error) && (
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
          {config.command && (
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
            disabled={!config.command}
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
      {!config.command && (
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
}

function FlippableTest({ theme, config }) {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div
      className={`widget-flip-container ${isFlipped ? 'flipped' : ''}`}
      style={{ width: '100%', height: '100%' }}
      onContextMenu={(e) => {
        e.preventDefault();
        setIsFlipped(!isFlipped);
      }}
    >
      <div className="widget-flip-inner">
        {/* Front - Just apply .widget class like normal */}
        <div
          className="widget widget-flip-front"
          style={{
            background: theme.widgetBg,
            border: theme.widgetBorder,
            borderRadius: theme.widgetRadius,
          }}
        >
          <div className="widget-label" style={{ fontFamily: theme.textBody }}>
            Flippable Test Widget
          </div>
          <p style={{ marginBottom: '8px', color: theme.accent, fontSize: '0.85rem' }}>
            Right-click to flip and see JSON
          </p>
          {[...Array(20)].map((_, i) => (
            <div key={i} style={{
              padding: '8px 12px',
              marginBottom: '4px',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '4px',
              border: '1px solid rgba(255,255,255,0.1)',
              color: theme.textColor,
              fontSize: '0.85rem',
            }}>
              Item {i + 1} - This is scrollable content to test the flip animation
            </div>
          ))}
        </div>

        {/* Back - Just apply .widget class like normal */}
        <div
          className="widget widget-flip-back"
          style={{
            background: theme.widgetBg,
            border: theme.widgetBorder,
            borderRadius: theme.widgetRadius,
          }}
        >
          <div className="widget-label" style={{ fontFamily: theme.textBody, marginBottom: '12px' }}>
            Back Side - JSON Configuration
          </div>
          <textarea
            style={{
              width: '100%',
              height: 'calc(100% - 40px)',
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '4px',
              padding: '12px',
              color: theme.textColor,
              fontFamily: 'monospace',
              fontSize: '0.75rem',
              resize: 'none',
            }}
            value={JSON.stringify(config, null, 2)}
            readOnly
          />
        </div>
      </div>
    </div>
  );
}

function WebView({ theme, config }) {
  const [currentUrl, setCurrentUrl] = useState(config.url || '');
  const [inputUrl, setInputUrl] = useState(config.url || '');
  const [canGoBack, setCanGoBack] = useState(false);
  const [canGoForward, setCanGoForward] = useState(false);
  const containerRef = useRef(null);
  const widgetId = config.id;

  // Create WebContentsView on mount
  useEffect(() => {
    const createView = async () => {
      if (!window.webContentsViewAPI || !containerRef.current) return;

      // Get bounds from container
      const rect = containerRef.current.getBoundingClientRect();
      const bounds = {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      };

      // Use widget background color from theme, fallback to bgApp
      // Convert rgba to rgb by removing alpha channel if present
      let backgroundColor = theme.widgetBg || theme.bgApp;
      if (typeof backgroundColor === 'string' && backgroundColor.startsWith('rgba')) {
        // Extract rgb values from rgba(r, g, b, a) format
        const match = backgroundColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (match) {
          backgroundColor = `rgb(${match[1]}, ${match[2]}, ${match[3]})`;
        }
      }

      console.log('[WebView] Creating WebContentsView:', { widgetId, bounds, url: currentUrl, backgroundColor });

      await window.webContentsViewAPI.create(widgetId, currentUrl, bounds, backgroundColor);

      // Check initial navigation state
      const navState = await window.webContentsViewAPI.canNavigate(widgetId);
      if (navState?.success) {
        setCanGoBack(navState.canGoBack || false);
        setCanGoForward(navState.canGoForward || false);
      }
    };

    createView();

    // Listen for navigation events
    const handleNavigated = ({ widgetId: navWidgetId, url, canGoBack: navCanGoBack, canGoForward: navCanGoForward }) => {
      if (navWidgetId === widgetId) {
        console.log('[WebView] Navigated to:', url);
        setCurrentUrl(url);
        setInputUrl(url);
        setCanGoBack(navCanGoBack || false);
        setCanGoForward(navCanGoForward || false);
      }
    };

    const handler = window.webContentsViewAPI.onNavigated(handleNavigated);

    // Cleanup on unmount
    return () => {
      window.webContentsViewAPI.offNavigated(handler);
      window.webContentsViewAPI.destroy(widgetId);
    };
  }, [widgetId]);

  // Update bounds when container resizes
  useEffect(() => {
    if (!containerRef.current || !window.webContentsViewAPI) return;

    const resizeObserver = new ResizeObserver(() => {
      const rect = containerRef.current.getBoundingClientRect();
      const bounds = {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      };

      window.webContentsViewAPI.updateBounds(widgetId, bounds);
    });

    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, [widgetId]);

  const handleNavigate = async () => {
    let url = inputUrl.trim();
    if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://' + url;
    }

    if (window.webContentsViewAPI) {
      await window.webContentsViewAPI.navigate(widgetId, url);
    }
  };

  const handleReload = async () => {
    if (window.webContentsViewAPI) {
      await window.webContentsViewAPI.reload(widgetId);
    }
  };

  const goBack = async () => {
    if (window.webContentsViewAPI) {
      const result = await window.webContentsViewAPI.goBack(widgetId);
      if (!result.success) {
        setCanGoBack(false);
      }
    }
  };

  const goForward = async () => {
    if (window.webContentsViewAPI) {
      const result = await window.webContentsViewAPI.goForward(widgetId);
      if (!result.success) {
        setCanGoForward(false);
      }
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="widget-label" style={{ fontFamily: theme.textBody, marginBottom: 8 }}>{config.label}</div>

      {/* Navigation Bar */}
      <div style={{
        display: 'flex',
        gap: 6,
        marginBottom: 8,
        alignItems: 'center'
      }}>
        <button
          onClick={goBack}
          disabled={!canGoBack}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            cursor: !canGoBack ? 'not-allowed' : 'pointer',
            fontSize: '0.75rem',
            opacity: !canGoBack ? 0.4 : 1
          }}
          title="Back"
        >
          ←
        </button>

        <button
          onClick={goForward}
          disabled={!canGoForward}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            cursor: !canGoForward ? 'not-allowed' : 'pointer',
            fontSize: '0.75rem',
            opacity: !canGoForward ? 0.4 : 1
          }}
          title="Forward"
        >
          →
        </button>

        <button
          onClick={handleReload}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            cursor: 'pointer',
            fontSize: '0.75rem'
          }}
          title="Reload"
        >
          ↻
        </button>

        <input
          type="text"
          value={inputUrl}
          onChange={(e) => setInputUrl(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleNavigate()}
          placeholder="Enter URL..."
          style={{
            flex: 1,
            background: 'rgba(255,255,255,0.05)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            fontFamily: theme.textCode,
            fontSize: '0.75rem',
            outline: 'none'
          }}
        />

        <button
          onClick={handleNavigate}
          style={{
            background: theme.accent,
            border: 'none',
            borderRadius: 4,
            padding: '6px 12px',
            color: '#000',
            cursor: 'pointer',
            fontSize: '0.75rem',
            fontWeight: 600
          }}
        >
          Go
        </button>
      </div>

      {/* WebContentsView Placeholder */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          position: 'relative',
          border: `1px solid ${theme.accent}22`,
          borderRadius: 4,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {!currentUrl && (
          <div style={{
            fontFamily: theme.textBody,
            color: theme.neutral,
            fontSize: '0.8rem',
            padding: '12px',
            textAlign: 'center',
            position: 'absolute'
          }}>
            Enter a URL to navigate
          </div>
        )}
      </div>
    </div>
  );
}

// Format tool call descriptions with relevant details
function formatToolDescription(toolName, input) {
  if (!input) return toolName;

  switch (toolName) {
    case 'Read':
      return `Read ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Write':
      return `Write ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Edit':
      return `Edit ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Bash':
      const cmd = input.command || '';
      const shortCmd = cmd.length > 40 ? cmd.substring(0, 40) + '...' : cmd;
      return `Run: ${shortCmd}`;
    case 'Grep':
      return `Search for "${input.pattern || ''}"`;
    case 'Glob':
      return `Find files: ${input.pattern || ''}`;
    case 'Task':
      return `${input.description || 'Start task'}`;
    case 'WebFetch':
      const url = input.url || '';
      const domain = url.replace(/^https?:\/\//, '').split('/')[0];
      return `Fetch ${domain}`;
    case 'WebSearch':
      return `Search: ${input.query || ''}`;
    default:
      return toolName;
  }
}

function ChatMessage({ msg, theme }) {
  const components = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={oneDark}
          language={match[1]}
          PreTag="div"
          customStyle={{ margin: '8px 0', borderRadius: '6px', fontSize: '0.7rem' }}
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className="inline-code" {...props}>{children}</code>
      );
    }
  };

  return (
    <>
      {msg.toolCalls && msg.toolCalls.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '6px',
          marginBottom: '8px'
        }}>
          {msg.toolCalls.map((tool, idx) => (
            <div
              key={idx}
              style={{
                fontFamily: theme.textBody,
                fontSize: '0.7rem',
                padding: '4px 8px',
                backgroundColor: `${theme.accent}22`,
                border: `1px solid ${theme.accent}44`,
                borderRadius: '4px',
                color: theme.accent,
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <span style={{ opacity: 0.6 }}>🔧</span>
              {tool.description || tool.name}
            </div>
          ))}
        </div>
      )}
      <div className={`chat-bubble ${msg.from}`} style={{
        fontFamily: theme.textBody,
        backgroundColor: msg.from === 'user' ? 'rgba(255,255,255,0.1)' : `${theme.accent}22`,
        borderColor: msg.from === 'assistant' ? theme.accent : 'transparent',
      }}>
        <ReactMarkdown components={components}>{msg.text}</ReactMarkdown>
      </div>
    </>
  );
}

function Chat({ theme, config, dashboardId, dashboard, widgetKey, currentConversationId, setCurrentConversationId }) {
  const [messages, setMessages] = useState([]);
  // Initialize input from global state if available
  const [input, setInput] = useState(() => globalChatInputs.get(currentConversationId) || '');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamText, setCurrentStreamText] = useState(''); // Temporary state for streaming display
  const [currentToolCalls, setCurrentToolCalls] = useState([]); // Track active tool calls
  const [isLoading, setIsLoading] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [showConversations, setShowConversations] = useState(false);
  const [permissionMode, setPermissionMode] = useState('bypassPermissions'); // 'plan' or 'bypassPermissions', default to bypassPermissions
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const processedTextsRef = useRef(new Set()); // Track processed text content to avoid duplicates
  const processedToolsRef = useRef(new Set()); // Track processed tool calls to avoid duplicates
  const currentStreamTextRef = useRef(''); // Track stream text for saving (ref to avoid closure issues)
  const currentToolCallsRef = useRef([]); // Track tool calls for saving (ref to avoid closure issues)
  const hasInitializedRef = useRef(false); // Track if we've already loaded conversations
  const previousMessageCountRef = useRef(0); // Track previous message count to detect new messages
  const hasLoadedInitialMessagesRef = useRef(false); // Track if we've loaded the initial messages

  // Capture dashboard context once at mount - use latest value when needed but don't cause re-renders
  const dashboardContextRef = useRef(null);
  if (dashboard && !dashboardContextRef.current) {
    dashboardContextRef.current = {
      filePath: dashboard._sourcePath,
      id: dashboard.id,
      title: dashboard.title,
      config: dashboard
    };
  }

  // Determine backend type: 'claude' for Claude Agent SDK, 'mock' for demo, or custom handler
  const backend = config.backend || 'mock';

  // Sync input with global state when it changes
  useEffect(() => {
    if (currentConversationId) {
      globalChatInputs.set(currentConversationId, input);
    }
  }, [input, currentConversationId]);

  // Load input from global state when conversation changes
  useEffect(() => {
    if (currentConversationId) {
      const savedInput = globalChatInputs.get(currentConversationId) || '';
      setInput(savedInput);
    }
  }, [currentConversationId]);

  // Load conversations on mount ONLY - runs once per widget instance
  useEffect(() => {
    // Skip if already initialized (prevents re-initialization on remount)
    if (hasInitializedRef.current) return;

    const loadConversations = async () => {
      if (backend !== 'claude' || !window.claudeAPI) return;

      hasInitializedRef.current = true; // Mark as initialized

      try {
        const result = await window.claudeAPI.listConversations(widgetKey);
        if (result.success) {
          setConversations(result.conversations);

          // Only set conversation if not already set from parent
          if (!currentConversationId) {
            // If no conversations exist, create a default one
            if (result.conversations.length === 0) {
              const createResult = await window.claudeAPI.createConversation(widgetKey, 'New Conversation');
              if (createResult.success) {
                setConversations([createResult.conversation]);
                setCurrentConversationId(createResult.conversation.id);
              }
            } else {
              // Load most recent conversation
              setCurrentConversationId(result.conversations[0].id);
            }
          }
        }
      } catch (error) {
        console.error('[Chat UI] Error loading conversations:', error);
      }
    };

    loadConversations();
  }, []); // Empty deps - but protected by hasInitializedRef

  // Function to reload messages from backend (single source of truth)
  const reloadMessages = async () => {
    if (!currentConversationId) {
      setMessages([]);
      setIsStreaming(false);
      return;
    }

    if (backend === 'claude' && window.claudeAPI) {
      try {
        const result = await window.claudeAPI.getMessages(currentConversationId);
        if (result.success && result.messages) {
          console.log(`[Chat UI] Loaded ${result.messages.length} messages for ${currentConversationId}, streaming: ${result.isStreaming}`);
          console.log('[Chat UI] Messages from backend:', JSON.stringify(result.messages, null, 2));

          // Check if any messages have toolCalls
          const messagesWithTools = result.messages.filter(m => m.toolCalls && m.toolCalls.length > 0);
          console.log(`[Chat UI] Messages with toolCalls: ${messagesWithTools.length}`);

          setMessages(result.messages);
          // Sync streaming state from backend
          setIsStreaming(result.isStreaming || false);
        } else {
          setMessages([]);
          setIsStreaming(false);
        }
      } catch (error) {
        console.error('[Chat UI] Error loading messages:', error);
        setMessages([]);
        setIsStreaming(false);
      }
    } else {
      // For non-Claude backends, use config messages
      setMessages(config.messages || []);
      setIsStreaming(false);
    }
  };

  // Load messages when conversation changes
  useEffect(() => {
    const loadMessages = async () => {
      setIsLoading(true);
      hasLoadedInitialMessagesRef.current = false; // Reset flag before loading
      previousMessageCountRef.current = 0; // Reset count to enable initial scroll positioning
      await reloadMessages();
      setIsLoading(false);
      // Scroll to bottom immediately after initial load (no animation)
      // Use setTimeout to ensure DOM has updated
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
        hasLoadedInitialMessagesRef.current = true; // Mark as loaded after scroll
      }, 0);
    };

    loadMessages();
  }, [backend, config.messages, currentConversationId]);

  // Only scroll when new messages are added or when streaming (not on initial load)
  useEffect(() => {
    const currentCount = messages.length;
    const previousCount = previousMessageCountRef.current;

    // Only scroll if we've finished initial load AND count actually increased OR we're streaming
    if (hasLoadedInitialMessagesRef.current && (currentCount > previousCount || isStreaming)) {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }
    }

    // Update the ref for next comparison
    previousMessageCountRef.current = currentCount;
  }, [messages, isStreaming]);

  // Scroll when streaming text or tool calls update
  useEffect(() => {
    if (isStreaming && hasLoadedInitialMessagesRef.current) {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTo({
          top: messagesContainerRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }
    }
  }, [currentStreamText, currentToolCalls, isStreaming]);

  // Handle Esc key to interrupt streaming and Shift+Tab to toggle mode
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Esc to interrupt streaming
      if (e.key === 'Escape' && isStreaming && backend === 'claude' && currentConversationId) {
        console.log('[Chat UI] Interrupting stream...');
        if (window.claudeAPI) {
          window.claudeAPI.interrupt(currentConversationId).then(() => {
            console.log('[Chat UI] Stream interrupted');
            setIsStreaming(false);
            setCurrentStreamText('');
            setCurrentToolCalls([]);
            currentStreamTextRef.current = '';
            currentToolCallsRef.current = [];
          }).catch(err => {
            console.error('[Chat UI] Error interrupting:', err);
          });
        }
      }

      // Shift+Tab to toggle permission mode
      if (e.key === 'Tab' && e.shiftKey && backend === 'claude') {
        e.preventDefault();
        setPermissionMode(mode => mode === 'plan' ? 'bypassPermissions' : 'plan');
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isStreaming, currentConversationId, backend]);

  // Set up Claude API listeners for streaming responses
  useEffect(() => {
    if (backend !== 'claude' || !window.claudeAPI || !currentConversationId) return;

    console.log('[Chat UI] Setting up event listeners for conversationId:', currentConversationId);

    // Listen for message chunks
    const handleMessage = (data) => {
      if (data.chatId !== currentConversationId) return;

      console.log('[Chat UI] Received message type:', data.message?.type);

      // IMPORTANT: Ignore 'result' messages - they're summaries we've already shown via streaming
      if (data.message?.type === 'result') {
        console.log('[Chat UI] Ignoring result message (already displayed via streaming)');
        return;
      }

      // Handle SDKAssistantMessage - contains Claude's responses
      if (data.message?.type === 'assistant') {
        const content = data.message.message?.content;

        if (Array.isArray(content)) {
          // Extract text blocks and tool calls
          content.forEach((block) => {
            if (block.type === 'text' && block.text) {
              // Use the actual text content as the key to avoid duplicates
              const textHash = block.text.trim();

              if (!processedTextsRef.current.has(textHash)) {
                console.log('[Chat UI] NEW text block, appending to stream. First 50 chars:', textHash.substring(0, 50));
                processedTextsRef.current.add(textHash);

                // Append to currentStreamText for display during streaming
                setCurrentStreamText(prev => {
                  const needsSpace = prev && !prev.endsWith(' ') && !prev.endsWith('\n');
                  const newText = prev + (needsSpace ? ' ' : '') + block.text;
                  currentStreamTextRef.current = newText; // Also update ref for saving
                  return newText;
                });
              } else {
                console.log('[Chat UI] DUPLICATE text block detected, skipping. First 50 chars:', textHash.substring(0, 50));
              }
            } else if (block.type === 'tool_use' && block.name) {
              // Track tool calls with their inputs
              const toolId = block.id || block.name;
              if (!processedToolsRef.current.has(toolId)) {
                console.log('[Chat UI] NEW tool call:', block.name, block.input);
                processedToolsRef.current.add(toolId);

                // Format tool description with details
                const description = formatToolDescription(block.name, block.input);
                const toolCall = {
                  name: block.name,
                  id: toolId,
                  description,
                  input: block.input
                };

                // Update both state (for display) and ref (for saving)
                setCurrentToolCalls(prev => [...prev, toolCall]);
                currentToolCallsRef.current = [...currentToolCallsRef.current, toolCall];
              }
            }
          });
        }
      }
    };

    const handleComplete = async (data) => {
      if (data.chatId !== currentConversationId) return;

      console.log('[Chat UI] Stream complete - reloading (global handler already saved)');

      // Note: Global stream manager already saved the message
      // Just reload messages and clear UI state
      await reloadMessages();

      setIsStreaming(false);
      setCurrentStreamText('');
      setCurrentToolCalls([]);
      currentStreamTextRef.current = '';
      currentToolCallsRef.current = [];
      processedTextsRef.current.clear();
      processedToolsRef.current.clear();
    };

    const handleError = async (data) => {
      if (data.chatId !== currentConversationId) return;

      console.error('[Chat UI] Error:', data.error);

      // Note: Global stream manager already saved the error message
      // Just reload and clear state
      await reloadMessages();
      setCurrentStreamText('');
      setCurrentToolCalls([]);
      currentStreamTextRef.current = '';
      currentToolCallsRef.current = '';
      processedTextsRef.current.clear();
      processedToolsRef.current.clear();
      setIsStreaming(false);
    };

    const messageHandler = window.claudeAPI.onMessage(handleMessage);
    const completeHandler = window.claudeAPI.onComplete(handleComplete);
    const errorHandler = window.claudeAPI.onError(handleError);

    return () => {
      console.log('[Chat UI] Cleaning up event listeners for conversationId:', currentConversationId);
      // Note: Global stream manager handles saving, even if component unmounts
      // Remove only this instance's listeners
      window.claudeAPI.offMessage(messageHandler);
      window.claudeAPI.offComplete(completeHandler);
      window.claudeAPI.offError(errorHandler);
    };
  }, [backend, currentConversationId]);

  // Helper to save a message to backend
  const saveMessageToBackend = async (message) => {
    if (backend === 'claude' && window.claudeAPI && currentConversationId) {
      try {
        // Remove UI-only fields before saving
        const { finalized, ...messageToSave } = message;
        await window.claudeAPI.saveMessage(currentConversationId, messageToSave);

        // Update conversation's lastMessageAt
        await window.claudeAPI.updateConversation(widgetKey, currentConversationId, {
          lastMessageAt: new Date().toISOString()
        });
      } catch (error) {
        console.error('[Chat UI] Error saving message:', error);
      }
    }
  };

  // Create a new conversation
  const handleNewConversation = async () => {
    if (backend !== 'claude' || !window.claudeAPI) return;

    try {
      const result = await window.claudeAPI.createConversation(widgetKey, 'New Conversation');
      if (result.success) {
        setConversations(prev => [result.conversation, ...prev]);
        setCurrentConversationId(result.conversation.id);
        setMessages([]);
      }
    } catch (error) {
      console.error('[Chat UI] Error creating conversation:', error);
    }
  };

  // Switch to a different conversation
  const handleSwitchConversation = (conversationId) => {
    setCurrentConversationId(conversationId);
    setShowConversations(false);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { from: 'user', text: input };

    // Save user message to backend immediately
    await saveMessageToBackend(userMessage);

    // Reload messages to show user message (backend is single source of truth)
    await reloadMessages();

    const messageToSend = input;
    setInput('');
    setIsStreaming(true);
    setCurrentStreamText(''); // Reset stream text
    setCurrentToolCalls([]); // Reset tool calls
    currentStreamTextRef.current = ''; // Reset text ref
    currentToolCallsRef.current = []; // Reset tool calls ref
    processedTextsRef.current.clear(); // Reset for new message
    processedToolsRef.current.clear(); // Reset for new message

    if (backend === 'claude' && window.claudeAPI) {
      try {
        console.log('[Chat UI] Sending message to Claude...');

        // Prepare options with dashboard context for system prompt
        const options = {
          ...(config.claudeOptions || {}),
          permissionMode: permissionMode === 'plan' ? 'plan' : 'bypassPermissions',
          dashboardContext: dashboard ? {
            filePath: dashboard._sourcePath,
            id: dashboard.id,
            title: dashboard.title,
            config: dashboard
          } : null
        };

        // Send to Claude Agent SDK - response will come via event listeners
        const result = await window.claudeAPI.sendMessage(
          currentConversationId,
          messageToSend,
          options
        );

        if (!result.success) {
          throw new Error(result.error);
        }

        console.log('[Chat UI] Message sent successfully, waiting for stream...');
        // Streaming and completion are handled by event listeners
      } catch (error) {
        console.error('[Chat UI] Send error:', error);
        const errorMessage = {
          from: 'assistant',
          text: `Error: ${error.message}`
        };
        await saveMessageToBackend(errorMessage);
        await reloadMessages();
        setIsStreaming(false);
        setCurrentStreamText('');
      }
    } else if (backend === 'mock') {
      // Mock response for demo
      setTimeout(() => {
        setMessages(prev => [...prev, {
          from: 'assistant',
          text: 'Here\'s an example:\n\n```javascript\nconst hello = "world";\nconsole.log(hello);\n```\n\nLet me know if you need more help!'
        }]);
        setIsStreaming(false);
      }, 500);
    } else if (typeof backend === 'function') {
      // Custom backend handler
      try {
        const response = await backend(input, messages);
        setMessages(prev => [...prev, {
          from: 'assistant',
          text: response
        }]);
      } catch (error) {
        setMessages(prev => [...prev, {
          from: 'assistant',
          text: `Error: ${error.message}`
        }]);
      }
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const currentConvo = conversations.find(c => c.id === currentConversationId);

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          {config.label}
          {backend === 'claude' && <span style={{ marginLeft: 8, opacity: 0.5 }}>• Claude</span>}
        </div>
        {backend === 'claude' && (
          <div style={{ display: 'flex', gap: 8, fontSize: '0.7rem' }}>
            <button
              onClick={handleNewConversation}
              style={{
                background: theme.accent,
                border: 'none',
                borderRadius: 4,
                padding: '4px 8px',
                color: '#000',
                cursor: 'pointer',
                fontSize: '0.7rem'
              }}
            >
              + New
            </button>
            {conversations.length > 1 && (
              <button
                onClick={() => setShowConversations(!showConversations)}
                style={{
                  background: 'rgba(255,255,255,0.1)',
                  border: `1px solid ${theme.accent}44`,
                  borderRadius: 4,
                  padding: '4px 8px',
                  color: theme.textColor,
                  cursor: 'pointer',
                  fontSize: '0.7rem'
                }}
              >
                {conversations.length} conversations
              </button>
            )}
          </div>
        )}
      </div>
      {showConversations && conversations.length > 0 && (
        <div style={{
          maxHeight: 150,
          overflowY: 'auto',
          background: 'rgba(0,0,0,0.3)',
          borderRadius: 4,
          marginBottom: 8,
          padding: 4
        }}>
          {conversations.map(convo => (
            <div
              key={convo.id}
              onClick={() => handleSwitchConversation(convo.id)}
              style={{
                padding: '6px 8px',
                cursor: 'pointer',
                background: convo.id === currentConversationId ? theme.accent + '22' : 'transparent',
                borderLeft: convo.id === currentConversationId ? `2px solid ${theme.accent}` : '2px solid transparent',
                fontSize: '0.7rem',
                marginBottom: 2,
                borderRadius: 2
              }}
            >
              <div style={{ fontWeight: convo.id === currentConversationId ? 'bold' : 'normal' }}>
                {convo.title}
              </div>
              <div style={{ opacity: 0.5, fontSize: '0.65rem' }}>
                {new Date(convo.lastMessageAt).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="chat-messages" ref={messagesContainerRef}>
        {messages.map((msg, i) => (
          <ChatMessage key={i} msg={msg} theme={theme} />
        ))}
        {isStreaming && (currentToolCalls.length > 0 || currentStreamText) && (
          <ChatMessage
            msg={{
              from: 'assistant',
              text: currentStreamText,
              toolCalls: currentToolCalls
            }}
            theme={theme}
          />
        )}
        {isStreaming && !currentStreamText && currentToolCalls.length === 0 && (
          <div className="chat-bubble assistant" style={{
            fontFamily: theme.textBody,
            backgroundColor: `${theme.accent}22`,
            borderColor: theme.accent,
            minHeight: '32px'
          }}>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {backend === 'claude' && (
        <div style={{
          fontSize: '0.65rem',
          color: theme.accent,
          padding: '4px 8px',
          opacity: 0.7,
          fontFamily: theme.textBody
        }}>
          {permissionMode === 'bypassPermissions' ? 'Bypass permissions: On' : 'Plan mode: On'} (Shift+Tab to toggle)
        </div>
      )}
      <div className="chat-input-row">
        <textarea
          className="chat-input"
          style={{
            fontFamily: theme.textBody,
            borderColor: `${theme.accent}44`,
            resize: 'none',
            minHeight: '28px',
            maxHeight: '150px',
            overflow: 'auto'
          }}
          placeholder="Type a message..."
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            // Auto-resize textarea
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
          }}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          className="chat-send"
          style={{ backgroundColor: theme.accent }}
          onClick={handleSend}
        >
          →
        </button>
      </div>
    </>
  );
}

// Widget type registry
const widgetComponents = {
  barChart: BarChart,
  diffList: DiffList,
  testResults: TestResults,
  stat: Stat,
  progress: Progress,
  chat: Chat,
  fileList: FileList,
  todoList: TodoList,
  claudeTodos: ClaudeTodoList,
  keyValue: KeyValue,
  jsonViewer: JsonViewer,
  markdown: Markdown,
  errorTest: ErrorTestWidget,
  layoutSettings: LayoutSettings,
  commandRunner: CommandRunner,
  webView: WebView,
  flippableTest: FlippableTest,
};

function Widget({ theme, config, clipPath, onResize, onDelete, dashboardId, dashboard, layout, allWidgets, widgetConversations, setWidgetConversations, reloadTrigger }) {
  const Component = widgetComponents[config.type];
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuPos, setContextMenuPos] = useState({ x: 0, y: 0 });
  const [isRegenerating, setIsRegenerating] = useState(false);

  if (!Component) {
    // Show unknown widget type instead of returning null
    return (
      <div
        className="widget"
        style={{
          background: theme.widgetBg,
          border: '2px solid ' + theme.negative,
          borderRadius: theme.widgetRadius,
          clipPath: clipPath,
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
        <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px', color: theme.negative }}>
          Unknown Widget Type
        </div>
        <div style={{ fontSize: '14px', color: theme.textColor, opacity: 0.7, marginBottom: '8px' }}>
          Type: "{config.type}"
        </div>
        <div style={{ fontSize: '12px', color: theme.textColor, opacity: 0.5 }}>
          Widget ID: {config.id}
        </div>
      </div>
    );
  }

  const isChat = config.type === 'chat';
  const widgetKey = `${dashboardId}-${config.id}`;
  const hasRegenerate = config.regenerateCommand || config.regenerateScript || config.regenerate;

  const handleDrag = ({ x, y }) => {
    if (onResize && dashboardId) {
      onResize(dashboardId, config.id, { x, y });
    }
  };

  const handleResizeEnd = ({ width, height }) => {
    if (onResize && dashboardId) {
      onResize(dashboardId, config.id, { width, height });
    }
  };

  const handleMouseDown = (e) => {
    // Check for ctrl + option + command (Mac)
    if (e.metaKey && e.ctrlKey && e.altKey) {
      e.preventDefault();
      e.stopPropagation();
      if (onDelete && dashboardId) {
        onDelete(dashboardId, config.id);
      }
    }
  };

  const handleContextMenu = (e) => {
    console.log('Context menu triggered, hasRegenerate:', hasRegenerate, 'config:', config.id);
    if (!hasRegenerate) return;
    e.preventDefault();
    e.stopPropagation();
    setContextMenuPos({ x: e.clientX, y: e.clientY });
    setShowContextMenu(true);
    console.log('Context menu shown at:', e.clientX, e.clientY);
  };

  const handleRegenerate = async () => {
    console.log('Regenerating widget:', config.id);
    setShowContextMenu(false);
    setIsRegenerating(true);
    try {
      if (!window.dashboardAPI || !window.dashboardAPI.regenerateWidget) {
        console.error('dashboardAPI.regenerateWidget not available');
        return;
      }
      const result = await window.dashboardAPI.regenerateWidget(dashboardId, config.id);
      console.log('Regenerate result:', result);
      if (!result.success) {
        console.error('Regenerate failed:', result.error);
        alert(`Regenerate failed: ${result.error}`);
      } else {
        console.log('Regenerate succeeded');
      }
    } catch (error) {
      console.error('Regenerate error:', error);
      alert(`Regenerate error: ${error.message}`);
    } finally {
      setIsRegenerating(false);
    }
  };

  // Close context menu when clicking outside
  useEffect(() => {
    if (showContextMenu) {
      const handleClick = () => setShowContextMenu(false);
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [showContextMenu]);

  const widgetContent = (
    <div
      className={`widget ${isChat ? 'chat-widget' : ''}`}
      onMouseDown={handleMouseDown}
      onContextMenu={handleContextMenu}
      style={{
        background: theme.widgetBg,
        border: theme.widgetBorder,
        borderRadius: theme.widgetRadius,
        clipPath: clipPath,
        width: '100%',
        height: '100%',
        position: 'relative',
        opacity: isRegenerating ? 0.6 : 1,
        transition: 'opacity 0.2s',
      }}
    >
      <ErrorBoundary theme={theme}>
        <Component
          theme={theme}
          config={config}
          dashboardId={dashboardId}
          dashboard={dashboard}
          layout={layout}
          widgetKey={widgetKey}
          currentConversationId={widgetConversations[widgetKey] || null}
          setCurrentConversationId={(id) => setWidgetConversations(prev => ({ ...prev, [widgetKey]: id }))}
          widgetConversations={widgetConversations}
          reloadTrigger={reloadTrigger}
        />
      </ErrorBoundary>
      {/* Context Menu */}
      {showContextMenu && (
        <div
          style={{
            position: 'fixed',
            left: contextMenuPos.x,
            top: contextMenuPos.y,
            background: 'rgba(30, 30, 30, 0.95)',
            border: `1px solid ${theme.accent}`,
            borderRadius: 6,
            padding: 4,
            zIndex: 10000,
            minWidth: 150,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            onClick={handleRegenerate}
            style={{
              padding: '8px 12px',
              cursor: 'pointer',
              color: theme.accent,
              fontSize: '0.85rem',
              fontFamily: theme.textBody,
              transition: 'background-color 0.2s',
              borderRadius: 4,
            }}
            onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.1)'}
            onMouseLeave={(e) => e.target.style.backgroundColor = 'transparent'}
          >
            ⟳ Regenerate
          </div>
        </div>
      )}
    </div>
  );

  // Parse width/height - they might be strings like "200px" or numbers
  const parseSize = (size, defaultSize) => {
    if (!size) return defaultSize;
    if (typeof size === 'string') {
      return parseInt(size.replace('px', ''));
    }
    return size;
  };

  // Default dimensions for widgets without explicit sizes
  const defaultWidgetDimensions = {
    chat: { width: 400, height: 500 },
    barChart: { width: 300, height: 200 },
    stat: { width: 200, height: 150 },
    progress: { width: 250, height: 100 },
    fileList: { width: 250, height: 200 },
    todoList: { width: 250, height: 200 },
    claudeTodos: { width: 250, height: 200 },
    keyValue: { width: 200, height: 150 },
    diffList: { width: 300, height: 200 },
    layoutSettings: { width: 250, height: 200 },
    commandRunner: { width: 350, height: 250 },
    webView: { width: 400, height: 400 },
  };

  const defaults = defaultWidgetDimensions[config.type] || { width: 250, height: 200 };

  // Always use GridItem for draggable/resizable widgets
  return (
    <GridItem
      x={config.x || 0}
      y={config.y || 0}
      width={parseSize(config.width, defaults.width)}
      height={parseSize(config.height, defaults.height)}
      resizable={true}
      draggable={true}
      onDrag={handleDrag}
      onDragEnd={handleDrag}
      onResize={handleResizeEnd}
    >
      {widgetContent}
    </GridItem>
  );
}

function ProjectNode({ icon, active, accent, hoverAccent, onClick, onDelete }) {
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });

  // Support inline SVG string or predefined icon name
  const iconContent = typeof icon === 'string' && icons[icon]
    ? icons[icon]
    : typeof icon === 'string' && icon.trim().startsWith('<svg')
      ? <span dangerouslySetInnerHTML={{ __html: icon }} />
      : icons.square;

  const handleContextMenu = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setMenuPosition({ x: e.clientX, y: e.clientY });
    setShowContextMenu(true);
  };

  const handleDelete = () => {
    setShowContextMenu(false);
    if (onDelete) {
      onDelete();
    }
  };

  useEffect(() => {
    if (showContextMenu) {
      const handleClick = () => setShowContextMenu(false);
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [showContextMenu]);

  return (
    <>
      <div
        className={`project-node ${active ? 'active' : ''}`}
        style={{ '--accent': accent, '--hover-accent': hoverAccent }}
        onClick={onClick}
        onContextMenu={handleContextMenu}
      >
        {iconContent}
      </div>
      {showContextMenu && (
        <div
          style={{
            position: 'fixed',
            left: menuPosition.x,
            top: menuPosition.y,
            backgroundColor: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '6px',
            padding: '4px 0',
            zIndex: 10000,
            minWidth: 150,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            onClick={handleDelete}
            style={{
              padding: '8px 12px',
              cursor: 'pointer',
              color: '#f85149',
              fontSize: '0.85rem',
              fontFamily: 'system-ui',
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(248, 81, 73, 0.1)'}
            onMouseLeave={(e) => e.target.style.backgroundColor = 'transparent'}
          >
            Remove Project
          </div>
        </div>
      )}
    </>
  );
}

const defaultTheme = {
  bgApp: '#0d1117',
  textHead: 'system-ui, sans-serif',
  textBody: 'system-ui, sans-serif',
  accent: '#58a6ff',
  textColor: '#c9d1d9',
  positive: '#3fb950',
  negative: '#f85149',
  widgetBg: 'rgba(22, 27, 34, 0.8)',
  widgetBorder: '1px solid rgba(48, 54, 61, 0.8)',
  widgetRadius: '6px',
  chartRadius: '2px',
  bgLayer: {},
};

function Dashboard({ dashboard, allDashboards, onSelect, onWidgetResize, onWidgetDelete, widgetConversations, setWidgetConversations, dashboardVersion, onRefreshProjects, onDashboardsChange }) {
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState(null);

  const theme = dashboard.theme && typeof dashboard.theme === 'object'
    ? { ...defaultTheme, ...dashboard.theme }
    : defaultTheme;
  const { layout } = dashboard;

  const handleAddProject = (project) => {
    if (onRefreshProjects) {
      onRefreshProjects();
    }
  };

  const handleDeleteProject = (dashboardToDelete) => {
    setProjectToDelete(dashboardToDelete);
    setShowDeleteConfirm(true);
  };

  const confirmDelete = async () => {
    if (projectToDelete) {
      try {
        // If we're deleting the active dashboard, switch to another one first
        if (projectToDelete.id === dashboard.id && allDashboards.length > 1) {
          const nextDashboard = allDashboards.find(d => d.id !== projectToDelete.id);
          if (nextDashboard) {
            onSelect(nextDashboard.id);
          }
        }

        // Check if this dashboard is associated with a project
        if (projectToDelete._projectId && window.projectAPI) {
          // Remove as a project
          console.log('Removing project:', projectToDelete._projectId);
          await window.projectAPI.removeProject(projectToDelete._projectId);
        } else if (window.dashboardAPI && projectToDelete._sourcePath) {
          // Remove as a standalone dashboard file
          console.log('Removing standalone dashboard file:', projectToDelete._sourcePath);
          await window.dashboardAPI.removeConfigPath(projectToDelete._sourcePath);
        }

        // Trigger both refreshes
        if (onRefreshProjects) {
          onRefreshProjects();
        }

        // Notify parent to update dashboards list
        if (onDashboardsChange) {
          onDashboardsChange();
        }
      } catch (error) {
        console.error('Failed to remove:', error);
      }
    }
    setShowDeleteConfirm(false);
    setProjectToDelete(null);
  };

  // Use dashboardVersion as the reload trigger - this increments whenever any dashboard file changes
  // This will force widgets to reload their data from files
  const reloadTrigger = dashboardVersion;

  const [isRegeneratingAll, setIsRegeneratingAll] = useState(false);
  const [regenerateMessage, setRegenerateMessage] = useState(null);

  const handleRegenerateAll = async () => {
    setIsRegeneratingAll(true);
    setRegenerateMessage(null);

    try {
      const result = await window.dashboardAPI.regenerateAllWidgets(dashboard.id);
      if (result.success) {
        setRegenerateMessage({ type: 'success', text: result.message });
      } else {
        setRegenerateMessage({ type: 'error', text: result.error });
      }
      setTimeout(() => setRegenerateMessage(null), 5000);
    } catch (error) {
      setRegenerateMessage({ type: 'error', text: error.message });
      setTimeout(() => setRegenerateMessage(null), 5000);
    } finally {
      setIsRegeneratingAll(false);
    }
  };

  // Check if any widgets have regenerate commands
  const hasRegeneratableWidgets = dashboard.widgets?.some(w =>
    w.regenerateCommand || w.regenerateScript
  );

  const gridStyle = {
    gridTemplateColumns: layout.columns,
    gridTemplateRows: layout.rows,
    gridTemplateAreas: layout.areas,
  };

  // Get grid settings from layout
  const cellSize = layout?.gridSize || 16;
  const gapX = layout?.widgetGap || 10;
  const gapY = layout?.widgetGap || 10;
  const layoutMode = layout?.mode || 'single-pane';


  // Get overflow styles for window-frame based on mode
  const getWindowFrameOverflow = () => {
    switch (layoutMode) {
      case 'vertical-scroll':
      case 'horizontal-scroll':
        return 'visible'; // Allow scrolling to pass through
      case 'infinite-canvas':
      case 'single-pane':
      default:
        return 'hidden';
    }
  };

  return (
    <div className="window-frame" style={{ '--accent': theme.accent, overflow: getWindowFrameOverflow() }}>
      <div className="titlebar" />
      <div className="bg-layer" style={theme.bgLayer} />
      <div className="sidebar">
        {allDashboards.map((d) => {
          const dTheme = d.theme && typeof d.theme === 'object'
            ? { ...defaultTheme, ...d.theme }
            : defaultTheme;
          return (
            <ProjectNode
              key={d.id}
              icon={d.icon}
              active={d.id === dashboard.id}
              accent={dTheme.accent}
              hoverAccent={dTheme.accent}
              onClick={() => onSelect(d.id)}
              onDelete={() => handleDeleteProject(d)}
            />
          );
        })}
        <div
          className="project-node add-button"
          style={{ '--accent': theme.accent, '--hover-accent': theme.accent }}
          onClick={() => setShowAddDialog(true)}
          title="Add Project"
        >
          <svg viewBox="0 0 60 60">
            <line x1="30" y1="15" x2="30" y2="45" strokeWidth="4" />
            <line x1="15" y1="30" x2="45" y2="30" strokeWidth="4" />
          </svg>
        </div>
      </div>
      <div className="dashboard" style={{ backgroundColor: theme.bgApp }}>
        <div className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ fontFamily: theme.textHead, color: theme.textColor }}>{dashboard.title}</h1>
            <p style={{ fontFamily: theme.textBody, color: theme.accent }}>{dashboard.subtitle}</p>
          </div>
          {hasRegeneratableWidgets && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {regenerateMessage && (
                <span style={{
                  fontSize: '0.75rem',
                  color: regenerateMessage.type === 'success' ? theme.positive : theme.negative,
                  fontFamily: theme.textBody
                }}>
                  {regenerateMessage.text}
                </span>
              )}
              <button
                onClick={handleRegenerateAll}
                disabled={isRegeneratingAll}
                title="Regenerate all widgets with commands"
                style={{
                  background: theme.accent,
                  border: 'none',
                  borderRadius: 6,
                  padding: '8px 16px',
                  color: '#000',
                  cursor: isRegeneratingAll ? 'wait' : 'pointer',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  opacity: isRegeneratingAll ? 0.6 : 1,
                  fontFamily: theme.textBody,
                  transition: 'opacity 0.2s'
                }}
              >
                {isRegeneratingAll ? '⟳ Refreshing...' : '⟳ Refresh All Data'}
              </button>
            </div>
          )}
        </div>
        <Grid cellSize={cellSize} gapX={gapX} gapY={gapY} mode={layoutMode} width="100%" height="calc(100% - 80px)">
          {dashboard.widgets.map((widget) => (
            <Widget
              key={`${dashboard.id}-${widget.id}`}
              theme={theme}
              config={widget}
              clipPath={theme.clipPath}
              onResize={onWidgetResize}
              onDelete={onWidgetDelete}
              dashboardId={dashboard.id}
              dashboard={dashboard}
              layout={layout}
              allWidgets={dashboard.widgets}
              widgetConversations={widgetConversations}
              setWidgetConversations={setWidgetConversations}
              reloadTrigger={reloadTrigger}
            />
          ))}
        </Grid>
      </div>
      {showAddDialog && (
        <AddProjectDialog
          theme={theme}
          onClose={() => setShowAddDialog(false)}
          onAdd={handleAddProject}
        />
      )}
      {showDeleteConfirm && projectToDelete && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: theme.widgetBg || '#1a1a1a',
            border: theme.widgetBorder || '1px solid #333',
            borderRadius: theme.widgetRadius || '6px',
            padding: 24,
            width: 400,
            maxWidth: '90%',
            fontFamily: theme.textBody || 'system-ui'
          }}>
            <h2 style={{
              margin: '0 0 16px 0',
              color: theme.textColor || '#fff',
              fontSize: '1.2rem'
            }}>
              Remove Project
            </h2>
            <p style={{
              margin: '0 0 20px 0',
              color: theme.textColor || '#fff',
              opacity: 0.8,
              fontSize: '0.9rem'
            }}>
              Are you sure you want to remove "{projectToDelete.title || projectToDelete.id}"? This will remove it from the dashboard but won't delete any files from disk.
            </p>
            <div style={{
              display: 'flex',
              gap: 12,
              justifyContent: 'flex-end'
            }}>
              <button
                onClick={() => {
                  setShowDeleteConfirm(false);
                  setProjectToDelete(null);
                }}
                style={{
                  padding: '8px 16px',
                  backgroundColor: 'rgba(255,255,255,0.1)',
                  border: `1px solid ${theme.accent}44`,
                  borderRadius: 4,
                  color: theme.textColor || '#fff',
                  cursor: 'pointer',
                  fontFamily: theme.textBody || 'system-ui',
                  fontSize: '0.9rem'
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#f85149',
                  border: 'none',
                  borderRadius: 4,
                  color: '#fff',
                  cursor: 'pointer',
                  fontFamily: theme.textBody || 'system-ui',
                  fontSize: '0.9rem',
                  fontWeight: 600
                }}
              >
                Remove
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LoadingFallback() {
  return (
    <div className="app" style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      backgroundColor: '#0d1117',
      color: '#58a6ff',
      fontFamily: 'system-ui',
      fontSize: '1.2rem'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ marginBottom: 12 }}>Loading dashboard...</div>
        <div style={{
          width: 40,
          height: 40,
          border: '3px solid rgba(88, 166, 255, 0.2)',
          borderTopColor: '#58a6ff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto'
        }} />
      </div>
    </div>
  );
}

function AddProjectDialog({ theme, onClose, onAdd }) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('');
  const [pickerOpen, setPickerOpen] = useState(true);

  const handleSelectFolder = async () => {
    try {
      console.log('[AddProject] Starting folder selection...');

      if (!window.projectAPI) {
        console.error('[AddProject] window.projectAPI is not available!');
        setError('Project API not available');
        setPickerOpen(false);
        return;
      }

      const result = await window.projectAPI.selectFolder();
      console.log('[AddProject] Folder selection result:', result);

      setPickerOpen(false);

      if (result.canceled) {
        console.log('[AddProject] User canceled folder selection');
        onClose();
        return;
      }

      if (result.success && result.path) {
        console.log('[AddProject] Starting project creation for:', result.path);
        setIsSubmitting(true);
        setError(null);
        setStatus('Generating AI-powered icon and theme...');

        // Extract project name from path
        const projectName = result.path.split('/').pop() || result.path.split('\\').pop();
        console.log('[AddProject] Project name:', projectName);

        // Generate AI-powered design
        console.log('[AddProject] Calling generateDesign...');
        const designResult = await window.projectAPI.generateDesign(projectName);
        console.log('[AddProject] Design result:', designResult);
        const design = designResult.design;

        setStatus('Creating project...');

        console.log('[AddProject] Calling addProject...');
        const addResult = await window.projectAPI.addProject(
          result.path,
          'embedded', // Default to embedded
          undefined, // Auto-generate name from folder
          undefined  // No description
        );
        console.log('[AddProject] Add result:', addResult);

        if (addResult.success) {
          setStatus('Saving icon and theme...');

          // Update the project with the generated icon and theme
          console.log('[AddProject] Updating project with icon and theme...');
          await window.projectAPI.updateProject(addResult.project.id, {
            icon: design.icon,
            theme: design.theme
          });
          console.log('[AddProject] Project updated successfully');

          onAdd(addResult.project);
          onClose();
        } else {
          console.error('[AddProject] Failed to add project:', addResult.error);
          setError(addResult.error || 'Failed to add project');
          setIsSubmitting(false);
        }
      } else {
        console.warn('[AddProject] Invalid result from selectFolder:', result);
      }
    } catch (err) {
      console.error('[AddProject] Error in handleSelectFolder:', err);
      setError(err.message || 'Failed to add project');
      setIsSubmitting(false);
      setPickerOpen(false);
    }
  };

  useEffect(() => {
    // Auto-open folder picker when dialog opens
    handleSelectFolder();
  }, []);

  // Don't show anything while the native picker is open
  if (pickerOpen && !error && !isSubmitting) {
    return null;
  }

  // Show loading dialog if submitting
  if (isSubmitting && !error) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000
      }}>
        <div style={{
          backgroundColor: theme.widgetBg || '#1a1a1a',
          border: theme.widgetBorder || '1px solid #333',
          borderRadius: theme.widgetRadius || '6px',
          padding: 24,
          width: 400,
          maxWidth: '90%',
          fontFamily: theme.textBody || 'system-ui',
          textAlign: 'center'
        }}>
          <div style={{
            width: 40,
            height: 40,
            border: `3px solid ${theme.accent}44`,
            borderTopColor: theme.accent,
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 16px'
          }} />
          <div style={{
            color: theme.textColor || '#fff',
            fontSize: '1rem',
            marginBottom: 8
          }}>
            Adding Project
          </div>
          <div style={{
            color: theme.accent || '#58a6ff',
            fontSize: '0.85rem',
            opacity: 0.8
          }}>
            {status}
          </div>
        </div>
      </div>
    );
  }

  // Don't show anything if no error and not submitting (folder picker is open)
  if (!error) {
    return null;
  }

  // Only show error dialog if something went wrong
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: theme.widgetBg || '#1a1a1a',
        border: theme.widgetBorder || '1px solid #333',
        borderRadius: theme.widgetRadius || '6px',
        padding: 24,
        width: 400,
        maxWidth: '90%',
        fontFamily: theme.textBody || 'system-ui'
      }}>
        <h2 style={{
          margin: '0 0 16px 0',
          color: theme.textColor || '#fff',
          fontSize: '1.2rem'
        }}>
          Error Adding Project
        </h2>

        <div style={{
          marginBottom: 20,
          padding: 12,
          backgroundColor: 'rgba(248, 81, 73, 0.15)',
          border: '1px solid rgba(248, 81, 73, 0.3)',
          borderRadius: 4,
          color: theme.negative || '#f85149',
          fontSize: '0.9rem'
        }}>
          {error}
        </div>

        <div style={{
          display: 'flex',
          gap: 12,
          justifyContent: 'flex-end'
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              backgroundColor: theme.accent || '#58a6ff',
              border: 'none',
              borderRadius: 4,
              color: '#000',
              cursor: 'pointer',
              fontFamily: theme.textBody || 'system-ui',
              fontSize: '0.9rem',
              fontWeight: 600
            }}
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}

// Global stream manager to persist stream data across component unmounts
const globalStreamManager = {
  activeStreams: new Map(), // conversationId -> {text, toolCalls, processedTexts, processedTools}

  initStream(conversationId) {
    if (!this.activeStreams.has(conversationId)) {
      this.activeStreams.set(conversationId, {
        text: '',
        toolCalls: [],
        processedTexts: new Set(),
        processedTools: new Set()
      });
    }
    return this.activeStreams.get(conversationId);
  },

  getStream(conversationId) {
    return this.activeStreams.get(conversationId);
  },

  clearStream(conversationId) {
    this.activeStreams.delete(conversationId);
  },

  appendText(conversationId, text) {
    const stream = this.initStream(conversationId);
    const needsSpace = stream.text && !stream.text.endsWith(' ') && !stream.text.endsWith('\n');
    stream.text += (needsSpace ? ' ' : '') + text;
  },

  addToolCall(conversationId, toolCall) {
    const stream = this.initStream(conversationId);
    stream.toolCalls.push(toolCall);
  }
};

// Global state manager for chat inputs (persists across component unmounts)
const globalChatInputs = new Map(); // conversationId -> input text

// Global state manager for command outputs (persists across component unmounts)
const globalCommandOutputs = new Map(); // widgetId -> output text

// Set up global event handlers for Claude streaming (persists across component mounts/unmounts)
if (typeof window !== 'undefined' && !window.__claudeStreamHandlersInitialized) {
  window.__claudeStreamHandlersInitialized = true;

  // Wait for claudeAPI to be available
  const initGlobalHandlers = () => {
    if (!window.claudeAPI) {
      setTimeout(initGlobalHandlers, 100);
      return;
    }

    console.log('[Global Stream Manager] Initializing global Claude stream handlers');

    // Global message handler
    window.claudeAPI.onMessage((data) => {
      const { chatId, message } = data;
      if (!chatId || !message) return;

      if (message.type === 'result') return; // Ignore result messages

      if (message.type === 'assistant' && message.message?.content) {
        const stream = globalStreamManager.initStream(chatId);

        message.message.content.forEach((block) => {
          if (block.type === 'text' && block.text) {
            const textHash = block.text.trim();
            if (!stream.processedTexts.has(textHash)) {
              stream.processedTexts.add(textHash);
              globalStreamManager.appendText(chatId, block.text);
              console.log('[Global Stream Manager] Appended text for', chatId);
            }
          } else if (block.type === 'tool_use' && block.name) {
            const toolId = block.id || block.name;
            if (!stream.processedTools.has(toolId)) {
              stream.processedTools.add(toolId);
              const toolCall = {
                name: block.name,
                id: toolId,
                description: block.name, // Simplified, can enhance later
                input: block.input
              };
              globalStreamManager.addToolCall(chatId, toolCall);
              console.log('[Global Stream Manager] Added tool call for', chatId);
            }
          }
        });
      }
    });

    // Global completion handler
    window.claudeAPI.onComplete(async (data) => {
      const { chatId } = data;
      if (!chatId) return;

      const stream = globalStreamManager.getStream(chatId);
      if (!stream) {
        console.warn('[Global Stream Manager] No stream data for', chatId);
        return;
      }

      console.log('[Global Stream Manager] Stream complete for', chatId, '- saving message');

      // Save the message
      if (stream.text || stream.toolCalls.length > 0) {
        const messageToSave = {
          from: 'assistant',
          text: stream.text || ''
        };

        if (stream.toolCalls.length > 0) {
          messageToSave.toolCalls = stream.toolCalls.map(({ name, description, input }) => ({
            name,
            description,
            input
          }));
        }

        try {
          await window.claudeAPI.saveMessage(chatId, messageToSave);
          console.log('[Global Stream Manager] Message saved successfully');
        } catch (error) {
          console.error('[Global Stream Manager] Error saving message:', error);
        }
      }

      // Clear stream data
      globalStreamManager.clearStream(chatId);
    });

    // Global error handler
    window.claudeAPI.onError(async (data) => {
      const { chatId, error } = data;
      if (!chatId) return;

      console.error('[Global Stream Manager] Stream error for', chatId, error);

      const errorMessage = {
        from: 'assistant',
        text: `Error: ${error}`
      };

      try {
        await window.claudeAPI.saveMessage(chatId, errorMessage);
      } catch (err) {
        console.error('[Global Stream Manager] Error saving error message:', err);
      }

      globalStreamManager.clearStream(chatId);
    });
  };

  initGlobalHandlers();
}

export default function App() {
  const [activeId, setActiveId] = useState(null);
  const [dashboards, setDashboards] = useState([]);
  const [projects, setProjects] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const debounceTimers = useRef(new Map());
  const [dashboardVersion, setDashboardVersion] = useState(0);

  // Conversation state: Map of widgetKey -> currentConversationId
  const [widgetConversations, setWidgetConversations] = useState({});

  // Load projects on mount
  useEffect(() => {
    if (window.projectAPI) {
      window.projectAPI.listProjects().then((result) => {
        if (result.success) {
          setProjects(result.projects || []);
        }
      });
    }
  }, []);

  // Function to refresh projects list
  const refreshProjects = async () => {
    if (window.projectAPI) {
      const result = await window.projectAPI.listProjects();
      if (result.success) {
        setProjects(result.projects || []);
      }
    }
  };

  // Function to refresh dashboards list
  const refreshDashboards = async () => {
    if (window.dashboardAPI) {
      const loaded = await window.dashboardAPI.loadDashboards();
      setDashboards(loaded);
    }
  };

  // Load dashboards and set up listener
  useEffect(() => {
    if (window.dashboardAPI) {
      window.dashboardAPI.loadDashboards().then((loaded) => {
        setDashboards(loaded);
        if (loaded.length > 0 && !activeId) {
          setActiveId(loaded[0].id);
        }
        setIsLoading(false);
      });

      window.dashboardAPI.onDashboardUpdate((updated) => {
        setDashboards(updated);
        // Increment version to force reload of all widgets
        setDashboardVersion(v => v + 1);
        // Only set activeId if it's not already set
        setActiveId(prev => {
          if (!prev && updated.length > 0) {
            return updated[0].id;
          }
          return prev;
        });
      });
    }
  }, []);

  const handleWidgetResize = async (dashboardId, widgetId, dimensions) => {
    // Optimistically update local state immediately
    setDashboards(prevDashboards => {
      return prevDashboards.map(dashboard => {
        if (dashboard.id !== dashboardId) return dashboard;

        return {
          ...dashboard,
          widgets: dashboard.widgets.map(widget => {
            if (widget.id !== widgetId) return widget;

            // Update only the properties that changed
            return {
              ...widget,
              ...(dimensions.width !== undefined && { width: dimensions.width }),
              ...(dimensions.height !== undefined && { height: dimensions.height }),
              ...(dimensions.x !== undefined && { x: dimensions.x }),
              ...(dimensions.y !== undefined && { y: dimensions.y })
            };
          })
        };
      });
    });

    if (!window.dashboardAPI) return;

    // Create a unique key for this widget
    const key = `${dashboardId}-${widgetId}`;

    // Clear any existing timer for this widget
    if (debounceTimers.current.has(key)) {
      clearTimeout(debounceTimers.current.get(key));
    }

    // Set a new timer to save after 300ms of inactivity
    const timer = setTimeout(async () => {
      try {
        await window.dashboardAPI.updateWidgetDimensions(dashboardId, widgetId, dimensions);
        debounceTimers.current.delete(key);
      } catch (error) {
        console.error('Failed to update widget dimensions:', error);
      }
    }, 300);

    debounceTimers.current.set(key, timer);
  };

  const handleWidgetDelete = async (dashboardId, widgetId) => {
    // Optimistically update local state immediately
    setDashboards(prevDashboards => {
      return prevDashboards.map(dashboard => {
        if (dashboard.id !== dashboardId) return dashboard;

        return {
          ...dashboard,
          widgets: dashboard.widgets.filter(widget => widget.id !== widgetId)
        };
      });
    });

    if (!window.dashboardAPI) return;

    try {
      await window.dashboardAPI.deleteWidget(dashboardId, widgetId);
    } catch (error) {
      console.error('Failed to delete widget:', error);
      // Reload dashboards to restore state on error
      if (window.dashboardAPI) {
        const loaded = await window.dashboardAPI.loadDashboards();
        setDashboards(loaded);
      }
    }
  };

  const activeDashboard = dashboards.find((d) => d.id === activeId) || dashboards[0];

  // Get theme from active dashboard
  const currentTheme = activeDashboard?.theme && typeof activeDashboard.theme === 'object'
    ? { ...defaultTheme, ...activeDashboard.theme }
    : defaultTheme;

  // Show loading state while initial data is being fetched
  if (isLoading) {
    return <LoadingFallback />;
  }

  // Show no dashboards message only after loading is complete
  if (!activeDashboard) {
    return (
      <div className="app" style={{ color: '#fff', padding: 40, fontFamily: 'system-ui' }}>
        <h2>No dashboards loaded</h2>
        <p>Add a dashboard JSON file using:</p>
        <code style={{ background: '#222', padding: '8px 12px', borderRadius: 4, display: 'block', marginTop: 8 }}>
          window.dashboardAPI.addConfigPath('/path/to/dashboard.json')
        </code>
      </div>
    );
  }

  return (
    <div className="app">
      <Dashboard
        dashboard={activeDashboard}
        allDashboards={dashboards}
        onSelect={setActiveId}
        onWidgetResize={handleWidgetResize}
        onWidgetDelete={handleWidgetDelete}
        widgetConversations={widgetConversations}
        setWidgetConversations={setWidgetConversations}
        dashboardVersion={dashboardVersion}
        onRefreshProjects={refreshProjects}
        onDashboardsChange={refreshDashboards}
      />
    </div>
  );
}
