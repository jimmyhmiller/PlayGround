import { useState, useRef, useEffect, useMemo, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Grid, GridItem } from './components';
import './styles.css';

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
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div className="kv-list">
        {config.items.map((item, i) => (
          <div key={i} className="kv-item" style={{ fontFamily: theme.textBody }}>
            <span className="kv-key" style={{ color: theme.textColor }}>{item.key}</span>
            <span className="kv-value" style={{ color: theme.accent }}>{item.value}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function LayoutSettings({ theme, config, dashboardId, layout }) {
  const [widgetGap, setWidgetGap] = useState(layout?.widgetGap ?? 10);
  const [buffer, setBuffer] = useState(layout?.buffer ?? 20);

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

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>Layout Settings</div>
      <div style={{ fontFamily: theme.textBody, fontSize: '0.85rem', color: theme.textColor }}>
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
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamText, setCurrentStreamText] = useState(''); // Temporary state for streaming display
  const [currentToolCalls, setCurrentToolCalls] = useState([]); // Track active tool calls
  const [isLoading, setIsLoading] = useState(true);
  const [conversations, setConversations] = useState([]);
  const [showConversations, setShowConversations] = useState(false);
  const [permissionMode, setPermissionMode] = useState('bypassPermissions'); // 'plan' or 'bypassPermissions', default to bypassPermissions
  const messagesEndRef = useRef(null);
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
      await reloadMessages();
      setIsLoading(false);
      hasLoadedInitialMessagesRef.current = true; // Mark as loaded after initial load
    };

    loadMessages();
  }, [backend, config.messages, currentConversationId]);

  // Only scroll when new messages are added or when streaming (not on initial load)
  useEffect(() => {
    const currentCount = messages.length;
    const previousCount = previousMessageCountRef.current;

    // Scroll if: (we've loaded initial messages AND count increased) OR we're streaming
    // Don't scroll on initial load (hasLoadedInitialMessagesRef.current is false during first load)
    if ((hasLoadedInitialMessagesRef.current && currentCount > previousCount) || isStreaming) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }

    // Update the ref for next comparison
    previousMessageCountRef.current = currentCount;
  }, [messages, isStreaming]);

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

      console.log('[Chat UI] Stream complete - saving final message and reloading');
      console.log('[Chat UI] Current stream text length (ref):', currentStreamTextRef.current?.length);
      console.log('[Chat UI] Current stream text (ref):', currentStreamTextRef.current);
      console.log('[Chat UI] Tool calls (ref):', currentToolCallsRef.current);

      // Save the current stream text as the final assistant message with tool calls
      if (currentStreamTextRef.current || currentToolCallsRef.current.length > 0) {
        console.log('[Chat UI] Saving assistant message to backend...');
        const messageToSave = {
          from: 'assistant',
          text: currentStreamTextRef.current || ''
        };

        // Include tool calls if any were made
        if (currentToolCallsRef.current.length > 0) {
          console.log('[Chat UI] Adding toolCalls to message:', currentToolCallsRef.current.length, 'tool calls');
          messageToSave.toolCalls = currentToolCallsRef.current.map(({ name, description, input }) => ({
            name,
            description,
            input
          }));
          console.log('[Chat UI] Message with toolCalls:', JSON.stringify(messageToSave, null, 2));
        }

        await saveMessageToBackend(messageToSave);
        console.log('[Chat UI] Assistant message saved');
      } else {
        console.warn('[Chat UI] No stream text or tool calls to save!');
      }

      // Reload messages from backend (single source of truth)
      console.log('[Chat UI] Reloading messages from backend...');
      await reloadMessages();

      setIsStreaming(false);
      setCurrentStreamText('');
      setCurrentToolCalls([]); // Clear tool calls
      currentStreamTextRef.current = ''; // Reset ref
      currentToolCallsRef.current = []; // Reset tool calls ref
    };

    const handleError = async (data) => {
      if (data.chatId !== currentConversationId) return;

      console.error('[Chat UI] Error:', data.error);

      const errorMessage = {
        from: 'assistant',
        text: `Error: ${data.error}`
      };
      await saveMessageToBackend(errorMessage);
      await reloadMessages();
      setCurrentStreamText('');
      setCurrentToolCalls([]);
      currentStreamTextRef.current = '';
      currentToolCallsRef.current = [];
      setIsStreaming(false);
    };

    const messageHandler = window.claudeAPI.onMessage(handleMessage);
    const completeHandler = window.claudeAPI.onComplete(handleComplete);
    const errorHandler = window.claudeAPI.onError(handleError);

    return () => {
      console.log('[Chat UI] Cleaning up event listeners for conversationId:', currentConversationId);
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
      <div className="chat-messages">
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
  stat: Stat,
  progress: Progress,
  chat: Chat,
  fileList: FileList,
  todoList: TodoList,
  claudeTodos: ClaudeTodoList,
  keyValue: KeyValue,
  layoutSettings: LayoutSettings,
};

function Widget({ theme, config, clipPath, onResize, dashboardId, dashboard, layout, allWidgets, widgetConversations, setWidgetConversations, reloadTrigger }) {
  const Component = widgetComponents[config.type];
  if (!Component) return null;

  const isChat = config.type === 'chat';
  const widgetKey = `${dashboardId}-${config.id}`;

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

  const widgetContent = (
    <div
      className={`widget ${isChat ? 'chat-widget' : ''}`}
      style={{
        background: theme.widgetBg,
        border: theme.widgetBorder,
        borderRadius: theme.widgetRadius,
        clipPath: clipPath,
        width: '100%',
        height: '100%',
      }}
    >
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
    </div>
  );

  // If we have explicit dimensions, make it resizable and draggable with GridItem
  if (config.width || config.height) {
    // Parse width/height - they might be strings like "200px" or numbers
    const parseSize = (size) => {
      if (typeof size === 'string') {
        return parseInt(size.replace('px', ''));
      }
      return size;
    };

    return (
      <GridItem
        x={config.x || 0}
        y={config.y || 0}
        width={parseSize(config.width)}
        height={parseSize(config.height)}
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

  // Otherwise use standard grid positioning (for CSS grid-based layouts)
  return (
    <div style={{ gridArea: config.area }}>
      {widgetContent}
    </div>
  );
}

function ProjectNode({ icon, active, accent, hoverAccent, onClick }) {
  // Support inline SVG string or predefined icon name
  const iconContent = typeof icon === 'string' && icons[icon]
    ? icons[icon]
    : typeof icon === 'string' && icon.trim().startsWith('<svg')
      ? <span dangerouslySetInnerHTML={{ __html: icon }} />
      : icons.square;

  return (
    <div
      className={`project-node ${active ? 'active' : ''}`}
      style={{ '--accent': accent, '--hover-accent': hoverAccent }}
      onClick={onClick}
    >
      {iconContent}
    </div>
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

function Dashboard({ dashboard, allDashboards, onSelect, onWidgetResize, widgetConversations, setWidgetConversations, dashboardVersion }) {
  const theme = dashboard.theme && typeof dashboard.theme === 'object'
    ? { ...defaultTheme, ...dashboard.theme }
    : defaultTheme;
  const { layout } = dashboard;

  // Use dashboardVersion as the reload trigger - this increments whenever any dashboard file changes
  // This will force widgets to reload their data from files
  const reloadTrigger = dashboardVersion;

  const gridStyle = {
    gridTemplateColumns: layout.columns,
    gridTemplateRows: layout.rows,
    gridTemplateAreas: layout.areas,
  };

  // Get grid settings from layout
  const cellSize = layout?.gridSize || 16;
  const gapX = layout?.widgetGap || 10;
  const gapY = layout?.widgetGap || 10;

  return (
    <div className="window-frame" style={{ '--accent': theme.accent }}>
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
              accent={theme.accent}
              hoverAccent={dTheme.accent}
              onClick={() => onSelect(d.id)}
            />
          );
        })}
      </div>
      <div className="dashboard" style={{ backgroundColor: theme.bgApp }}>
        <div className="header">
          <h1 style={{ fontFamily: theme.textHead, color: theme.textColor }}>{dashboard.title}</h1>
          <p style={{ fontFamily: theme.textBody, color: theme.accent }}>{dashboard.subtitle}</p>
        </div>
        <div className="grid" style={gridStyle}>
          {dashboard.widgets.filter(w => !w.width && !w.height).map((widget) => (
            <Widget
              key={`${dashboard.id}-${widget.id}`}
              theme={theme}
              config={widget}
              clipPath={theme.clipPath}
              onResize={onWidgetResize}
              dashboardId={dashboard.id}
              dashboard={dashboard}
              layout={layout}
              allWidgets={dashboard.widgets}
              widgetConversations={widgetConversations}
              setWidgetConversations={setWidgetConversations}
              reloadTrigger={reloadTrigger}
            />
          ))}
        </div>
        <Grid cellSize={cellSize} gapX={gapX} gapY={gapY} width="100%" height="calc(100% - 80px)">
          {dashboard.widgets.filter(w => w.width || w.height).map((widget) => (
            <Widget
              key={`${dashboard.id}-${widget.id}`}
              theme={theme}
              config={widget}
              clipPath={theme.clipPath}
              onResize={onWidgetResize}
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

export default function App() {
  const [activeId, setActiveId] = useState(null);
  const [dashboards, setDashboards] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const debounceTimers = useRef(new Map());
  const [dashboardVersion, setDashboardVersion] = useState(0);

  // Conversation state: Map of widgetKey -> currentConversationId
  const [widgetConversations, setWidgetConversations] = useState({});

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

  const activeDashboard = dashboards.find((d) => d.id === activeId) || dashboards[0];

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
        widgetConversations={widgetConversations}
        setWidgetConversations={setWidgetConversations}
        dashboardVersion={dashboardVersion}
      />
    </div>
  );
}
