import { useState, useRef, useEffect, useMemo, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Rnd } from 'react-rnd';
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

// Widget Components
function BarChart({ theme, config }) {
  const bars = useMemo(() => Array.from({ length: 30 }, () => Math.floor(Math.random() * 80 + 20)), []);
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        <span>{config.label[0]}</span><span>{config.label[1]}</span>
      </div>
      <div className="chart-container">
        {bars.map((h, i) => (
          <div key={i} className="bar" style={{ height: `${h}%`, backgroundColor: theme.accent, borderRadius: theme.chartRadius }} />
        ))}
      </div>
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
  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{config.label}</div>
      <div className="todo-list">
        {config.items.map((item, i) => (
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
    <div className={`chat-bubble ${msg.from}`} style={{
      fontFamily: theme.textBody,
      backgroundColor: msg.from === 'user' ? 'rgba(255,255,255,0.1)' : `${theme.accent}22`,
      borderColor: msg.from === 'assistant' ? theme.accent : 'transparent',
    }}>
      <ReactMarkdown components={components}>{msg.text}</ReactMarkdown>
    </div>
  );
}

function Chat({ theme, config }) {
  const [messages, setMessages] = useState(config.messages || []);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef(null);
  const chatIdRef = useRef(config.id || `chat-${Date.now()}`);
  const processedTextsRef = useRef(new Set()); // Track processed text content to avoid duplicates
  const currentAssistantMessageRef = useRef(null); // Track the current assistant message being built

  // Determine backend type: 'claude' for Claude Agent SDK, 'mock' for demo, or custom handler
  const backend = config.backend || 'mock';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Set up Claude API listeners for streaming responses
  useEffect(() => {
    if (backend !== 'claude' || !window.claudeAPI) return;

    console.log('[Chat UI] Setting up event listeners for chatId:', chatIdRef.current);

    // Clean up any existing listeners first
    window.claudeAPI.removeAllListeners();

    // Listen for message chunks
    const handleMessage = (data) => {
      if (data.chatId !== chatIdRef.current) return;

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
          // Extract only text blocks and check if we've seen this exact text before
          content.forEach((block) => {
            if (block.type === 'text' && block.text) {
              // Use the actual text content as the key to avoid duplicates
              const textHash = block.text.trim();

              if (!processedTextsRef.current.has(textHash)) {
                console.log('[Chat UI] NEW text block, adding to messages. First 50 chars:', textHash.substring(0, 50));
                processedTextsRef.current.add(textHash);

                // Add as a new message or append to the current assistant message
                setMessages(msgs => {
                  // Check if the last message is from assistant and not finalized
                  const lastMsg = msgs[msgs.length - 1];
                  if (lastMsg && lastMsg.from === 'assistant' && !lastMsg.finalized) {
                    // Append to existing message
                    const updatedMsg = {
                      ...lastMsg,
                      text: lastMsg.text + (lastMsg.text.endsWith(' ') || lastMsg.text.endsWith('\n') ? '' : ' ') + block.text
                    };
                    return [...msgs.slice(0, -1), updatedMsg];
                  } else {
                    // Create new assistant message
                    return [...msgs, {
                      from: 'assistant',
                      text: block.text,
                      finalized: false
                    }];
                  }
                });
              } else {
                console.log('[Chat UI] DUPLICATE text block detected, skipping. First 50 chars:', textHash.substring(0, 50));
              }
            }
          });
        }
      }
    };

    const handleComplete = (data) => {
      if (data.chatId !== chatIdRef.current) return;

      console.log('[Chat UI] Stream complete - marking message as finalized');

      // Mark the last assistant message as finalized
      setMessages(msgs => {
        const lastMsg = msgs[msgs.length - 1];
        if (lastMsg && lastMsg.from === 'assistant' && !lastMsg.finalized) {
          return [...msgs.slice(0, -1), { ...lastMsg, finalized: true }];
        }
        return msgs;
      });
      setIsStreaming(false);
    };

    const handleError = (data) => {
      if (data.chatId !== chatIdRef.current) return;

      console.error('[Chat UI] Error:', data.error);

      setMessages(prev => [...prev, {
        from: 'assistant',
        text: `Error: ${data.error}`
      }]);
      setCurrentStreamText('');
      setIsStreaming(false);
    };

    window.claudeAPI.onMessage(handleMessage);
    window.claudeAPI.onComplete(handleComplete);
    window.claudeAPI.onError(handleError);

    return () => {
      console.log('[Chat UI] Cleaning up event listeners for chatId:', chatIdRef.current);
      window.claudeAPI.removeAllListeners();
    };
  }, [backend]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { from: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    const messageToSend = input;
    setInput('');
    setIsStreaming(true);
    processedTextsRef.current.clear(); // Reset for new message

    if (backend === 'claude' && window.claudeAPI) {
      try {
        console.log('[Chat UI] Sending message to Claude...');

        // Send to Claude Agent SDK - response will come via event listeners
        const result = await window.claudeAPI.sendMessage(
          chatIdRef.current,
          messageToSend,
          config.claudeOptions || {}
        );

        if (!result.success) {
          throw new Error(result.error);
        }

        console.log('[Chat UI] Message sent successfully, waiting for stream...');
        // Streaming and completion are handled by event listeners
      } catch (error) {
        console.error('[Chat UI] Send error:', error);
        setMessages(prev => [...prev, {
          from: 'assistant',
          text: `Error: ${error.message}`
        }]);
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

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {config.label}
        {backend === 'claude' && <span style={{ marginLeft: 8, opacity: 0.5 }}>• Claude</span>}
      </div>
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <ChatMessage key={i} msg={msg} theme={theme} />
        ))}
        {isStreaming && messages.length > 0 && messages[messages.length - 1]?.from === 'user' && (
          <div className="chat-bubble assistant" style={{
            fontFamily: theme.textBody,
            backgroundColor: `${theme.accent}22`,
            borderColor: theme.accent,
            opacity: 0.3,
            minHeight: '32px'
          }}>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-row">
        <input
          type="text"
          className="chat-input"
          style={{ fontFamily: theme.textBody, borderColor: `${theme.accent}44` }}
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
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
  keyValue: KeyValue,
};

function Widget({ theme, config, clipPath, onResize, dashboardId, layout, allWidgets }) {
  const Component = widgetComponents[config.type];
  if (!Component) return null;

  const isChat = config.type === 'chat';
  const gap = layout?.widgetGap || 10; // Default 10px gap between widgets

  // Determine minimum sizes based on widget type
  const getMinSize = () => {
    switch (config.type) {
      case 'chat':
        return { width: 280, height: 300 };
      case 'barChart':
        return { width: 200, height: 150 };
      case 'stat':
        return { width: 150, height: 100 };
      case 'diffList':
      case 'fileList':
      case 'todoList':
      case 'keyValue':
        return { width: 200, height: 150 };
      case 'progress':
        return { width: 200, height: 80 };
      default:
        return { width: 150, height: 100 };
    }
  };

  const minSize = getMinSize();

  // Use explicit dimensions if provided, otherwise default to grid sizing
  const size = {
    width: config.width || 'auto',
    height: config.height || 'auto'
  };

  // Use position if provided
  const position = {
    x: config.x || 0,
    y: config.y || 0
  };

  const handleResizeStop = (e, direction, ref, delta, position) => {
    const newWidth = ref.style.width;
    const newHeight = ref.style.height;

    if (onResize && dashboardId) {
      // Check if the new size causes collision
      const hasCollision = (x, y, w, h) => {
        if (!allWidgets) return false;

        const width = parseInt(w) || 200;
        const height = parseInt(h) || 150;

        return allWidgets.some(widget => {
          if (widget.id === config.id) return false; // Skip self
          if (!widget.x || !widget.y || !widget.width || !widget.height) return false;

          const otherX = widget.x;
          const otherY = widget.y;
          const otherWidth = parseInt(widget.width) || 200;
          const otherHeight = parseInt(widget.height) || 150;

          // Check if rectangles overlap with gap
          return !(
            x + width + gap <= otherX ||
            x >= otherX + otherWidth + gap ||
            y + height + gap <= otherY ||
            y >= otherY + otherHeight + gap
          );
        });
      };

      // If collision, revert to original size
      let finalWidth = newWidth;
      let finalHeight = newHeight;

      if (hasCollision(position.x, position.y, newWidth, newHeight)) {
        finalWidth = config.width;
        finalHeight = config.height;
      }

      onResize(dashboardId, config.id, {
        width: finalWidth,
        height: finalHeight,
        x: position.x,
        y: position.y
      });
    }
  };

  const handleDragStop = (e, d) => {
    if (onResize && dashboardId) {
      const gridSize = layout?.gridSize || 0;

      // Snap to grid on release if grid is enabled
      let snappedX = gridSize > 0 ? Math.round(d.x / gridSize) * gridSize : d.x;
      let snappedY = gridSize > 0 ? Math.round(d.y / gridSize) * gridSize : d.y;

      // Get current widget dimensions
      const currentWidth = parseInt(config.width) || 200;
      const currentHeight = parseInt(config.height) || 150;

      // Check for collisions with other widgets
      const hasCollision = (x, y) => {
        if (!allWidgets) return false;

        return allWidgets.some(w => {
          if (w.id === config.id) return false; // Skip self
          if (!w.x || !w.y || !w.width || !w.height) return false; // Skip non-positioned widgets

          const otherX = w.x;
          const otherY = w.y;
          const otherWidth = parseInt(w.width) || 200;
          const otherHeight = parseInt(w.height) || 150;

          // Check if rectangles overlap with gap
          return !(
            x + currentWidth + gap <= otherX ||
            x >= otherX + otherWidth + gap ||
            y + currentHeight + gap <= otherY ||
            y >= otherY + otherHeight + gap
          );
        });
      };

      // If there's a collision, try to find nearest non-colliding position
      if (hasCollision(snappedX, snappedY)) {
        // Revert to original position
        snappedX = config.x || 0;
        snappedY = config.y || 0;
      }

      onResize(dashboardId, config.id, {
        x: snappedX,
        y: snappedY
      });
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
      <Component theme={theme} config={config} />
    </div>
  );

  // If we have explicit dimensions, make it resizable
  if (config.width || config.height) {
    // Check if grid snapping is enabled
    const gridSize = layout?.gridSize || 0; // 0 means freeform, >0 means snap to grid

    return (
      <Rnd
        size={size}
        position={position}
        onResizeStop={handleResizeStop}
        onDragStop={handleDragStop}
        bounds={{
          left: 0,
          top: 0,
          right: typeof window !== 'undefined' ? window.innerWidth - parseInt(size.width) - 160 : 0,
          bottom: typeof window !== 'undefined' ? window.innerHeight - parseInt(size.height) - 105 : 0
        }}
        minWidth={minSize.width}
        minHeight={minSize.height}
        resizeGrid={gridSize > 0 ? [gridSize, gridSize] : undefined}
        style={{
          gridArea: config.area,
          padding: `${gap / 2}px`,
        }}
        enableResizing={{
          top: true,
          right: true,
          bottom: true,
          left: true,
          topRight: true,
          bottomRight: true,
          bottomLeft: true,
          topLeft: true
        }}
      >
        {widgetContent}
      </Rnd>
    );
  }

  // Otherwise use standard grid positioning
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

function Dashboard({ dashboard, allDashboards, onSelect, onWidgetResize }) {
  const theme = dashboard.theme && typeof dashboard.theme === 'object'
    ? { ...defaultTheme, ...dashboard.theme }
    : defaultTheme;
  const { layout } = dashboard;

  const gridStyle = {
    gridTemplateColumns: layout.columns,
    gridTemplateRows: layout.rows,
    gridTemplateAreas: layout.areas,
  };

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
          {dashboard.widgets.map((widget) => (
            <Widget
              key={widget.id}
              theme={theme}
              config={widget}
              clipPath={theme.clipPath}
              onResize={onWidgetResize}
              dashboardId={dashboard.id}
              layout={layout}
              allWidgets={dashboard.widgets}
            />
          ))}
        </div>
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
        if (updated.length > 0 && !activeId) {
          setActiveId(updated[0].id);
        }
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
      />
    </div>
  );
}
