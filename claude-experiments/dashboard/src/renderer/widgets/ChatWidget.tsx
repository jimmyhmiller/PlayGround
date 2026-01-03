/**
 * ChatWidget - ACP-integrated chat interface
 *
 * A dashboard widget that provides a chat interface for interacting with
 * claude-code-acp via the Agent Client Protocol.
 */

import React, { useState, useEffect, useRef } from 'react';
import { usePersistentState } from '../hooks/useWidgetState';
import type { ChatMessage, UIPlanTask, UIToolCall, SessionNotification } from '../../types/acp';

// Sub-components
import { MessageList, ChatInput, ToolCallBlock, TodoList, PermissionDialog } from './chat';

interface ChatWidgetProps {
  sessionCwd?: string;
  title?: string;
}

interface PermissionRequest {
  requestId: string;
  toolCallId: string;
  title: string;
  description?: string;
}

/**
 * Main ChatWidget component
 */
export function ChatWidget({
  sessionCwd,
  title = 'Claude Chat',
}: ChatWidgetProps): React.ReactElement {
  // Persistent state (survives dashboard switches)
  const [messages, setMessages] = usePersistentState<ChatMessage[]>('messages', []);
  const [sessionId, setSessionId] = usePersistentState<string | null>('sessionId', null);
  const [mode, setMode] = usePersistentState<'plan' | 'act'>('mode', 'act');
  const [todos, setTodos] = usePersistentState<UIPlanTask[]>('todos', []);

  // Ephemeral state (not persisted)
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentText, setCurrentText] = useState('');
  const [toolCalls, setToolCalls] = useState<Map<string, UIToolCall>>(new Map());
  const [pendingPermission, setPendingPermission] = useState<PermissionRequest | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentMessageIdRef = useRef<string | null>(null);
  // Refs for async closures - needed because handleSend awaits across renders
  const currentTextRef = useRef(currentText);
  const toolCallsRef = useRef(toolCalls);
  const messagesRef = useRef(messages);
  currentTextRef.current = currentText;
  toolCallsRef.current = toolCalls;
  messagesRef.current = messages;

  // Generate unique message ID
  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  // Scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Initialize connection on mount
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        // Check if already connected
        const wasConnected = await window.acpAPI.isConnected();
        if (mounted) setIsConnected(wasConnected);

        if (!wasConnected) {
          // Spawn and initialize - this starts a new agent process
          await window.acpAPI.spawn();
          await window.acpAPI.initialize();
          if (mounted) setIsConnected(true);

          // Agent was just started, so any old sessionId is invalid
          // Clear old messages since Claude won't have that context
          if (mounted && messagesRef.current.length > 0) {
            setMessages([]);
            setTodos([]);
          }

          // Create a fresh session
          const cwd = sessionCwd || process.cwd?.() || '/';
          const result = await window.acpAPI.newSession(cwd);
          if (mounted) setSessionId(result.sessionId);
        } else if (!sessionId) {
          // Already connected but no session - create one
          const cwd = sessionCwd || process.cwd?.() || '/';
          const result = await window.acpAPI.newSession(cwd);
          if (mounted) setSessionId(result.sessionId);
        }
        // If already connected AND we have a sessionId, assume it's valid
      } catch (err) {
        console.error('[ChatWidget] Init error:', err);
        if (mounted) setError((err as Error).message);
      }
    }

    init();

    return () => {
      mounted = false;
    };
  }, [sessionCwd, sessionId, setSessionId]);

  // Subscribe to session updates via event system
  // Note: Skip 'acp.session.update' as it duplicates the specific event types
  useEffect(() => {
    const unsubscribe = window.eventAPI?.subscribe('acp.session.**', (event) => {
      // Skip the generic 'update' event to avoid processing twice
      if (event.type === 'acp.session.update') return;

      const payload = event.payload as SessionNotification & { update: { sessionUpdate: string } };
      if (!payload?.update) return;

      // Only process events for our session
      if (payload.sessionId && payload.sessionId !== sessionId) {
        return;
      }

      const update = payload.update;
      const updateType = update.sessionUpdate;

      switch (updateType) {
        case 'agent_message_chunk': {
          // Content can be a string, ContentBlock, or array of ContentBlocks
          const chunk = update as unknown as { content: unknown };
          let textToAdd = '';

          if (typeof chunk.content === 'string') {
            textToAdd = chunk.content;
          } else if (Array.isArray(chunk.content)) {
            // Array of ContentBlocks
            textToAdd = chunk.content
              .map((block: { type?: string; text?: string }) =>
                block.type === 'text' ? block.text ?? '' : '')
              .join('');
          } else if (chunk.content && typeof chunk.content === 'object') {
            // Single ContentBlock
            const block = chunk.content as { type?: string; text?: string };
            if (block.type === 'text') {
              textToAdd = block.text ?? '';
            }
          }

          if (textToAdd) {
            setCurrentText((prev) => prev + textToAdd);
          }
          break;
        }

        case 'tool_call': {
          const tc = update as unknown as {
            toolCallId: string;
            title: string;
            kind: string;
            status: string;
          };
          setToolCalls((prev) => {
            const next = new Map(prev);
            next.set(tc.toolCallId, {
              toolCallId: tc.toolCallId,
              title: tc.title,
              kind: tc.kind,
              status: tc.status as UIToolCall['status'],
            });
            return next;
          });
          break;
        }

        case 'tool_call_update': {
          const tcu = update as unknown as {
            toolCallId: string;
            status?: string;
            content?: unknown[];
          };
          setToolCalls((prev) => {
            const next = new Map(prev);
            const existing = next.get(tcu.toolCallId);
            if (existing) {
              next.set(tcu.toolCallId, {
                ...existing,
                status: (tcu.status as UIToolCall['status']) || existing.status,
                content: tcu.content || existing.content,
              });
            }
            return next;
          });
          break;
        }

        case 'plan': {
          const plan = update as unknown as { tasks?: Array<{ id: string; title: string; status: string }> };
          if (plan.tasks) {
            setTodos(
              plan.tasks.map((t) => ({
                id: t.id,
                title: t.title,
                status: t.status as UIPlanTask['status'],
              }))
            );
          }
          break;
        }

        case 'current_mode_update': {
          const modeUpdate = update as unknown as { currentModeId: string };
          if (modeUpdate.currentModeId === 'plan' || modeUpdate.currentModeId === 'act') {
            setMode(modeUpdate.currentModeId);
          }
          break;
        }
      }
    });

    return () => unsubscribe?.();
  }, [sessionId, setTodos, setMode]);

  // Subscribe to permission requests
  useEffect(() => {
    const unsubscribe = window.eventAPI?.subscribe('acp.permission.request', (event) => {
      const request = event.payload as PermissionRequest;
      setPendingPermission(request);
    });

    return () => unsubscribe?.();
  }, []);

  // Handle prompt completion - uses refs for async closure safety
  const finishMessage = () => {
    const text = currentTextRef.current;
    const calls = toolCallsRef.current;
    if (text.trim() || calls.size > 0) {
      const newMessage: ChatMessage = {
        id: currentMessageIdRef.current || generateId(),
        role: 'assistant',
        content: text,
        timestamp: Date.now(),
        toolCallIds: Array.from(calls.keys()),
      };
      setMessages((prev) => [...prev, newMessage]);
    }
    setCurrentText('');
    setToolCalls(new Map());
    setIsStreaming(false);
    currentMessageIdRef.current = null;
    scrollToBottom();
  };

  // Send message
  const handleSend = async (text: string) => {
    if (!sessionId || !text.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);
    scrollToBottom();

    // Start streaming
    setIsStreaming(true);
    setCurrentText('');
    setToolCalls(new Map());
    currentMessageIdRef.current = generateId();
    setError(null);

    try {
      const result = await window.acpAPI.prompt(sessionId, text);
      console.log('[ChatWidget] Prompt result:', result);
      finishMessage();
    } catch (err) {
      console.error('[ChatWidget] Prompt error:', err);
      setError((err as Error).message);
      setIsStreaming(false);
    }
  };

  // Cancel current prompt
  const handleCancel = async () => {
    if (!sessionId) return;
    try {
      await window.acpAPI.cancel(sessionId);
      finishMessage();
    } catch (err) {
      console.error('[ChatWidget] Cancel error:', err);
    }
  };

  // Toggle mode
  const handleModeToggle = async () => {
    if (!sessionId) return;
    const newMode = mode === 'plan' ? 'act' : 'plan';
    try {
      await window.acpAPI.setMode(sessionId, newMode);
      setMode(newMode);
    } catch (err) {
      console.error('[ChatWidget] Mode change error:', err);
    }
  };

  // Handle permission response
  const handlePermission = async (outcome: 'allow' | 'deny') => {
    if (!pendingPermission) return;
    try {
      await window.acpAPI.respondToPermission(pendingPermission.requestId, outcome);
    } catch (err) {
      console.error('[ChatWidget] Permission response error:', err);
    }
    setPendingPermission(null);
  };

  // Clear chat
  const handleClear = () => {
    setMessages([]);
    setTodos([]);
    setCurrentText('');
    setToolCalls(new Map());
  };

  // Styles
  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    backgroundColor: 'var(--theme-bg-primary)',
    color: 'var(--theme-text-primary)',
    fontFamily: 'var(--theme-font-family)',
    fontSize: 'var(--theme-font-size-md)',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
    borderBottom: '1px solid var(--theme-border-primary)',
    backgroundColor: 'var(--theme-bg-secondary)',
  };

  const titleStyle: React.CSSProperties = {
    fontWeight: 600,
    fontSize: 'var(--theme-font-size-sm)',
  };

  const statusStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--theme-spacing-sm)',
    fontSize: 'var(--theme-font-size-sm)',
    color: 'var(--theme-text-muted)',
  };

  const dotStyle: React.CSSProperties = {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: isConnected ? 'var(--theme-accent-success)' : 'var(--theme-accent-error)',
  };

  const contentStyle: React.CSSProperties = {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  };

  const messagesContainerStyle: React.CSSProperties = {
    flex: 1,
    overflow: 'auto',
    padding: 'var(--theme-spacing-md)',
  };

  const todoContainerStyle: React.CSSProperties = {
    borderTop: '1px solid var(--theme-border-primary)',
    maxHeight: '150px',
    overflow: 'auto',
  };

  const inputContainerStyle: React.CSSProperties = {
    borderTop: '1px solid var(--theme-border-primary)',
    padding: 'var(--theme-spacing-md)',
  };

  const errorStyle: React.CSSProperties = {
    padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
    backgroundColor: 'rgba(244, 67, 54, 0.1)',
    borderBottom: '1px solid rgba(244, 67, 54, 0.3)',
    color: 'var(--theme-accent-error)',
    fontSize: 'var(--theme-font-size-sm)',
  };

  // Auto-scroll when streaming text updates or messages change
  useEffect(() => {
    scrollToBottom();
  }, [currentText, messages]);

  // Render current streaming content
  const streamingContent = isStreaming && (currentText || toolCalls.size > 0) && (
    <div style={{ marginBottom: '16px', lineHeight: '1.6' }}>
      <div style={{
        fontWeight: 600,
        color: 'var(--theme-text-muted)',
        marginBottom: '4px',
        fontSize: 'var(--theme-font-size-sm)',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        Claude
      </div>
      {currentText && (
        <div style={{ whiteSpace: 'pre-wrap', color: 'var(--theme-text-primary)', fontSize: 'var(--theme-font-size-md)' }}>
          {currentText}
        </div>
      )}
      {Array.from(toolCalls.values()).map((tc) => (
        <ToolCallBlock key={tc.toolCallId} toolCall={tc} />
      ))}
    </div>
  );

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={headerStyle}>
        <span style={titleStyle}>{title}</span>
        <div style={statusStyle}>
          <span>{mode === 'plan' ? 'Plan Mode' : 'Act Mode'}</span>
          <button
            onClick={handleModeToggle}
            style={{
              padding: '2px 8px',
              fontSize: 'var(--theme-font-size-xs)',
              backgroundColor: 'transparent',
              border: '1px solid var(--theme-border-primary)',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'inherit',
              cursor: 'pointer',
            }}
          >
            Toggle
          </button>
          <div style={dotStyle} title={isConnected ? 'Connected' : 'Disconnected'} />
        </div>
      </div>

      {/* Error display */}
      {error && <div style={errorStyle}>{error}</div>}

      {/* Permission dialog */}
      {pendingPermission && (
        <PermissionDialog
          request={pendingPermission}
          onAllow={() => handlePermission('allow')}
          onDeny={() => handlePermission('deny')}
        />
      )}

      {/* Content */}
      <div style={contentStyle}>
        {/* Messages */}
        <div style={messagesContainerStyle}>
          <MessageList messages={messages} />
          {streamingContent}
          <div ref={messagesEndRef} />
        </div>

        {/* Todo list (when in plan mode or has todos) */}
        {todos.length > 0 && (
          <div style={todoContainerStyle}>
            <TodoList todos={todos} />
          </div>
        )}

        {/* Input */}
        <div style={inputContainerStyle}>
          <ChatInput
            onSend={handleSend}
            onCancel={handleCancel}
            onClear={handleClear}
            isStreaming={isStreaming}
            disabled={!isConnected || !sessionId}
          />
        </div>
      </div>
    </div>
  );
}

export default ChatWidget;
