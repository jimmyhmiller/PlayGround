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
import type { ContentBlock } from './chat/ChatInput';

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
  // Mode type from ACP
  interface SessionMode {
    id: string;
    name: string;
  }

  // Persistent state (survives dashboard switches)
  const [messages, setMessages] = usePersistentState<ChatMessage[]>('messages', []);
  const [sessionId, setSessionId, sessionIdLoaded] = usePersistentState<string | null>('sessionId', null);
  const [currentModeId, setCurrentModeId] = usePersistentState<string | null>('currentModeId', null);
  const [todos, setTodos] = usePersistentState<UIPlanTask[]>('todos', []);
  const [isStreaming, setIsStreaming] = usePersistentState<boolean>('isStreaming', false);
  const [streamingText, setStreamingText] = usePersistentState<string>('streamingText', '');
  const [streamingToolCalls, setStreamingToolCalls] = usePersistentState<Record<string, UIToolCall>>('streamingToolCalls', {});

  // Ephemeral state (not persisted)
  const [isConnected, setIsConnected] = useState(false);

  // Derived state from persisted streaming state
  const currentText = streamingText;
  const setCurrentText = setStreamingText;
  const toolCalls = new Map(Object.entries(streamingToolCalls));
  const setToolCalls = (updater: Map<string, UIToolCall> | ((prev: Map<string, UIToolCall>) => Map<string, UIToolCall>)) => {
    if (typeof updater === 'function') {
      setStreamingToolCalls(prev => {
        const prevMap = new Map(Object.entries(prev));
        const nextMap = updater(prevMap);
        return Object.fromEntries(nextMap);
      });
    } else {
      setStreamingToolCalls(Object.fromEntries(updater));
    }
  };
  const [pendingPermission, setPendingPermission] = useState<PermissionRequest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModes, setAvailableModes] = useState<SessionMode[]>([]);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentMessageIdRef = useRef<string | null>(null);
  const hasInitializedRef = useRef(false);
  const hasScrolledInitiallyRef = useRef(false);
  const currentPromptIdRef = useRef<string | null>(null);
  const isStreamingRef = useRef(isStreaming);
  isStreamingRef.current = isStreaming;
  // Refs for async closures - needed because handleSend awaits across renders
  const currentTextRef = useRef(currentText);
  const toolCallsRef = useRef(toolCalls);
  const messagesRef = useRef(messages);
  currentTextRef.current = currentText;
  toolCallsRef.current = toolCalls;
  messagesRef.current = messages;

  // Generate unique message ID
  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  // Scroll to bottom - instant on initial load, smooth afterwards
  const scrollToBottom = (instant = false) => {
    messagesEndRef.current?.scrollIntoView({ behavior: instant ? 'instant' : 'smooth' });
  };

  // Initialize connection once persistence has loaded
  useEffect(() => {
    // Wait for persistence to load before initializing
    if (!sessionIdLoaded) {
      return;
    }

    // Skip if we've already initialized (prevents re-running when sessionId changes after newSession)
    if (hasInitializedRef.current) {
      return;
    }

    let mounted = true;

    async function init() {
      try {
        // Check if already connected
        const wasConnected = await window.acpAPI.isConnected();
        if (mounted) setIsConnected(wasConnected);

        const cwd = sessionCwd || process.cwd?.() || '/';

        if (!wasConnected) {
          // Spawn and initialize - this starts a new agent process
          await window.acpAPI.spawn();
          await window.acpAPI.initialize();
          if (!mounted) return;
          setIsConnected(true);
        }

        // If we have a persisted sessionId, try to resume it
        if (sessionId) {
          try {
            console.log('[ChatWidget] Resuming existing session:', sessionId);
            const result = await window.acpAPI.resumeSession(sessionId, cwd);
            console.log('[ChatWidget] Session resumed, verifying connection...');

            // Verify the session is actually usable by checking if we're still connected
            // The resume might "succeed" but Claude Code may have exited if session wasn't found
            const stillConnected = await window.acpAPI.isConnected();
            if (!stillConnected) {
              throw new Error('Session resume failed - agent disconnected');
            }

            console.log('[ChatWidget] Session resumed successfully, loading history from disk...');

            // Load conversation history from Claude's local files
            const history = await window.acpAPI.loadSessionHistory(sessionId, cwd);
            if (mounted) {
              setMessages(history as ChatMessage[]);
              setTodos([]); // Todos are not persisted in session files
            }
            console.log('[ChatWidget] Loaded', history.length, 'messages from session history');

            if (mounted && result.modes) {
              setAvailableModes(result.modes.availableModes);

              // If we have a persisted mode that differs from the session mode, restore it
              if (currentModeId && currentModeId !== result.modes.currentModeId) {
                console.log('[ChatWidget] Restoring persisted mode:', currentModeId);
                try {
                  await window.acpAPI.setMode(sessionId, currentModeId);
                  // Keep the persisted mode (don't overwrite with session mode)
                } catch (modeErr) {
                  console.warn('[ChatWidget] Failed to restore mode, using session mode:', modeErr);
                  setCurrentModeId(result.modes.currentModeId);
                }
              } else {
                setCurrentModeId(result.modes.currentModeId);
              }
            }
          } catch (err) {
            console.error('[ChatWidget] Failed to resume session, creating new one:', err);
            // Session resume failed - need to respawn and create a new session
            // Clear old messages since they're not part of this session
            try {
              await window.acpAPI.spawn();
              await window.acpAPI.initialize();
            } catch {
              // May already be connected, ignore
            }
            const result = await window.acpAPI.newSession(cwd);
            if (mounted) {
              setSessionId(result.sessionId);
              setMessages([]); // Clear messages - old session context is gone
              setTodos([]);    // Clear todos too
              if (result.modes) {
                setAvailableModes(result.modes.availableModes);
                setCurrentModeId(result.modes.currentModeId);
              }
            }
          }
        } else {
          // No persisted session - create a new one
          const result = await window.acpAPI.newSession(cwd);
          if (mounted) {
            setSessionId(result.sessionId);
            setMessages([]); // Clear any stale messages
            if (result.modes) {
              setAvailableModes(result.modes.availableModes);
              setCurrentModeId(result.modes.currentModeId);
            }
          }
        }

        // Mark as initialized
        hasInitializedRef.current = true;
      } catch (err) {
        console.error('[ChatWidget] Init error:', err);
        if (mounted) setError((err as Error).message);
      }
    }

    init();

    return () => {
      mounted = false;
    };
  }, [sessionIdLoaded, sessionId, sessionCwd]);

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

      // Ignore streaming events if there's no active prompt (e.g., after cancel)
      // This prevents stale events from overwriting new prompt state
      if (!currentPromptIdRef.current && !isStreamingRef.current) {
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
          setCurrentModeId(modeUpdate.currentModeId);
          break;
        }
      }
    });

    return () => unsubscribe?.();
  }, [sessionId, setTodos, setCurrentModeId]);

  // Subscribe to permission requests via IPC (has requestId needed for response)
  useEffect(() => {
    const unsubscribe = window.acpAPI?.subscribePermissions((rawRequest: unknown) => {
      // Map ACP request structure to our PermissionRequest interface
      const req = rawRequest as {
        requestId: string;
        toolCall?: { title?: string; toolCallId?: string };
      };
      setPendingPermission({
        requestId: req.requestId,
        toolCallId: req.toolCall?.toolCallId ?? '',
        title: req.toolCall?.title ?? 'Unknown Tool',
      });
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
    currentPromptIdRef.current = null;
    scrollToBottom();
  };

  // Send message with content blocks (text and/or images)
  const handleSend = async (content: ContentBlock[]) => {
    if (!sessionId || content.length === 0) return;

    // Generate new prompt ID to filter out stale events
    const newPromptId = `prompt-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    // If currently streaming, cancel first and save current progress
    if (isStreamingRef.current) {
      // Invalidate old prompt immediately
      const oldPromptId = currentPromptIdRef.current;
      currentPromptIdRef.current = null;

      try {
        await window.acpAPI.cancel(sessionId);
      } catch (err) {
        console.error('[ChatWidget] Cancel before send error:', err);
      }

      // Save the interrupted message if there was content
      const text = currentTextRef.current;
      const calls = toolCallsRef.current;
      if (text.trim() || calls.size > 0) {
        const interruptedMessage: ChatMessage = {
          id: oldPromptId || generateId(),
          role: 'assistant',
          content: text + (text.trim() ? '\n\n' : '') + '[Interrupted]',
          timestamp: Date.now(),
          toolCallIds: Array.from(calls.keys()),
        };
        setMessages((prev) => [...prev, interruptedMessage]);
      }
    }

    // Set new prompt ID before clearing state
    currentPromptIdRef.current = newPromptId;

    // Extract text for display (images shown as [Image] placeholder)
    const displayText = content
      .map(block => {
        if (block.type === 'text') return block.text || '';
        if (block.type === 'image') return '[Image]';
        return '';
      })
      .filter(Boolean)
      .join(' ');

    // Add user message
    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: displayText,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);
    scrollToBottom();

    // Start streaming with fresh state
    setIsStreaming(true);
    setCurrentText('');
    setToolCalls(new Map());
    currentMessageIdRef.current = newPromptId;
    setError(null);

    // Convert to ACP content block format
    const acpContent = content.map(block => {
      if (block.type === 'text') {
        return { type: 'text' as const, text: block.text || '' };
      } else {
        return { type: 'image' as const, data: block.data || '', mimeType: block.mimeType || 'image/png' };
      }
    });

    try {
      const result = await window.acpAPI.prompt(sessionId, acpContent);
      console.log('[ChatWidget] Prompt result:', result);
      finishMessage();
    } catch (err) {
      console.error('[ChatWidget] Prompt error:', err);
      const errorMessage = (err as Error).message || '';

      // If the error is "process exited", the session doesn't exist - create a new one
      if (errorMessage.includes('process exited') || errorMessage.includes('Internal error')) {
        console.log('[ChatWidget] Session appears invalid, creating new session...');
        setError('Session expired - creating new session...');
        const previousModeId = currentModeId;

        try {
          // Respawn and create new session
          await window.acpAPI.spawn();
          await window.acpAPI.initialize();
          const cwd = sessionCwd || process.cwd?.() || '/';
          const newSession = await window.acpAPI.newSession(cwd);
          setSessionId(newSession.sessionId);
          if (newSession.modes) {
            setAvailableModes(newSession.modes.availableModes);
            // Restore previous mode if available
            if (previousModeId && newSession.modes.availableModes.some(m => m.id === previousModeId)) {
              if (newSession.modes.currentModeId !== previousModeId) {
                await window.acpAPI.setMode(newSession.sessionId, previousModeId);
              }
              setCurrentModeId(previousModeId);
            } else {
              setCurrentModeId(newSession.modes.currentModeId);
            }
          }
          setError(null);

          // Retry the prompt with the new session
          const result = await window.acpAPI.prompt(newSession.sessionId, acpContent);
          console.log('[ChatWidget] Retry prompt result:', result);
          finishMessage();
        } catch (retryErr) {
          console.error('[ChatWidget] Retry failed:', retryErr);
          setError((retryErr as Error).message);
          setIsStreaming(false);
        }
      } else {
        setError(errorMessage);
        setIsStreaming(false);
      }
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

  // Toggle mode - cycles through available modes
  const handleModeToggle = async () => {
    if (!sessionId || availableModes.length === 0) return;

    // Find current mode index and cycle to next
    const currentIndex = availableModes.findIndex(m => m.id === currentModeId);
    const nextIndex = (currentIndex + 1) % availableModes.length;
    const nextMode = availableModes[nextIndex];

    if (!nextMode) return;

    try {
      await window.acpAPI.setMode(sessionId, nextMode.id);
      setCurrentModeId(nextMode.id);
    } catch (err) {
      console.error('[ChatWidget] Mode change error:', err);
      setError(`Failed to change mode: ${(err as Error).message}`);
    }
  };

  // Handle permission response - optionId should be 'allow', 'allow_always', 'reject', etc.
  const handlePermission = async (optionId: string) => {
    if (!pendingPermission) return;
    try {
      await window.acpAPI.respondToPermission(pendingPermission.requestId, optionId);
    } catch (err) {
      console.error('[ChatWidget] Permission response error:', err);
    }
    setPendingPermission(null);
  };

  // Start a new session
  const handleNewSession = async () => {
    const cwd = sessionCwd || process.cwd?.() || '/';
    // Remember the current mode to restore it in the new session
    const previousModeId = currentModeId;

    try {
      // Create a new session (force=true to always create new, not reuse existing)
      const result = await window.acpAPI.newSession(cwd, undefined, true);
      setSessionId(result.sessionId);
      setMessages([]);
      setTodos([]);
      setCurrentText('');
      setToolCalls(new Map());
      if (result.modes) {
        setAvailableModes(result.modes.availableModes);

        // Restore the previous mode if it's available in the new session
        if (previousModeId && result.modes.availableModes.some(m => m.id === previousModeId)) {
          if (result.modes.currentModeId !== previousModeId) {
            await window.acpAPI.setMode(result.sessionId, previousModeId);
          }
          setCurrentModeId(previousModeId);
        } else {
          setCurrentModeId(result.modes.currentModeId);
        }
      }
      console.log('[ChatWidget] Started new session:', result.sessionId);
    } catch (err) {
      console.error('[ChatWidget] Failed to create new session:', err);
      // Still clear local state even if new session fails
      setMessages([]);
      setTodos([]);
      setCurrentText('');
      setToolCalls(new Map());
    }
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

  // Auto-scroll when streaming text updates, messages change, or tool calls update
  useEffect(() => {
    if (!hasScrolledInitiallyRef.current && messages.length > 0) {
      // First scroll after mount - use instant
      hasScrolledInitiallyRef.current = true;
      scrollToBottom(true);
    } else if (hasScrolledInitiallyRef.current) {
      // Subsequent scrolls - use smooth
      scrollToBottom();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentText, messages, toolCalls]);

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
          {availableModes.length > 0 && (
            <>
              <span>{availableModes.find(m => m.id === currentModeId)?.name ?? 'Unknown Mode'}</span>
              {availableModes.length > 1 && (
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
              )}
            </>
          )}
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
          onDeny={() => handlePermission('reject')}
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
            onNewSession={handleNewSession}
            isStreaming={isStreaming}
            disabled={!isConnected || !sessionId}
          />
        </div>
      </div>
    </div>
  );
}

export default ChatWidget;
