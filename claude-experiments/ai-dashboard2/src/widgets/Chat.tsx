import { FC, useState, useRef, useEffect } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import {
  globalChatInputs,
  formatToolDescription,
  ChatMessage,
  QuestionPrompt,
  MultiQuestionPrompt
} from './helpers';

interface ChatConfig {
  id: string;
  type: 'chat';
  label: string;
  messages?: Array<{ from: string; text: string; toolCalls?: any[] }>;
  backend?: 'claude' | 'mock' | Function;
  claudeOptions?: Record<string, any>;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const Chat: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboardId, dashboard, widgetKey, currentConversationId, setCurrentConversationId } = props;
  const chatConfig = config as ChatConfig;
  const [messages, setMessages] = useState<any[]>([]);
  // Initialize input from global state if available
  const [input, setInput] = useState(() => globalChatInputs.get(currentConversationId || '') || '');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamText, setCurrentStreamText] = useState(''); // Temporary state for streaming display
  const [currentToolCalls, setCurrentToolCalls] = useState<any[]>([]); // Track active tool calls
  const [isLoading, setIsLoading] = useState(true);
  const [conversations, setConversations] = useState<any[]>([]);
  const [showConversations, setShowConversations] = useState(false);
  const [permissionMode, setPermissionMode] = useState('bypassPermissions'); // 'plan' or 'bypassPermissions', default to bypassPermissions
  const [currentQuestion, setCurrentQuestion] = useState<any>(null); // Active question from AskUserQuestion tool
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const processedTextsRef = useRef(new Set<string>()); // Track processed text content to avoid duplicates
  const processedToolsRef = useRef(new Set<string>()); // Track processed tool calls to avoid duplicates
  const currentStreamTextRef = useRef(''); // Track stream text for saving (ref to avoid closure issues)
  const currentToolCallsRef = useRef<any[]>([]); // Track tool calls for saving (ref to avoid closure issues)
  const hasInitializedRef = useRef(false); // Track if we've already loaded conversations
  const previousMessageCountRef = useRef(0); // Track previous message count to detect new messages
  const hasLoadedInitialMessagesRef = useRef(false); // Track if we've loaded the initial messages

  // Capture dashboard context once at mount - use latest value when needed but don't cause re-renders
  const dashboardContextRef = useRef<any>(null);
  if (dashboard && !dashboardContextRef.current) {
    dashboardContextRef.current = {
      filePath: (dashboard as any)._sourcePath,
      id: dashboard.id,
      title: dashboard.title,
      config: dashboard
    };
  }

  // Determine backend type: 'claude' for Claude Agent SDK, 'mock' for demo, or custom handler
  const backend = chatConfig.backend || 'mock';

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
      if (backend !== 'claude' || !(window as any).claudeAPI) return;

      hasInitializedRef.current = true; // Mark as initialized

      try {
        const result = await (window as any).claudeAPI.listConversations(widgetKey);
        if (result.success) {
          setConversations(result.conversations);

          // Only set conversation if not already set from parent
          if (!currentConversationId) {
            // If no conversations exist, create a default one
            if (result.conversations.length === 0) {
              const createResult = await (window as any).claudeAPI.createConversation(widgetKey, 'New Conversation');
              if (createResult.success) {
                setConversations([createResult.conversation]);
                setCurrentConversationId?.(createResult.conversation.id);
              }
            } else {
              // Load most recent conversation
              setCurrentConversationId?.(result.conversations[0].id);
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

    if (backend === 'claude' && (window as any).claudeAPI) {
      try {
        const result = await (window as any).claudeAPI.getMessages(currentConversationId);
        if (result.success && result.messages) {
          console.log(`[Chat UI] Loaded ${result.messages.length} messages for ${currentConversationId}, streaming: ${result.isStreaming}`);
          console.log('[Chat UI] Messages from backend:', JSON.stringify(result.messages, null, 2));

          // Check if any messages have toolCalls
          const messagesWithTools = result.messages.filter((m: any) => m.toolCalls && m.toolCalls.length > 0);
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
      setMessages(chatConfig.messages || []);
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
  }, [backend, chatConfig.messages, currentConversationId]);

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
    const handleKeyDown = (e: KeyboardEvent) => {
      // Esc to interrupt streaming
      if (e.key === 'Escape' && isStreaming && backend === 'claude' && currentConversationId) {
        console.log('[Chat UI] Interrupting stream...');
        if ((window as any).claudeAPI) {
          (window as any).claudeAPI.interrupt(currentConversationId).then(() => {
            console.log('[Chat UI] Stream interrupted');
            setIsStreaming(false);
            setCurrentStreamText('');
            setCurrentToolCalls([]);
            currentStreamTextRef.current = '';
            currentToolCallsRef.current = [];
          }).catch((err: any) => {
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
    if (backend !== 'claude' || !(window as any).claudeAPI || !currentConversationId) return;

    console.log('[Chat UI] Setting up event listeners for conversationId:', currentConversationId);

    // Listen for message chunks
    const handleMessage = (data: any) => {
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

    const handleComplete = async (data: any) => {
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

    const handleError = async (data: any) => {
      if (data.chatId !== currentConversationId) return;

      console.error('[Chat UI] Error:', data.error);

      // Note: Global stream manager already saved the error message
      // Just reload and clear state
      await reloadMessages();
      setCurrentStreamText('');
      setCurrentToolCalls([]);
      currentStreamTextRef.current = '';
      currentToolCallsRef.current = [];
      processedTextsRef.current.clear();
      processedToolsRef.current.clear();
      setIsStreaming(false);
    };

    const messageHandler = (window as any).claudeAPI.onMessage(handleMessage);
    const completeHandler = (window as any).claudeAPI.onComplete(handleComplete);
    const errorHandler = (window as any).claudeAPI.onError(handleError);

    return () => {
      console.log('[Chat UI] Cleaning up event listeners for conversationId:', currentConversationId);
      // Note: Global stream manager handles saving, even if component unmounts
      // Call cleanup functions directly (they already remove the listeners)
      messageHandler();
      completeHandler();
      errorHandler();
    };
  }, [backend, currentConversationId]);

  // Listen for user questions from AskUserQuestion tool (plan mode only)
  useEffect(() => {
    if (backend !== 'claude' || !(window as any).claudeAPI) return;

    const handleQuestion = (questionData: any) => {
      console.log('[Chat UI] Received question:', questionData);
      setCurrentQuestion(questionData);
      // Scroll to bottom when question appears
      setTimeout(() => {
        if (messagesContainerRef.current) {
          messagesContainerRef.current.scrollTo({
            top: messagesContainerRef.current.scrollHeight,
            behavior: 'smooth'
          });
        }
      }, 100); // Small delay to ensure DOM has updated
    };

    const questionHandler = (window as any).claudeAPI.onUserQuestion(handleQuestion);

    return () => {
      // Call cleanup function directly
      questionHandler();
    };
  }, [backend]);

  // Handle answer submission
  const handleAnswerSubmit = (answer: any) => {
    if (!currentQuestion) return;

    console.log('[Chat UI] Submitting answer:', answer);
    (window as any).claudeAPI.sendQuestionAnswer(currentQuestion.id, answer);
    setCurrentQuestion(null); // Clear the question
  };

  // Helper to save a message to backend
  const saveMessageToBackend = async (message: any) => {
    if (backend === 'claude' && (window as any).claudeAPI && currentConversationId) {
      try {
        // Remove UI-only fields before saving
        const { finalized, ...messageToSave } = message;
        await (window as any).claudeAPI.saveMessage(currentConversationId, messageToSave);

        // Update conversation's lastMessageAt
        await (window as any).claudeAPI.updateConversation(widgetKey, currentConversationId, {
          lastMessageAt: new Date().toISOString()
        });
      } catch (error) {
        console.error('[Chat UI] Error saving message:', error);
      }
    }
  };

  // Create a new conversation
  const handleNewConversation = async () => {
    if (backend !== 'claude' || !(window as any).claudeAPI) return;

    try {
      const result = await (window as any).claudeAPI.createConversation(widgetKey, 'New Conversation');
      if (result.success) {
        setConversations(prev => [result.conversation, ...prev]);
        setCurrentConversationId?.(result.conversation.id);
        setMessages([]);
      }
    } catch (error) {
      console.error('[Chat UI] Error creating conversation:', error);
    }
  };

  // Switch to a different conversation
  const handleSwitchConversation = (conversationId: string) => {
    setCurrentConversationId?.(conversationId);
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

    if (backend === 'claude' && (window as any).claudeAPI) {
      try {
        console.log('[Chat UI] Sending message to Claude...');

        // Prepare options with dashboard context for system prompt
        const options = {
          ...(chatConfig.claudeOptions || {}),
          permissionMode: permissionMode === 'plan' ? 'plan' : 'bypassPermissions',
          dashboardContext: dashboard ? {
            filePath: (dashboard as any)._sourcePath,
            id: dashboard.id,
            title: dashboard.title,
            config: dashboard
          } : null
        };

        // Send to Claude Agent SDK - response will come via event listeners
        const result = await (window as any).claudeAPI.sendMessage(
          currentConversationId,
          messageToSend,
          options
        );

        if (!result.success) {
          throw new Error(result.error);
        }

        console.log('[Chat UI] Message sent successfully, waiting for stream...');
        // Streaming and completion are handled by event listeners
      } catch (error: any) {
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
      } catch (error: any) {
        setMessages(prev => [...prev, {
          from: 'assistant',
          text: `Error: ${error.message}`
        }]);
      }
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
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
          {chatConfig.label}
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
        {isStreaming && !currentStreamText && currentToolCalls.length === 0 && !currentQuestion && (
          <div className="chat-bubble assistant" style={{
            fontFamily: theme.textBody,
            backgroundColor: `${theme.accent}22`,
            borderColor: theme.accent,
            minHeight: '32px'
          }}>
          </div>
        )}
        {currentQuestion && (
          currentQuestion.isMultiple && currentQuestion.questions?.length > 1 ? (
            <MultiQuestionPrompt
              questions={currentQuestion.questions}
              theme={theme}
              onAnswer={handleAnswerSubmit}
            />
          ) : (
            <QuestionPrompt
              question={currentQuestion.questions?.[0] || currentQuestion}
              theme={theme}
              onAnswer={handleAnswerSubmit}
            />
          )
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
};
