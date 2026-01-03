/**
 * ChatInput - Text input for chat messages
 */

import React, { useState, useCallback, useRef, memo } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  onCancel?: () => void;
  onClear?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
}

export const ChatInput = memo(function ChatInput({
  onSend,
  onCancel,
  onClear,
  isStreaming = false,
  disabled = false,
}: ChatInputProps) {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(() => {
    if (text.trim() && !disabled && !isStreaming) {
      onSend(text.trim());
      setText('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [text, disabled, isStreaming, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }, []);

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  };

  const inputRowStyle: React.CSSProperties = {
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-end',
  };

  const textareaStyle: React.CSSProperties = {
    flex: 1,
    minHeight: '40px',
    maxHeight: '200px',
    padding: '10px 12px',
    backgroundColor: 'var(--theme-bg-secondary)',
    border: '1px solid var(--theme-border-primary)',
    borderRadius: 'var(--theme-radius-md)',
    color: 'var(--theme-text-primary)',
    fontSize: 'var(--theme-font-size-md)',
    fontFamily: 'inherit',
    resize: 'none',
    outline: 'none',
  };

  const buttonStyle: React.CSSProperties = {
    padding: '10px 16px',
    backgroundColor: 'var(--theme-accent-primary)',
    border: 'none',
    borderRadius: 'var(--theme-radius-md)',
    color: '#fff',
    fontSize: 'var(--theme-font-size-md)',
    fontWeight: 500,
    cursor: disabled || isStreaming ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    transition: 'background-color 0.2s',
  };

  const cancelButtonStyle: React.CSSProperties = {
    ...buttonStyle,
    backgroundColor: 'var(--theme-accent-error)',
  };

  const secondaryButtonStyle: React.CSSProperties = {
    padding: '6px 12px',
    backgroundColor: 'transparent',
    border: '1px solid var(--theme-border-primary)',
    borderRadius: 'var(--theme-radius-sm)',
    color: 'var(--theme-text-muted)',
    fontSize: 'var(--theme-font-size-sm)',
    cursor: 'pointer',
  };

  const actionsRowStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  };

  const hintsStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-xs)',
    color: 'var(--theme-text-muted)',
  };

  return (
    <div style={containerStyle}>
      <div style={inputRowStyle}>
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={isStreaming ? 'Waiting for response...' : 'Type a message...'}
          disabled={disabled || isStreaming}
          style={textareaStyle}
          rows={1}
        />
        {isStreaming ? (
          <button onClick={onCancel} style={cancelButtonStyle}>
            Cancel
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={disabled || !text.trim()}
            style={buttonStyle}
          >
            Send
          </button>
        )}
      </div>
      <div style={actionsRowStyle}>
        <span style={hintsStyle}>
          Press Enter to send, Shift+Enter for new line
        </span>
        <button onClick={onClear} style={secondaryButtonStyle}>
          Clear Chat
        </button>
      </div>
    </div>
  );
});
