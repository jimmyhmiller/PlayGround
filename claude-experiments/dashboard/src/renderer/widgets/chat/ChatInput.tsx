/**
 * ChatInput - Text input for chat messages with image support
 */

import React, { useState, useCallback, useRef, memo } from 'react';

export interface ImageAttachment {
  id: string;
  data: string; // base64
  mimeType: string;
  name?: string;
}

export interface ContentBlock {
  type: 'text' | 'image';
  text?: string;
  data?: string;
  mimeType?: string;
}

interface ChatInputProps {
  onSend: (content: ContentBlock[]) => void;
  onCancel?: () => void;
  onNewSession?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
}

export const ChatInput = memo(function ChatInput({
  onSend,
  onCancel,
  onNewSession,
  isStreaming = false,
  disabled = false,
}: ChatInputProps) {
  const [text, setText] = useState('');
  const [images, setImages] = useState<ImageAttachment[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = useCallback(() => {
    if ((text.trim() || images.length > 0) && !disabled) {
      const content: ContentBlock[] = [];

      // Add images first
      for (const img of images) {
        content.push({
          type: 'image',
          data: img.data,
          mimeType: img.mimeType,
        });
      }

      // Add text if present
      if (text.trim()) {
        content.push({
          type: 'text',
          text: text.trim(),
        });
      }

      onSend(content);
      setText('');
      setImages([]);

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [text, images, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape' && isStreaming && onCancel) {
        e.preventDefault();
        onCancel();
      } else if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit, isStreaming, onCancel]
  );

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }, []);

  // Process image file to base64
  const processImageFile = useCallback((file: File): Promise<ImageAttachment> => {
    return new Promise((resolve, reject) => {
      if (!file.type.startsWith('image/')) {
        reject(new Error('Not an image file'));
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        // Extract base64 data (remove data:image/...;base64, prefix)
        const base64 = dataUrl.split(',')[1] || '';
        resolve({
          id: `img-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          data: base64,
          mimeType: file.type,
          name: file.name,
        });
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }, []);

  // Handle paste events for images
  const handlePaste = useCallback(async (e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith('image/')) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) {
          try {
            const attachment = await processImageFile(file);
            setImages(prev => [...prev, attachment]);
          } catch (err) {
            console.error('Failed to process pasted image:', err);
          }
        }
        break;
      }
    }
  }, [processImageFile]);

  // Handle drag and drop
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const files = e.dataTransfer?.files;
    if (!files) return;

    for (const file of files) {
      if (file.type.startsWith('image/')) {
        try {
          const attachment = await processImageFile(file);
          setImages(prev => [...prev, attachment]);
        } catch (err) {
          console.error('Failed to process dropped image:', err);
        }
      }
    }
  }, [processImageFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  // Handle file input change
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    for (const file of files) {
      if (file.type.startsWith('image/')) {
        try {
          const attachment = await processImageFile(file);
          setImages(prev => [...prev, attachment]);
        } catch (err) {
          console.error('Failed to process selected image:', err);
        }
      }
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [processImageFile]);

  // Remove an image attachment
  const removeImage = useCallback((id: string) => {
    setImages(prev => prev.filter(img => img.id !== id));
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
    cursor: disabled ? 'not-allowed' : 'pointer',
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

  const imagesRowStyle: React.CSSProperties = {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap',
  };

  const imageThumbnailStyle: React.CSSProperties = {
    position: 'relative',
    width: '60px',
    height: '60px',
    borderRadius: 'var(--theme-radius-sm)',
    overflow: 'hidden',
    border: '1px solid var(--theme-border-primary)',
  };

  const removeButtonStyle: React.CSSProperties = {
    position: 'absolute',
    top: '2px',
    right: '2px',
    width: '18px',
    height: '18px',
    borderRadius: '50%',
    backgroundColor: 'rgba(0,0,0,0.6)',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    lineHeight: 1,
  };

  const attachButtonStyle: React.CSSProperties = {
    padding: '10px',
    backgroundColor: 'transparent',
    border: '1px solid var(--theme-border-primary)',
    borderRadius: 'var(--theme-radius-md)',
    color: 'var(--theme-text-muted)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  return (
    <div
      style={containerStyle}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      {/* Image attachments */}
      {images.length > 0 && (
        <div style={imagesRowStyle}>
          {images.map(img => (
            <div key={img.id} style={imageThumbnailStyle}>
              <img
                src={`data:${img.mimeType};base64,${img.data}`}
                alt={img.name || 'Attachment'}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              <button
                style={removeButtonStyle}
                onClick={() => removeImage(img.id)}
                title="Remove image"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div style={inputRowStyle}>
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        {/* Attach button */}
        <button
          style={attachButtonStyle}
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          title="Attach image"
        >
          📎
        </button>

        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={isStreaming ? 'Type to interrupt...' : 'Type a message or paste/drop images...'}
          disabled={disabled}
          style={textareaStyle}
          rows={1}
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || (!text.trim() && images.length === 0)}
          style={buttonStyle}
        >
          Send
        </button>
        {isStreaming && (
          <button onClick={onCancel} style={cancelButtonStyle}>
            Stop
          </button>
        )}
      </div>
      <div style={actionsRowStyle}>
        <span style={hintsStyle}>
          Enter to send, Shift+Enter for new line{isStreaming ? ', Escape to stop' : ''}, paste/drop images
        </span>
        <button onClick={onNewSession} style={secondaryButtonStyle}>
          New Session
        </button>
      </div>
    </div>
  );
});
