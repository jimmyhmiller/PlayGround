import { memo, useState, useEffect, useRef, useCallback, ReactElement } from 'react';

interface PromptDialogProps {
  isOpen: boolean;
  title: string;
  placeholder?: string;
  defaultValue?: string;
  onSubmit: (value: string) => void;
  onCancel: () => void;
}

/**
 * A simple prompt dialog for text input
 * Replaces window.prompt() which isn't supported in Electron
 */
const PromptDialog = memo(function PromptDialog({
  isOpen,
  title,
  placeholder = '',
  defaultValue = '',
  onSubmit,
  onCancel,
}: PromptDialogProps): ReactElement | null {
  const [value, setValue] = useState(defaultValue);
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset value and focus when dialog opens
  useEffect(() => {
    if (isOpen) {
      setValue(defaultValue);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [isOpen, defaultValue]);

  const handleSubmit = useCallback(() => {
    if (value.trim()) {
      onSubmit(value.trim());
    }
  }, [value, onSubmit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSubmit();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    },
    [handleSubmit, onCancel]
  );

  if (!isOpen) return null;

  return (
    <div
      className="prompt-dialog-overlay"
      onClick={onCancel}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'var(--theme-palette-overlay)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '20vh',
        zIndex: 10001,
        backdropFilter: 'var(--theme-backdrop-blur)',
      }}
    >
      <div
        className="prompt-dialog"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '400px',
          maxWidth: '90vw',
          background: 'var(--theme-palette-bg)',
          border: '1px solid var(--theme-palette-border)',
          borderRadius: 'var(--theme-palette-radius)',
          boxShadow: 'var(--theme-palette-shadow)',
          overflow: 'hidden',
        }}
      >
        <div
          className="prompt-dialog-header"
          style={{
            padding: 'var(--theme-spacing-md)',
            borderBottom: '1px solid var(--theme-palette-border)',
            fontSize: 'var(--theme-font-size-md)',
            fontWeight: 'var(--theme-font-weight-medium)',
            color: 'var(--theme-text-primary)',
          }}
        >
          {title}
        </div>
        <div
          className="prompt-dialog-body"
          style={{
            padding: 'var(--theme-spacing-md)',
          }}
        >
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            style={{
              width: '100%',
              padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
              background: 'var(--theme-palette-input-bg)',
              border: '1px solid var(--theme-palette-input-border)',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'var(--theme-text-primary)',
              fontSize: 'var(--theme-font-size-md)',
              fontFamily: 'var(--theme-font-family)',
              outline: 'none',
            }}
          />
        </div>
        <div
          className="prompt-dialog-footer"
          style={{
            padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
            borderTop: '1px solid var(--theme-palette-border)',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 'var(--theme-spacing-sm)',
          }}
        >
          <button
            onClick={onCancel}
            style={{
              padding: 'var(--theme-spacing-xs) var(--theme-spacing-md)',
              background: 'var(--theme-bg-tertiary)',
              border: '1px solid var(--theme-border-primary)',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'var(--theme-text-secondary)',
              fontSize: 'var(--theme-font-size-sm)',
              fontFamily: 'var(--theme-font-family)',
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            style={{
              padding: 'var(--theme-spacing-xs) var(--theme-spacing-md)',
              background: 'var(--theme-accent-primary)',
              border: 'none',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'white',
              fontSize: 'var(--theme-font-size-sm)',
              fontFamily: 'var(--theme-font-family)',
              cursor: 'pointer',
            }}
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
});

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * A simple confirm dialog
 * Replaces window.confirm() which isn't supported in Electron
 */
export const ConfirmDialog = memo(function ConfirmDialog({
  isOpen,
  title,
  message,
  onConfirm,
  onCancel,
}: ConfirmDialogProps): ReactElement | null {
  const confirmRef = useRef<HTMLButtonElement>(null);

  // Focus confirm button when dialog opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => confirmRef.current?.focus(), 10);
    }
  }, [isOpen]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        onConfirm();
      }
    },
    [onConfirm, onCancel]
  );

  if (!isOpen) return null;

  return (
    <div
      className="confirm-dialog-overlay"
      onClick={onCancel}
      onKeyDown={handleKeyDown}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'var(--theme-palette-overlay)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '20vh',
        zIndex: 10001,
        backdropFilter: 'var(--theme-backdrop-blur)',
      }}
    >
      <div
        className="confirm-dialog"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '400px',
          maxWidth: '90vw',
          background: 'var(--theme-palette-bg)',
          border: '1px solid var(--theme-palette-border)',
          borderRadius: 'var(--theme-palette-radius)',
          boxShadow: 'var(--theme-palette-shadow)',
          overflow: 'hidden',
        }}
      >
        <div
          className="confirm-dialog-header"
          style={{
            padding: 'var(--theme-spacing-md)',
            borderBottom: '1px solid var(--theme-palette-border)',
            fontSize: 'var(--theme-font-size-md)',
            fontWeight: 'var(--theme-font-weight-medium)',
            color: 'var(--theme-text-primary)',
          }}
        >
          {title}
        </div>
        <div
          className="confirm-dialog-body"
          style={{
            padding: 'var(--theme-spacing-md)',
            color: 'var(--theme-text-secondary)',
            fontSize: 'var(--theme-font-size-md)',
          }}
        >
          {message}
        </div>
        <div
          className="confirm-dialog-footer"
          style={{
            padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
            borderTop: '1px solid var(--theme-palette-border)',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: 'var(--theme-spacing-sm)',
          }}
        >
          <button
            onClick={onCancel}
            style={{
              padding: 'var(--theme-spacing-xs) var(--theme-spacing-md)',
              background: 'var(--theme-bg-tertiary)',
              border: '1px solid var(--theme-border-primary)',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'var(--theme-text-secondary)',
              fontSize: 'var(--theme-font-size-sm)',
              fontFamily: 'var(--theme-font-family)',
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            ref={confirmRef}
            onClick={onConfirm}
            style={{
              padding: 'var(--theme-spacing-xs) var(--theme-spacing-md)',
              background: 'var(--theme-accent-error)',
              border: 'none',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'white',
              fontSize: 'var(--theme-font-size-sm)',
              fontFamily: 'var(--theme-font-family)',
              cursor: 'pointer',
            }}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
});

export default PromptDialog;
