/**
 * ToolCallBlock - Displays a tool call with status
 */

import React, { useState, memo } from 'react';
import type { UIToolCall } from '../../../types/acp';

interface ToolCallBlockProps {
  toolCall: UIToolCall;
}

// Icons for tool kinds
const kindIcons: Record<string, string> = {
  read: '📖',
  edit: '✏️',
  delete: '🗑️',
  move: '📁',
  search: '🔍',
  execute: '⚡',
  think: '🤔',
  fetch: '🌐',
  other: '🔧',
};

// Status colors
const statusColors: Record<string, string> = {
  pending: '#888',
  in_progress: '#f9a825',
  completed: '#4caf50',
  failed: '#f44336',
};

export const ToolCallBlock = memo(function ToolCallBlock({ toolCall }: ToolCallBlockProps) {
  const [expanded, setExpanded] = useState(false);

  const containerStyle: React.CSSProperties = {
    backgroundColor: 'var(--theme-bg-tertiary)',
    borderRadius: 'var(--theme-radius-md)',
    border: '1px solid var(--theme-border-primary)',
    marginTop: '8px',
    overflow: 'hidden',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    cursor: 'pointer',
    userSelect: 'none',
  };

  const iconStyle: React.CSSProperties = {
    fontSize: '16px',
  };

  const titleStyle: React.CSSProperties = {
    flex: 1,
    fontSize: 'var(--theme-font-size-sm)',
    fontWeight: 500,
    color: 'var(--theme-text-primary)',
  };

  const statusStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: 'var(--theme-font-size-sm)',
    color: statusColors[toolCall.status] || 'var(--theme-text-muted)',
  };

  const spinnerStyle: React.CSSProperties = {
    width: '12px',
    height: '12px',
    border: '2px solid transparent',
    borderTopColor: statusColors[toolCall.status],
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  };

  const contentStyle: React.CSSProperties = {
    padding: '8px 12px',
    borderTop: '1px solid var(--theme-border-primary)',
    fontSize: 'var(--theme-font-size-sm)',
    fontFamily: 'var(--theme-font-mono)',
    color: 'var(--theme-text-secondary)',
    maxHeight: '200px',
    overflow: 'auto',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
  };

  const locationStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-xs)',
    color: 'var(--theme-text-muted)',
    marginTop: '4px',
    fontFamily: 'var(--theme-font-mono)',
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'pending':
        return 'Pending';
      case 'in_progress':
        return 'Running';
      case 'completed':
        return 'Done';
      case 'failed':
        return 'Failed';
      default:
        return status;
    }
  };

  return (
    <div style={containerStyle}>
      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
      <div style={headerStyle} onClick={() => setExpanded(!expanded)}>
        <span style={iconStyle}>{kindIcons[toolCall.kind] || kindIcons.other}</span>
        <span style={titleStyle}>{toolCall.title}</span>
        <div style={statusStyle}>
          {toolCall.status === 'in_progress' && <div style={spinnerStyle} />}
          <span>{getStatusLabel(toolCall.status)}</span>
        </div>
        <span style={{ fontSize: '10px', color: 'var(--theme-text-muted)' }}>{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div style={contentStyle}>
          {/* Show locations if available */}
          {toolCall.locations && toolCall.locations.length > 0 && (
            <div style={{ marginBottom: '8px' }}>
              <strong>Files:</strong>
              {toolCall.locations.map((loc, i) => (
                <div key={i} style={locationStyle}>
                  {loc.path}
                  {loc.lineRange && `:${loc.lineRange.start}-${loc.lineRange.end}`}
                </div>
              ))}
            </div>
          )}

          {/* Show content if available */}
          {((): React.ReactNode => {
            const content = toolCall.content;
            if (!content || content.length === 0) return null;
            return (
              <div>
                <strong>Content:</strong>
                <pre style={{ margin: '4px 0 0 0' }}>
                  {JSON.stringify(content, null, 2)}
                </pre>
              </div>
            );
          })()}

          {/* Show result if available */}
          {toolCall.result !== undefined && toolCall.result !== null ? (
            <div>
              <strong>Result:</strong>
              <pre style={{ margin: '4px 0 0 0' }}>
                {typeof toolCall.result === 'string'
                  ? toolCall.result
                  : JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          ) : null}

          {/* Show error if failed */}
          {toolCall.error && (
            <div style={{ color: 'var(--theme-accent-error)' }}>
              <strong>Error:</strong> {toolCall.error}
            </div>
          )}

          {/* Show nothing if empty */}
          {!toolCall.locations?.length && !toolCall.content?.length && !toolCall.result && !toolCall.error && (
            <span style={{ fontStyle: 'italic' }}>No details available</span>
          )}
        </div>
      )}
    </div>
  );
});
