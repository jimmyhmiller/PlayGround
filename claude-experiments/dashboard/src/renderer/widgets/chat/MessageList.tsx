/**
 * MessageList - Displays chat messages as plain text
 */

import React from 'react';
import type { ChatMessage } from '../../../types/acp';

interface MessageListProps {
  messages: ChatMessage[];
}

function MessageItem({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  const containerStyle: React.CSSProperties = {
    marginBottom: '16px',
    lineHeight: '1.6',
  };

  const labelStyle: React.CSSProperties = {
    fontWeight: 600,
    color: isUser ? 'var(--theme-accent-primary)' : 'var(--theme-text-muted)',
    marginBottom: '4px',
    fontSize: 'var(--theme-font-size-sm)',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  };

  const contentStyle: React.CSSProperties = {
    color: 'var(--theme-text-primary)',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    fontSize: 'var(--theme-font-size-md)',
  };

  return (
    <div style={containerStyle}>
      <div style={labelStyle}>{isUser ? 'You' : 'Claude'}</div>
      <div style={contentStyle}>{renderMarkdown(message.content)}</div>
    </div>
  );
}

/**
 * Render inline formatting (bold, italic, code, links)
 */
function renderInline(text: string, keyPrefix: string = ''): React.ReactNode {
  if (!text) return null;

  // Process inline code first
  const parts = text.split(/(`[^`]+`)/g);

  return parts.map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code
          key={`${keyPrefix}-code-${i}`}
          style={{
            backgroundColor: 'var(--theme-bg-tertiary)',
            padding: '2px 6px',
            borderRadius: 'var(--theme-radius-sm)',
            fontFamily: 'var(--theme-font-mono)',
            fontSize: 'var(--theme-font-size-sm)',
          }}
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    // Bold and italic
    return part.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).map((segment, j) => {
      if (segment.startsWith('**') && segment.endsWith('**')) {
        return <strong key={`${keyPrefix}-bold-${i}-${j}`}>{segment.slice(2, -2)}</strong>;
      }
      if (segment.startsWith('*') && segment.endsWith('*') && segment.length > 2) {
        return <em key={`${keyPrefix}-em-${i}-${j}`}>{segment.slice(1, -1)}</em>;
      }
      return segment;
    });
  });
}

/**
 * Markdown renderer for chat messages
 */
function renderMarkdown(content: string): React.ReactNode {
  if (!content) return null;

  // Split by code blocks first
  const blocks = content.split(/(```[\s\S]*?```)/g);

  return blocks.map((block, blockIdx) => {
    // Code block
    if (block.startsWith('```')) {
      const match = block.match(/```(\w+)?\n?([\s\S]*?)```/);
      if (match) {
        const [, lang, code] = match;
        return (
          <pre
            key={`block-${blockIdx}`}
            style={{
              backgroundColor: 'var(--theme-bg-tertiary)',
              padding: '10px',
              borderRadius: 'var(--theme-radius-md)',
              overflow: 'auto',
              fontSize: 'var(--theme-font-size-sm)',
              fontFamily: 'var(--theme-font-mono)',
              margin: '8px 0',
            }}
          >
            {lang && (
              <div style={{ fontSize: 'var(--theme-font-size-xs)', color: 'var(--theme-text-muted)', marginBottom: '6px' }}>
                {lang}
              </div>
            )}
            <code>{(code ?? '').trim()}</code>
          </pre>
        );
      }
    }

    // Process lines for headers, lists, etc.
    const lines = block.split('\n');
    const elements: React.ReactNode[] = [];
    let listItems: string[] = [];
    let listType: 'ul' | 'ol' | null = null;

    const flushList = () => {
      if (listItems.length > 0 && listType) {
        const ListTag = listType;
        elements.push(
          <ListTag key={`list-${blockIdx}-${elements.length}`} style={{ margin: '8px 0', paddingLeft: '24px' }}>
            {listItems.map((item, idx) => (
              <li key={idx}>{renderInline(item, `li-${blockIdx}-${elements.length}-${idx}`)}</li>
            ))}
          </ListTag>
        );
        listItems = [];
        listType = null;
      }
    };

    lines.forEach((line, lineIdx) => {
      // Headers
      const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
      if (headerMatch && headerMatch[1] && headerMatch[2]) {
        flushList();
        const level = headerMatch[1].length;
        const text = headerMatch[2];
        const sizes: Record<number, string> = { 1: '1.5em', 2: '1.3em', 3: '1.1em', 4: '1em', 5: '0.9em', 6: '0.85em' };
        elements.push(
          <div
            key={`h-${blockIdx}-${lineIdx}`}
            style={{
              fontSize: sizes[level] || '1em',
              fontWeight: 600,
              margin: '12px 0 8px 0',
            }}
          >
            {renderInline(text, `h-${blockIdx}-${lineIdx}`)}
          </div>
        );
        return;
      }

      // Unordered list
      const ulMatch = line.match(/^[\s]*[-*]\s+(.+)$/);
      if (ulMatch && ulMatch[1]) {
        if (listType !== 'ul') {
          flushList();
          listType = 'ul';
        }
        listItems.push(ulMatch[1]);
        return;
      }

      // Ordered list
      const olMatch = line.match(/^[\s]*\d+\.\s+(.+)$/);
      if (olMatch && olMatch[1]) {
        if (listType !== 'ol') {
          flushList();
          listType = 'ol';
        }
        listItems.push(olMatch[1]);
        return;
      }

      // Regular line
      flushList();
      if (line.trim()) {
        elements.push(
          <span key={`line-${blockIdx}-${lineIdx}`}>
            {renderInline(line, `line-${blockIdx}-${lineIdx}`)}
            {lineIdx < lines.length - 1 && '\n'}
          </span>
        );
      } else if (lineIdx < lines.length - 1) {
        elements.push(<br key={`br-${blockIdx}-${lineIdx}`} />);
      }
    });

    flushList();
    return <React.Fragment key={`block-${blockIdx}`}>{elements}</React.Fragment>;
  });
}

export function MessageList({ messages }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: 'var(--theme-text-muted)',
          fontSize: 'var(--theme-font-size-md)',
        }}
      >
        Start a conversation...
      </div>
    );
  }

  return (
    <div>
      {messages.map((msg) => (
        <MessageItem key={msg.id} message={msg} />
      ))}
    </div>
  );
}
