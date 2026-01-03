/**
 * TodoList - Displays plan tasks/todos
 */

import React, { memo } from 'react';
import type { UIPlanTask } from '../../../types/acp';

interface TodoListProps {
  todos: UIPlanTask[];
}

// Status icons
const statusIcons: Record<string, string> = {
  pending: '○',
  in_progress: '◐',
  completed: '●',
};

// Status colors
const statusColors: Record<string, string> = {
  pending: 'var(--theme-text-muted)',
  in_progress: 'var(--theme-accent-warning)',
  completed: 'var(--theme-accent-success)',
};

const TodoItem = memo(function TodoItem({ todo }: { todo: UIPlanTask }) {
  const containerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    padding: '6px 0',
  };

  const iconStyle: React.CSSProperties = {
    color: statusColors[todo.status] || '#888',
    fontSize: '14px',
    lineHeight: '1.4',
  };

  const textStyle: React.CSSProperties = {
    flex: 1,
    fontSize: 'var(--theme-font-size-sm)',
    color:
      todo.status === 'completed'
        ? 'var(--theme-text-muted)'
        : 'var(--theme-text-primary)',
    textDecoration: todo.status === 'completed' ? 'line-through' : 'none',
    lineHeight: '1.4',
  };

  const descStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-xs)',
    color: 'var(--theme-text-muted)',
    marginTop: '2px',
  };

  return (
    <div style={containerStyle}>
      <span style={iconStyle}>{statusIcons[todo.status] || '○'}</span>
      <div style={{ flex: 1 }}>
        <div style={textStyle}>{todo.title}</div>
        {todo.description && <div style={descStyle}>{todo.description}</div>}
      </div>
    </div>
  );
});

export const TodoList = memo(function TodoList({ todos }: TodoListProps) {
  const containerStyle: React.CSSProperties = {
    padding: '8px 12px',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '8px',
  };

  const titleStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-sm)',
    fontWeight: 600,
    color: 'var(--theme-text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  };

  const countStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-xs)',
    color: 'var(--theme-text-muted)',
  };

  const listStyle: React.CSSProperties = {
    margin: 0,
    padding: 0,
    listStyle: 'none',
  };

  const completed = todos.filter((t) => t.status === 'completed').length;
  const total = todos.length;

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <span style={titleStyle}>Plan</span>
        <span style={countStyle}>
          {completed}/{total} completed
        </span>
      </div>
      <ul style={listStyle}>
        {todos.map((todo) => (
          <li key={todo.id}>
            <TodoItem todo={todo} />
          </li>
        ))}
      </ul>
    </div>
  );
});
