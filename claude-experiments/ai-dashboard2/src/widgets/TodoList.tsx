import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface TodoItem {
  text: string;
  done: boolean;
}

interface TodoListConfig {
  id: string;
  type: 'todoList' | 'todo-list';
  label: string;
  items?: TodoItem[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const TodoList: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const todoConfig = config as TodoListConfig;
  const items = todoConfig.items || [];

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {todoConfig.label}
      </div>
      <div className="todo-list">
        {items.map((item, i) => (
          <div
            key={i}
            className="todo-item"
            style={{
              fontFamily: theme.textBody,
              color: theme.textColor
            }}
          >
            <span
              className="todo-check"
              style={{
                color: item.done ? theme.positive : theme.accent
              }}
            >
              {item.done ? '✓' : '○'}
            </span>
            <span className={item.done ? 'todo-done' : ''}>
              {item.text}
            </span>
          </div>
        ))}
      </div>
    </>
  );
};
