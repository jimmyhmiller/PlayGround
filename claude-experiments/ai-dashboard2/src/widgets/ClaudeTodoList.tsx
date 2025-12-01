import { FC, useState, useEffect } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface ClaudeTodoListConfig {
  id: string;
  type: 'claudeTodos' | 'claude-todo-list';
  label?: string;
  chatWidgetKey?: string;
  chatWidgetId?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

interface Todo {
  content: string;
  status: 'pending' | 'in_progress' | 'completed';
  activeForm?: string;
}

export const ClaudeTodoList: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboardId, widgetConversations } = props;
  const todoConfig = config as ClaudeTodoListConfig;
  const [todos, setTodos] = useState<Todo[]>([]);

  // Resolve conversation ID from linked chat widget
  // Support both chatWidgetKey (full key) and chatWidgetId (partial, needs dashboardId)
  let conversationId: string | null = null;

  if (todoConfig.chatWidgetKey) {
    // Use full key directly
    conversationId = widgetConversations?.[todoConfig.chatWidgetKey] || null;
  } else if (todoConfig.chatWidgetId) {
    // First try using chatWidgetId directly as a full key
    conversationId = widgetConversations?.[todoConfig.chatWidgetId] || null;

    // If not found, try with dashboard prefix
    if (!conversationId && dashboardId) {
      const widgetKey = `${dashboardId}-${todoConfig.chatWidgetId}`;
      conversationId = widgetConversations?.[widgetKey] || null;
    }

    // If still not found, search all widget conversations for a key ending with the chatWidgetId
    if (!conversationId && widgetConversations) {
      const matchingKey = Object.keys(widgetConversations).find(key =>
        key.endsWith(todoConfig.chatWidgetId!) || key.endsWith(`-${todoConfig.chatWidgetId}`)
      );
      if (matchingKey) {
        conversationId = widgetConversations[matchingKey];
        console.log('[ClaudeTodoList] Found matching key via search:', matchingKey);
      }
    }
  }

  console.log('[ClaudeTodoList] Dashboard ID:', dashboardId);
  console.log('[ClaudeTodoList] Config chatWidgetId:', todoConfig.chatWidgetId);
  console.log('[ClaudeTodoList] Config chatWidgetKey:', todoConfig.chatWidgetKey);
  console.log('[ClaudeTodoList] All widget conversation keys:', widgetConversations ? Object.keys(widgetConversations) : []);
  console.log('[ClaudeTodoList] Resolved conversation ID:', conversationId);

  // Load initial todos and listen for updates
  useEffect(() => {
    console.log('[ClaudeTodoList] useEffect running!', { conversationId, hasAPI: !!(window as any).claudeAPI });

    if (!conversationId || !(window as any).claudeAPI) {
      console.log('[ClaudeTodoList] Missing conversationId or claudeAPI:', { conversationId, hasAPI: !!(window as any).claudeAPI });
      return;
    }

    console.log('[ClaudeTodoList] Setting up listeners for conversation:', conversationId);

    // Load initial todos
    (window as any).claudeAPI.getTodos(conversationId).then((result: any) => {
      console.log('[ClaudeTodoList] Initial todos loaded:', result);
      if (result.success) {
        setTodos(result.todos || []);
      }
    });

    // Listen for real-time updates
    const handler = (window as any).claudeAPI.onTodoUpdate((data: any) => {
      console.log('[ClaudeTodoList] Received todo update:', data);
      console.log('[ClaudeTodoList] Comparing chatId:', data.chatId, 'with conversationId:', conversationId);
      if (data.chatId === conversationId) {
        console.log('[ClaudeTodoList] Match! Updating todos:', data.todos);
        setTodos(data.todos || []);
      } else {
        console.log('[ClaudeTodoList] No match, ignoring update');
      }
    });

    return () => {
      console.log('[ClaudeTodoList] Cleaning up listeners');
      (window as any).claudeAPI.offTodoUpdate(handler);
    };
  }, [conversationId]);

  // Helper to get status icon and color
  const getStatusDisplay = (status: string) => {
    switch (status) {
      case 'completed':
        return { icon: '✓', color: theme.positive };
      case 'in_progress':
        return { icon: '⋯', color: theme.accent };
      default: // pending
        return { icon: '○', color: theme.textColor };
    }
  };

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {todoConfig.label || 'Agent Tasks'}
      </div>
      <div className="todo-list">
        {todos.length === 0 ? (
          <div style={{
            fontFamily: theme.textBody,
            color: theme.textColor,
            opacity: 0.5,
            fontSize: '0.85rem',
            padding: '8px 0'
          }}>
            No active tasks
          </div>
        ) : (
          todos.map((todo, i) => {
            const { icon, color } = getStatusDisplay(todo.status);
            const displayText = todo.status === 'in_progress' && todo.activeForm
              ? todo.activeForm
              : todo.content;

            return (
              <div key={i} className="todo-item" style={{ fontFamily: theme.textBody, color: theme.textColor }}>
                <span className="todo-check" style={{ color }}>
                  {icon}
                </span>
                <span className={todo.status === 'completed' ? 'todo-done' : ''}>
                  {displayText}
                </span>
              </div>
            );
          })
        )}
      </div>
    </>
  );
};
