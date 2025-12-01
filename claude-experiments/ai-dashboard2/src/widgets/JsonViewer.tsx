import { FC, useState } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface JsonViewerConfig {
  id: string;
  type: 'jsonViewer' | 'json-viewer';
  label?: string;
  data?: any;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const JsonViewer: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const jsonConfig = config as JsonViewerConfig;
  const [collapsed, setCollapsed] = useState(new Set<string>());

  const data = jsonConfig.data || {};

  const toggleCollapse = (path: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const renderValue = (value: any, path: string = '', depth: number = 0): JSX.Element => {
    const indent = depth * 16;

    if (value === null) {
      return <span style={{ color: theme.accent, opacity: 0.6 }}>null</span>;
    }

    if (value === undefined) {
      return <span style={{ color: theme.accent, opacity: 0.6 }}>undefined</span>;
    }

    if (typeof value === 'boolean') {
      return <span style={{ color: theme.accent }}>{value.toString()}</span>;
    }

    if (typeof value === 'number') {
      return <span style={{ color: theme.positive }}>{value}</span>;
    }

    if (typeof value === 'string') {
      return <span style={{ color: theme.textColor }}>&quot;{value}&quot;</span>;
    }

    if (Array.isArray(value)) {
      const isCollapsed = collapsed.has(path);
      const isEmpty = value.length === 0;

      return (
        <div>
          <span
            onClick={() => !isEmpty && toggleCollapse(path)}
            style={{
              cursor: isEmpty ? 'default' : 'pointer',
              userSelect: 'none',
              color: theme.textColor,
              opacity: 0.7
            }}
          >
            {!isEmpty && (isCollapsed ? '▶ ' : '▼ ')}
            [
            {isEmpty && ']'}
            {!isEmpty && isCollapsed && ` ... ${value.length} items ]`}
          </span>
          {!isEmpty && !isCollapsed && (
            <div style={{ paddingLeft: 16 }}>
              {value.map((item, idx) => (
                <div key={idx} style={{ marginBottom: 4, fontFamily: theme.textBody }}>
                  <span style={{ color: theme.accent, opacity: 0.5 }}>{idx}: </span>
                  {renderValue(item, `${path}[${idx}]`, depth + 1)}
                  {idx < value.length - 1 && <span style={{ color: theme.textColor, opacity: 0.5 }}>,</span>}
                </div>
              ))}
              <span style={{ color: theme.textColor, opacity: 0.7 }}>]</span>
            </div>
          )}
        </div>
      );
    }

    if (typeof value === 'object') {
      const isCollapsed = collapsed.has(path);
      const keys = Object.keys(value);
      const isEmpty = keys.length === 0;

      return (
        <div>
          <span
            onClick={() => !isEmpty && toggleCollapse(path)}
            style={{
              cursor: isEmpty ? 'default' : 'pointer',
              userSelect: 'none',
              color: theme.textColor,
              opacity: 0.7
            }}
          >
            {!isEmpty && (isCollapsed ? '▶ ' : '▼ ')}
            {'{'}
            {isEmpty && '}'}
            {!isEmpty && isCollapsed && ` ... ${keys.length} keys }`}
          </span>
          {!isEmpty && !isCollapsed && (
            <div style={{ paddingLeft: 16 }}>
              {keys.map((key, idx) => (
                <div key={key} style={{ marginBottom: 4, fontFamily: theme.textBody }}>
                  <span style={{ color: theme.accent }}>&quot;{key}&quot;</span>
                  <span style={{ color: theme.textColor, opacity: 0.5 }}>: </span>
                  {renderValue(value[key], `${path}.${key}`, depth + 1)}
                  {idx < keys.length - 1 && <span style={{ color: theme.textColor, opacity: 0.5 }}>,</span>}
                </div>
              ))}
              <span style={{ color: theme.textColor, opacity: 0.7 }}>{'}'}</span>
            </div>
          )}
        </div>
      );
    }

    return <span style={{ color: theme.textColor }}>{String(value)}</span>;
  };

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>{jsonConfig.label || 'JSON'}</div>
      <div style={{
        fontFamily: 'Monaco, Menlo, "Courier New", monospace',
        fontSize: '0.75rem',
        padding: '8px 12px',
        overflow: 'auto',
        color: theme.textColor,
        lineHeight: 1.6
      }}>
        {renderValue(data)}
      </div>
    </>
  );
};
