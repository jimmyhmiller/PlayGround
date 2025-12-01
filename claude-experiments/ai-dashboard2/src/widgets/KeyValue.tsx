import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface KeyValueItem {
  key: string;
  value: string;
}

interface KeyValueConfig {
  id: string;
  type: 'keyValue' | 'key-value';
  label: string;
  items?: KeyValueItem[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const KeyValue: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const kvConfig = config as KeyValueConfig;
  const items = Array.isArray(kvConfig.items) ? kvConfig.items : [];

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {kvConfig.label}
      </div>
      <div className="kv-list">
        {items.length > 0 ? (
          items.map((item, i) => (
            <div
              key={i}
              className="kv-item"
              style={{ fontFamily: theme.textBody }}
            >
              <span className="kv-key" style={{ color: theme.textColor }}>
                {item.key}
              </span>
              <span className="kv-value" style={{ color: theme.accent }}>
                {item.value}
              </span>
            </div>
          ))
        ) : (
          <div
            style={{
              fontFamily: theme.textBody,
              color: theme.neutral,
              fontSize: '0.8rem',
              padding: '12px 0'
            }}
          >
            No items
          </div>
        )}
      </div>
    </>
  );
};
