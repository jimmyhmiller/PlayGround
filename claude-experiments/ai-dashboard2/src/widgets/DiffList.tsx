import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

type DiffItem = [string, number, number]; // [filename, added, removed]

interface DiffListConfig {
  id: string;
  type: 'diffList' | 'diff-list';
  label: string;
  items: DiffItem[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const DiffList: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const diffConfig = config as DiffListConfig;

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {diffConfig.label}
      </div>
      {diffConfig.items.map(([file, added, removed], i) => (
        <div
          key={i}
          className="diff-item"
          style={{
            fontFamily: theme.textBody,
            color: theme.textColor
          }}
        >
          <span className="diff-file">{file}</span>
          <span className="diff-stats">
            <span style={{ color: theme.positive }}>+{added}</span>
            <span style={{ color: theme.negative }}>-{removed}</span>
          </span>
        </div>
      ))}
    </>
  );
};
