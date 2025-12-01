import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface StatConfig {
  id: string;
  type: 'stat';
  label: string;
  value: string | number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const Stat: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const statConfig = config as StatConfig;

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {statConfig.label}
      </div>
      <div
        className="big-stat"
        style={{
          fontFamily: theme.textHead,
          color: theme.textColor
        }}
      >
        {statConfig.value}
      </div>
    </>
  );
};
