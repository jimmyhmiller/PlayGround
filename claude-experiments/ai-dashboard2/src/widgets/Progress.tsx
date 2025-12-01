import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface ProgressConfig {
  id: string;
  type: 'progress';
  label: string;
  value: number;
  text?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const Progress: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const progressConfig = config as ProgressConfig;

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {progressConfig.label}
      </div>
      {progressConfig.text && (
        <div style={{
          marginTop: 10,
          fontFamily: theme.textBody,
          fontSize: '0.9rem',
          color: theme.textColor
        }}>
          {progressConfig.text}
        </div>
      )}
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{
            width: `${progressConfig.value}%`,
            backgroundColor: theme.accent
          }}
        />
      </div>
    </>
  );
};
