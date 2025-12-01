import { FC, useState } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface ErrorTestConfig {
  id: string;
  type: 'errorTest' | 'error-test';
  label?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const ErrorTest: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const errorConfig = config as ErrorTestConfig;
  const [shouldError, setShouldError] = useState(false);

  if (shouldError) {
    throw new Error('This is a test error thrown by the button!');
  }

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {errorConfig.label || 'Error Test'}
      </div>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '20px',
        padding: '40px',
        fontFamily: theme.textBody,
        color: theme.textColor,
        height: '100%'
      }}>
        <div style={{ fontSize: '3rem' }}>💣</div>
        <div style={{ textAlign: 'center', opacity: 0.8 }}>
          Click the button below to throw an error and test the error boundary
        </div>
        <button
          onClick={() => setShouldError(true)}
          style={{
            padding: '12px 24px',
            backgroundColor: theme.negative || '#ff4757',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontFamily: theme.textBody,
            fontSize: '1rem',
            fontWeight: 'bold',
            transition: 'transform 0.1s',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
          onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
        >
          Throw Error
        </button>
      </div>
    </>
  );
};
