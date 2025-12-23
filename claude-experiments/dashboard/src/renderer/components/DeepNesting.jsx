import { memo, useRef } from 'react';
import CounterDisplay from './CounterDisplay';
import UserActionDisplay from './UserActionDisplay';

/**
 * Creates artificial deep nesting to prove that
 * event subscriptions don't cause parent re-renders.
 */

const Level = memo(function Level({ depth, children }) {
  const renderCount = useRef(0);
  renderCount.current += 1;

  return (
    <div style={{
      marginLeft: '12px',
      paddingLeft: '12px',
      borderLeft: '2px solid #ddd',
    }}>
      <span style={{ fontSize: '10px', color: '#999', fontFamily: 'monospace' }}>
        Level {depth} (renders: {renderCount.current})
      </span>
      {children}
    </div>
  );
});

function DeepNesting() {
  const renderCount = useRef(0);
  renderCount.current += 1;

  return (
    <div style={{
      background: '#fafafa',
      padding: '15px',
      borderRadius: '8px',
      border: '1px solid #ddd',
    }}>
      <div style={{ marginBottom: '10px' }}>
        <strong>Deep Nesting Test</strong>
        <span style={{ fontSize: '11px', color: '#f57c00', marginLeft: '10px', fontFamily: 'monospace' }}>
          Root renders: {renderCount.current}
        </span>
      </div>
      <p style={{ fontSize: '12px', color: '#666', marginBottom: '15px' }}>
        Each component shows its render count. Only the subscribed component should re-render when its events fire.
      </p>

      <Level depth={1}>
        <Level depth={2}>
          <Level depth={3}>
            <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
              <CounterDisplay />
              <UserActionDisplay />
            </div>
          </Level>
        </Level>
      </Level>
    </div>
  );
}

export default memo(DeepNesting);
