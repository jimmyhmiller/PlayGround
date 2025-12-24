import { memo, useRef, ReactElement } from 'react';
import { useLatestEvent } from '../hooks/useEvents';

interface CounterPayload {
  newValue: number;
  oldValue: number;
  delta: number;
}

/**
 * Deeply nested component that only subscribes to counter events.
 * Uses memo to prevent re-renders from parent updates.
 */
const CounterDisplay = memo(function CounterDisplay(): ReactElement {
  const renderCount = useRef(0);
  renderCount.current += 1;

  // Only subscribes to data.counter.* events
  const counterEvent = useLatestEvent('data.counter.*');

  return (
    <div style={{
      background: '#e3f2fd',
      padding: '15px',
      borderRadius: '4px',
      border: '2px solid #1976d2',
    }}>
      <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
        <strong>CounterDisplay</strong> (subscribes to: <code>data.counter.*</code>)
      </div>
      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1976d2' }}>
        {counterEvent ? (counterEvent.payload as CounterPayload).newValue : '—'}
      </div>
      <div style={{
        marginTop: '8px',
        fontSize: '11px',
        color: '#f57c00',
        fontFamily: 'monospace',
      }}>
        Render count: {renderCount.current}
      </div>
    </div>
  );
});

export default CounterDisplay;
