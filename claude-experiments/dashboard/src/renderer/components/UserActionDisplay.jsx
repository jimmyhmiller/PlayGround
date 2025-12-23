import { memo, useRef } from 'react';
import { useLatestEvent } from '../hooks/useEvents';

/**
 * Component that only subscribes to user action events.
 */
const UserActionDisplay = memo(function UserActionDisplay() {
  const renderCount = useRef(0);
  renderCount.current += 1;

  // Only subscribes to user.* events
  const userEvent = useLatestEvent('user.**');

  return (
    <div style={{
      background: '#fce4ec',
      padding: '15px',
      borderRadius: '4px',
      border: '2px solid #c2185b',
    }}>
      <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
        <strong>UserActionDisplay</strong> (subscribes to: <code>user.**</code>)
      </div>
      <div style={{ fontSize: '14px', color: '#c2185b' }}>
        {userEvent ? (
          <>
            <div>{userEvent.type}</div>
            <div style={{ fontFamily: 'monospace', fontSize: '12px' }}>
              {JSON.stringify(userEvent.payload)}
            </div>
          </>
        ) : (
          '—'
        )}
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

export default UserActionDisplay;
