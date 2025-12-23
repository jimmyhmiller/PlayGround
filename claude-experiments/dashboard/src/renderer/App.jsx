import { useState, useEffect } from 'react';
import { useEventSubscription, useEmit } from './hooks/useEvents';
import DeepNesting from './components/DeepNesting';
import { ComponentInstancesProvider, DynamicComponentPanel } from './components/ComponentRegistry';
import { ComponentControls } from './components/ComponentControls';
import { registerAllComponents } from './components/registerComponents';

// Register component types on load
registerAllComponents();

function App() {
  const [message, setMessage] = useState('Loading...');
  const [counter, setCounter] = useState(0);

  // Subscribe to all events for the event log
  const events = useEventSubscription('**', { maxEvents: 50 });

  // Get emit function
  const emit = useEmit();

  const fetchMessage = async () => {
    const msg = await window.electronAPI.getMessage();
    setMessage(msg);
  };

  const fetchCounter = async () => {
    const count = await window.electronAPI.getCounter();
    setCounter(count);
  };

  const handleIncrement = async () => {
    // Emit user action event
    emit('user.button.clicked', { buttonId: 'increment' });

    const newCount = await window.electronAPI.increment();
    setCounter(newCount);
    fetchMessage();
  };

  const handleRefresh = async () => {
    emit('user.button.clicked', { buttonId: 'refresh' });
    fetchMessage();
  };

  const handleUserActionOnly = () => {
    // Only emits a user event, no counter change
    emit('user.action.test', { action: 'test', timestamp: Date.now() });
  };

  useEffect(() => {
    fetchMessage();
    fetchCounter();
  }, []);

  return (
    <ComponentInstancesProvider>
      <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif' }}>
        <h1>Electron + Event Sourcing Dashboard</h1>

        <div style={{
          background: '#f0f0f0',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h2>Message from Backend:</h2>
          <p style={{ fontSize: '18px', color: '#333' }}>{message}</p>
        </div>

        <div style={{
          background: '#e8f4e8',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h2>Counter: {counter}</h2>
          <button
            onClick={handleIncrement}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              cursor: 'pointer',
              background: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              marginRight: '10px'
            }}
          >
            Increment
          </button>
          <button
            onClick={handleRefresh}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              cursor: 'pointer',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              marginRight: '10px'
            }}
          >
            Refresh Message
          </button>
          <button
            onClick={handleUserActionOnly}
            style={{
              padding: '10px 20px',
              fontSize: '16px',
              cursor: 'pointer',
              background: '#9c27b0',
              color: 'white',
              border: 'none',
              borderRadius: '4px'
            }}
          >
            User Action Only
          </button>
        </div>

        {/* Dynamic Components Section */}
        <ComponentControls />
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ marginBottom: '15px' }}>Dynamic Components</h3>
          <DynamicComponentPanel />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <DeepNesting />
        </div>

        <div style={{
          background: '#f3e5f5',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h3>Event Log (Live)</h3>
          <p style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
            Subscribed to pattern: <code>**</code> (all events)
          </p>
          <div style={{
            background: '#1e1e1e',
            color: '#d4d4d4',
            padding: '15px',
            borderRadius: '4px',
            fontFamily: 'monospace',
            fontSize: '12px',
            maxHeight: '300px',
            overflow: 'auto'
          }}>
            {events.length === 0 ? (
              <div style={{ color: '#888' }}>Waiting for events...</div>
            ) : (
              events.slice().reverse().map((evt) => (
                <div key={evt.id} style={{ marginBottom: '8px', borderBottom: '1px solid #333', paddingBottom: '8px' }}>
                  <div>
                    <span style={{ color: '#569cd6' }}>{evt.type}</span>
                    <span style={{ color: '#888', marginLeft: '10px' }}>
                      {new Date(evt.timestamp).toLocaleTimeString()}
                    </span>
                    <span style={{ color: '#6a9955', marginLeft: '10px' }}>
                      [{evt.meta.source}]
                    </span>
                  </div>
                  <div style={{ color: '#ce9178', marginTop: '4px' }}>
                    {JSON.stringify(evt.payload)}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div style={{
          background: '#fff3e0',
          padding: '20px',
          borderRadius: '8px',
          fontSize: '14px',
          color: '#666'
        }}>
          <h3>How It Works</h3>
          <p>
            <strong>Event Sourcing:</strong> All actions emit events that flow through the system.
            Components subscribe to event patterns and only re-render when their events fire.
          </p>
          <p style={{ marginTop: '10px' }}>
            <strong>File Watcher:</strong> Load or watch a file to receive <code>file.content.*</code> events.
          </p>
          <p style={{ marginTop: '10px' }}>
            <strong>Git Service:</strong> Refresh or poll git status to receive <code>git.*</code> events.
          </p>
          <p style={{ marginTop: '10px' }}>
            <strong>External Events:</strong> Connect via WebSocket to <code>ws://127.0.0.1:9876</code>
          </p>
        </div>
      </div>
    </ComponentInstancesProvider>
  );
}

export default App;
