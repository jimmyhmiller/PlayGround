import { useState, useEffect } from 'react';

function App() {
  const [message, setMessage] = useState('Loading...');
  const [counter, setCounter] = useState(0);

  const fetchMessage = async () => {
    const msg = await window.electronAPI.getMessage();
    setMessage(msg);
  };

  const fetchCounter = async () => {
    const count = await window.electronAPI.getCounter();
    setCounter(count);
  };

  const handleIncrement = async () => {
    const newCount = await window.electronAPI.increment();
    setCounter(newCount);
    // Also refresh the message to show updated counter
    fetchMessage();
  };

  useEffect(() => {
    fetchMessage();
    fetchCounter();
  }, []);

  return (
    <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Electron + Hot Reload Demo</h1>

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
          onClick={fetchMessage}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            cursor: 'pointer',
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px'
          }}
        >
          Refresh Message
        </button>
      </div>

      <div style={{
        background: '#fff3e0',
        padding: '20px',
        borderRadius: '8px',
        fontSize: '14px',
        color: '#666'
      }}>
        <h3>Try Hot Reloading!</h3>
        <p>
          Edit <code>src/main/index.js</code> and change the <code>getMessage()</code>
          function. Then click "Refresh Message" to see the update without restarting!
        </p>
      </div>
    </div>
  );
}

export default App;
