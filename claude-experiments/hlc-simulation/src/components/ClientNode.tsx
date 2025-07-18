import React, { useState, useEffect, useCallback } from 'react';
import { HybridLogicalClock, HLCTimestamp } from '../utils/hlc';
import { Message, Client, LWWEntity, Entity, MessageType } from '../types';
import './ClientNode.css';

interface ClientNodeProps {
  client: Client;
  onSendMessage: (from: string, to: string, timestamp: HLCTimestamp, entityUpdates?: Partial<Entity>, type?: MessageType, fullEntityState?: Entity, messageHistory?: Message[]) => void;
  onReceiveMessage: (clientId: string, message: Message) => void;
  allClients: Client[];
  position: { x: number; y: number };
  onUpdateClockSkew: (clientId: string, skew: number) => void;
  onToggleOnline: (clientId: string) => void;
}

export const ClientNode: React.FC<ClientNodeProps> = ({
  client,
  onSendMessage,
  onReceiveMessage,
  allClients,
  position,
  onUpdateClockSkew,
  onToggleOnline
}) => {
  const [hlc] = useState(() => new HybridLogicalClock());
  const [currentTime, setCurrentTime] = useState<HLCTimestamp>(hlc.getTimestamp());
  const [physicalTime, setPhysicalTime] = useState(Date.now());
  const [showLogs, setShowLogs] = useState(false);
  const [skewValue, setSkewValue] = useState(0);
  const [skewUnit, setSkewUnit] = useState<'ms' | 's' | 'm' | 'h' | 'd'>('s');
  const [lwwEntity] = useState(() => new LWWEntity(client.id, hlc.getTimestamp()));
  const [entityState, setEntityState] = useState<Entity>(lwwEntity.getEntity());
  const [nameInput, setNameInput] = useState('');
  const [foodInput, setFoodInput] = useState('');
  
  const convertToMs = (value: number, unit: 'ms' | 's' | 'm' | 'h' | 'd'): number => {
    switch (unit) {
      case 'ms': return value;
      case 's': return value * 1000;
      case 'm': return value * 60 * 1000;
      case 'h': return value * 60 * 60 * 1000;
      case 'd': return value * 24 * 60 * 60 * 1000;
      default: return value;
    }
  };

  const convertFromMs = (ms: number, unit: 'ms' | 's' | 'm' | 'h' | 'd'): number => {
    switch (unit) {
      case 'ms': return ms;
      case 's': return ms / 1000;
      case 'm': return ms / (60 * 1000);
      case 'h': return ms / (60 * 60 * 1000);
      case 'd': return ms / (24 * 60 * 60 * 1000);
      default: return ms;
    }
  };

  useEffect(() => {
    hlc.setClockSkew(client.clockSkew);
    setSkewValue(convertFromMs(client.clockSkew, skewUnit));
  }, [client.clockSkew, hlc, skewUnit]);

  useEffect(() => {
    const interval = setInterval(() => {
      setPhysicalTime(hlc.getPhysicalTime());
      setCurrentTime(hlc.getTimestamp());
    }, 100);
    return () => clearInterval(interval);
  }, [hlc]);

  const handleSendMessage = () => {
    if (!client.isOnline) return;
    
    const timestamp = hlc.tick();
    setCurrentTime(timestamp);
    
    // Update local entity with current inputs
    const updates: Partial<Entity> = {};
    
    if (nameInput.trim()) {
      lwwEntity.updateAttribute('name', nameInput, timestamp, client.id);
      updates.name = {
        value: nameInput,
        timestamp,
        author: client.id
      };
    }
    
    if (foodInput.trim()) {
      lwwEntity.updateAttribute('favoriteFood', foodInput, timestamp, client.id);
      updates.favoriteFood = {
        value: foodInput,
        timestamp,
        author: client.id
      };
    }
    
    setEntityState(lwwEntity.getEntity());
    
    if (Object.keys(updates).length > 0) {
      onSendMessage(client.id, 'broadcast', timestamp, updates);
    }
  };

  const handleReceiveMessage = useCallback((message: Message) => {
    if (!client.isOnline) return;
    
    const newTimestamp = hlc.update(message.timestamp);
    setCurrentTime(newTimestamp);
    
    // Handle different message types
    if (message.type === 'sync-request' && message.from !== client.id) {
      // Respond with our full entity state and message history
      const syncTimestamp = hlc.tick();
      setCurrentTime(syncTimestamp);
      const updateMessages = client.messages.filter(m => m.type === 'update');
      onSendMessage(client.id, message.from, syncTimestamp, undefined, 'sync-response', lwwEntity.getEntity(), updateMessages);
    } else if (message.type === 'sync-response') {
      // Replay messages from sync response to maintain identical logs
      if (message.messageHistory) {
        message.messageHistory.forEach(historicalMessage => {
          // Only process messages we don't already have
          const messageExists = client.messages.some(m => m.id === historicalMessage.id);
          if (!messageExists && historicalMessage.entityUpdates) {
            // Apply the entity updates from the historical message
            lwwEntity.merge(historicalMessage.entityUpdates);
            
            // Add the historical message to our logs with proper received timestamp
            const receivedTimestamp = hlc.update(historicalMessage.timestamp);
            onReceiveMessage(client.id, { 
              ...historicalMessage, 
              receivedAt: receivedTimestamp 
            });
          }
        });
      }
      setEntityState(lwwEntity.getEntity());
    } else if (message.type === 'update' && message.entityUpdates) {
      // Apply entity updates
      lwwEntity.merge(message.entityUpdates);
      setEntityState(lwwEntity.getEntity());
    }
    
    // Only call onReceiveMessage for messages that should be logged
    if (message.type !== 'sync-request') {
      onReceiveMessage(client.id, { ...message, receivedAt: newTimestamp });
    }
  }, [client.isOnline, client.id, client.messages, hlc, onReceiveMessage, lwwEntity, onSendMessage]);

  useEffect(() => {
    const handleMessage = (event: CustomEvent<{ message: Message }>) => {
      if (event.detail.message.from !== client.id) {
        handleReceiveMessage(event.detail.message);
      }
    };
    
    const handleSyncRequest = () => {
      if (client.isOnline) {
        // Send sync request to all other nodes
        const syncTimestamp = hlc.tick();
        setCurrentTime(syncTimestamp);
        onSendMessage(client.id, 'broadcast', syncTimestamp, undefined, 'sync-request');
      }
    };
    
    window.addEventListener(`message-${client.id}` as any, handleMessage as any);
    window.addEventListener(`request-sync-${client.id}` as any, handleSyncRequest as any);
    
    return () => {
      window.removeEventListener(`message-${client.id}` as any, handleMessage as any);
      window.removeEventListener(`request-sync-${client.id}` as any, handleSyncRequest as any);
    };
  }, [client.id, client.isOnline, handleReceiveMessage, hlc, onSendMessage]);

  const formatTimestamp = (ts: HLCTimestamp) => {
    const date = new Date(ts.physical).toISOString();
    return `${date}.${ts.logical}`;
  };

  const formatPhysicalTime = (time: number) => {
    return new Date(time).toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  };

  return (
    <div 
      className={`client-node ${!client.isOnline ? 'offline' : ''}`}
      style={{ left: position.x, top: position.y }}
    >
      <div className="client-header">
        <h3>{client.name}</h3>
        <button 
          className={`status-btn ${client.isOnline ? 'online' : 'offline'}`}
          onClick={() => onToggleOnline(client.id)}
        >
          {client.isOnline ? '🟢' : '🔴'}
        </button>
      </div>
      
      <div className="clock-info">
        <div>Physical: {formatPhysicalTime(physicalTime)}</div>
        <div>HLC: {formatTimestamp(currentTime)}</div>
        <div className="clock-skew">
          <label>Skew: </label>
          <input
            type="number"
            value={skewValue}
            onChange={(e) => {
              const value = parseFloat(e.target.value) || 0;
              setSkewValue(value);
              onUpdateClockSkew(client.id, convertToMs(value, skewUnit));
            }}
            disabled={!client.isOnline}
          />
          <select
            value={skewUnit}
            onChange={(e) => {
              const newUnit = e.target.value as 'ms' | 's' | 'm' | 'h' | 'd';
              setSkewUnit(newUnit);
              onUpdateClockSkew(client.id, convertToMs(skewValue, newUnit));
            }}
            disabled={!client.isOnline}
          >
            <option value="ms">ms</option>
            <option value="s">sec</option>
            <option value="m">min</option>
            <option value="h">hr</option>
            <option value="d">day</option>
          </select>
        </div>
      </div>

      <div className="entity-section">
        <div className="entity-state">
          <h4>Entity State</h4>
          <div className="entity-field">
            <span className="field-label">Name:</span>
            <span className="field-value">{entityState.name.value || '(empty)'}</span>
            <span className="field-meta">by {entityState.name.author}</span>
          </div>
          <div className="entity-field">
            <span className="field-label">Food:</span>
            <span className="field-value">{entityState.favoriteFood.value || '(empty)'}</span>
            <span className="field-meta">by {entityState.favoriteFood.author}</span>
          </div>
        </div>
        
        <div className="entity-inputs">
          <input
            type="text"
            placeholder="Name"
            value={nameInput}
            onChange={(e) => setNameInput(e.target.value)}
            disabled={!client.isOnline}
          />
          <input
            type="text"
            placeholder="Favorite Food"
            value={foodInput}
            onChange={(e) => setFoodInput(e.target.value)}
            disabled={!client.isOnline}
          />
        </div>
      </div>

      <div className="message-controls">
        <button 
          onClick={handleSendMessage} 
          disabled={!client.isOnline || (!nameInput.trim() && !foodInput.trim())}
          className="broadcast-btn"
        >
          Broadcast Updates
        </button>
      </div>

      <div className="logs-section">
        <button onClick={() => setShowLogs(!showLogs)}>
          {showLogs ? 'Hide' : 'Show'} Logs ({client.messages.length})
        </button>
        
        {showLogs && (
          <div className="message-logs">
            {client.messages.map((msg, idx) => (
              <div key={idx} className={`log-entry ${msg.from === client.id ? 'sent' : 'received'}`}>
                <div className="log-header">
                  {msg.from === client.id ? 'Broadcast' : `From ${msg.from}`}
                  {msg.type !== 'update' && ` (${msg.type})`}
                </div>
                <div className="log-timestamp">
                  Sent: {formatTimestamp(msg.timestamp)}
                  {msg.receivedAt && (
                    <div>Received: {formatTimestamp(msg.receivedAt)}</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};