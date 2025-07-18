import React, { useState, useCallback } from 'react';
import { ClientNode } from './components/ClientNode';
import { Client, Message, Entity, MessageType } from './types';
import { HLCTimestamp } from './utils/hlc';
import './App.css';

const generateClientId = () => Math.random().toString(36).substr(2, 9);
const generateMessageId = () => Math.random().toString(36).substr(2, 9);

const calculateCircularPositions = (count: number, centerX: number, centerY: number, radius: number) => {
  const positions = [];
  for (let i = 0; i < count; i++) {
    const angle = (i * 2 * Math.PI) / count - Math.PI / 2;
    const x = centerX + radius * Math.cos(angle) - 130;
    const y = centerY + radius * Math.sin(angle) - 120;
    positions.push({ x, y });
  }
  return positions;
};

function App() {
  const [clients, setClients] = useState<Client[]>([
    { id: '1', name: 'Client 1', isOnline: true, clockSkew: 0, messages: [] },
    { id: '2', name: 'Client 2', isOnline: true, clockSkew: 0, messages: [] }
  ]);

  const addClient = () => {
    if (clients.length >= 8) return;
    
    const newClient: Client = {
      id: generateClientId(),
      name: `Client ${clients.length + 1}`,
      isOnline: true,
      clockSkew: 0,
      messages: []
    };
    
    setClients([...clients, newClient]);
  };

  const removeClient = () => {
    if (clients.length <= 2) return;
    setClients(clients.slice(0, -1));
  };

  const handleSendMessage = useCallback((from: string, to: string, timestamp: HLCTimestamp, entityUpdates?: Partial<Entity>, type: MessageType = 'update', fullEntityState?: Entity, messageHistory?: Message[]) => {
    const message: Message = {
      id: generateMessageId(),
      from,
      to: to === 'broadcast' ? 'broadcast' : to,
      timestamp,
      content: `Message at ${timestamp.physical}:${timestamp.logical}`,
      type,
      entityUpdates,
      fullEntityState,
      messageHistory
    };

    setClients(prevClients => {
      const updatedClients = prevClients.map(client => {
        if (client.id === from) {
          // Only store update messages in sender's log
          if (type === 'update') {
            return {
              ...client,
              messages: [...client.messages, message]
            };
          }
        }
        return client;
      });
      
      // Handle different message destinations
      if (to === 'broadcast') {
        // Broadcast to all other online clients
        updatedClients.forEach(client => {
          if (client.id !== from && client.isOnline) {
            setTimeout(() => {
              window.dispatchEvent(new CustomEvent(`message-${client.id}`, { 
                detail: { message } 
              }));
            }, 100);
          }
        });
      } else {
        // Direct message to specific client
        const targetClient = updatedClients.find(c => c.id === to);
        if (targetClient && targetClient.isOnline) {
          setTimeout(() => {
            window.dispatchEvent(new CustomEvent(`message-${to}`, { 
              detail: { message } 
            }));
          }, 100);
        }
      }
      
      return updatedClients;
    });
  }, []);

  const handleReceiveMessage = useCallback((clientId: string, message: Message) => {
    setClients(prevClients => 
      prevClients.map(client => {
        if (client.id === clientId) {
          // Handle sync response with message history
          if (message.type === 'sync-response' && message.messageHistory) {
            // Add all historical messages that we don't already have
            const newMessages = message.messageHistory.filter(histMsg => 
              !client.messages.some(existingMsg => existingMsg.id === histMsg.id)
            );
            
            return {
              ...client,
              messages: [...client.messages, ...newMessages]
            };
          }
          
          // Don't store sync-request or sync-response messages themselves
          if (message.type === 'sync-request' || message.type === 'sync-response') {
            return client;
          }
          
          // Check if message already exists
          const messageExists = client.messages.some(m => m.id === message.id);
          if (messageExists) {
            return client;
          }
          
          return {
            ...client,
            messages: [...client.messages, message]
          };
        }
        return client;
      })
    );
  }, []);

  const updateClockSkew = (clientId: string, skew: number) => {
    setClients(prevClients =>
      prevClients.map(client =>
        client.id === clientId ? { ...client, clockSkew: skew } : client
      )
    );
  };

  const toggleOnline = (clientId: string) => {
    setClients(prevClients => {
      const updatedClients = prevClients.map(client =>
        client.id === clientId ? { ...client, isOnline: !client.isOnline } : client
      );
      
      // If coming online, trigger sync request
      const targetClient = updatedClients.find(c => c.id === clientId);
      if (targetClient && targetClient.isOnline) {
        // Send sync request after a brief delay to ensure state is updated
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent(`request-sync-${clientId}`));
        }, 100);
      }
      
      return updatedClients;
    });
  };

  const positions = calculateCircularPositions(clients.length, 450, 375, 280);

  return (
    <div className="App">
      <h1>Hybrid Logical Clock Simulation</h1>
      
      <div className="controls">
        <button onClick={addClient} disabled={clients.length >= 8}>
          Add Client
        </button>
        <button onClick={removeClient} disabled={clients.length <= 2}>
          Remove Client
        </button>
        <span className="client-count">Clients: {clients.length}</span>
      </div>

      <div className="simulation-container">
        {clients.map((client, index) => (
          <ClientNode
            key={client.id}
            client={client}
            onSendMessage={handleSendMessage}
            onReceiveMessage={handleReceiveMessage}
            allClients={clients}
            position={positions[index]}
            onUpdateClockSkew={updateClockSkew}
            onToggleOnline={toggleOnline}
          />
        ))}
        
        <div className="connection-lines">
          {clients.map((_, i) => 
            clients.slice(i + 1).map((_, j) => {
              const actualJ = i + j + 1;
              return (
                <svg
                  key={`${i}-${actualJ}`}
                  className="connection-line"
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none'
                  }}
                >
                  <line
                    x1={positions[i].x + 130}
                    y1={positions[i].y + 120}
                    x2={positions[actualJ].x + 130}
                    y2={positions[actualJ].y + 120}
                    stroke="#ddd"
                    strokeWidth="1"
                    strokeDasharray="5,5"
                  />
                </svg>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
