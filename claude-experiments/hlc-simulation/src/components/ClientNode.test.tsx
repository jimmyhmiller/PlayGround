import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ClientNode } from './ClientNode';
import { Client, Message } from '../types';

// Mock the HLC utility
jest.mock('../utils/hlc', () => ({
  HybridLogicalClock: jest.fn().mockImplementation(() => {
    let timestamp = { physical: 1000, logical: 0 };
    return {
      getTimestamp: jest.fn(() => ({ ...timestamp })),
      tick: jest.fn(() => {
        timestamp = { physical: timestamp.physical + 1, logical: 0 };
        return { ...timestamp };
      }),
      update: jest.fn((remoteTimestamp) => {
        timestamp = { physical: Math.max(timestamp.physical, remoteTimestamp.physical) + 1, logical: 0 };
        return { ...timestamp };
      }),
      getPhysicalTime: jest.fn(() => timestamp.physical),
      setClockSkew: jest.fn()
    };
  })
}));

// Mock the LWWEntity
jest.mock('../types/entity', () => ({
  LWWEntity: jest.fn().mockImplementation(() => ({
    getEntity: jest.fn().mockReturnValue({
      name: { value: '', timestamp: { physical: 1000, logical: 0 }, author: 'test' },
      favoriteFood: { value: '', timestamp: { physical: 1000, logical: 0 }, author: 'test' }
    }),
    updateAttribute: jest.fn().mockReturnValue(true),
    merge: jest.fn()
  }))
}));

describe('ClientNode Synchronization', () => {
  const mockClient: Client = {
    id: 'client1',
    name: 'Test Client',
    isOnline: true,
    clockSkew: 0,
    messages: []
  };

  const mockOtherClient: Client = {
    id: 'client2',
    name: 'Other Client',
    isOnline: true,
    clockSkew: 0,
    messages: []
  };

  const mockPosition = { x: 100, y: 100 };
  const mockOnSendMessage = jest.fn();
  const mockOnReceiveMessage = jest.fn();
  const mockOnUpdateClockSkew = jest.fn();
  const mockOnToggleOnline = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('should send sync request when coming online', async () => {
    const offlineClient = { ...mockClient, isOnline: false };
    
    render(
      <ClientNode
        client={offlineClient}
        onSendMessage={mockOnSendMessage}
        onReceiveMessage={mockOnReceiveMessage}
        allClients={[offlineClient, mockOtherClient]}
        position={mockPosition}
        onUpdateClockSkew={mockOnUpdateClockSkew}
        onToggleOnline={mockOnToggleOnline}
      />
    );

    // Simulate the node coming online by dispatching the sync request event
    const syncEvent = new CustomEvent('request-sync-client1');
    window.dispatchEvent(syncEvent);

    await waitFor(() => {
      expect(mockOnSendMessage).toHaveBeenCalledWith(
        'client1',
        'broadcast',
        { physical: 1001, logical: 0 },
        undefined,
        'sync-request'
      );
    });
  });

  test('should replay messages from sync response', async () => {
    const clientWithMessages = {
      ...mockClient,
      messages: [
        {
          id: 'msg1',
          from: 'client2',
          to: 'broadcast',
          timestamp: { physical: 900, logical: 0 },
          content: 'test message',
          type: 'update' as const,
          entityUpdates: {
            name: { value: 'John', timestamp: { physical: 900, logical: 0 }, author: 'client2' }
          }
        }
      ]
    };

    render(
      <ClientNode
        client={clientWithMessages}
        onSendMessage={mockOnSendMessage}
        onReceiveMessage={mockOnReceiveMessage}
        allClients={[clientWithMessages, mockOtherClient]}
        position={mockPosition}
        onUpdateClockSkew={mockOnUpdateClockSkew}
        onToggleOnline={mockOnToggleOnline}
      />
    );

    // Simulate receiving a sync response with message history
    const syncResponseMessage: Message = {
      id: 'sync-resp-1',
      from: 'client2',
      to: 'client1',
      timestamp: { physical: 1100, logical: 0 },
      content: 'sync response',
      type: 'sync-response',
      messageHistory: [
        {
          id: 'msg2', // Different message ID that client1 doesn't have
          from: 'client2',
          to: 'broadcast',
          timestamp: { physical: 950, logical: 0 },
          content: 'missed message',
          type: 'update',
          entityUpdates: {
            favoriteFood: { value: 'Pizza', timestamp: { physical: 950, logical: 0 }, author: 'client2' }
          }
        }
      ]
    };

    const messageEvent = new CustomEvent('message-client1', { 
      detail: { message: syncResponseMessage } 
    });
    window.dispatchEvent(messageEvent);

    await waitFor(() => {
      // Should call onReceiveMessage with the replayed message
      expect(mockOnReceiveMessage).toHaveBeenCalledWith(
        'client1',
        expect.objectContaining({
          id: 'msg2',
          from: 'client2',
          type: 'update',
          receivedAt: { physical: 1002, logical: 0 }
        })
      );
    });
  });

  test('should not replay messages that already exist', async () => {
    const clientWithMessages = {
      ...mockClient,
      messages: [
        {
          id: 'existing-msg',
          from: 'client2',
          to: 'broadcast',
          timestamp: { physical: 900, logical: 0 },
          content: 'existing message',
          type: 'update' as const,
          entityUpdates: {
            name: { value: 'John', timestamp: { physical: 900, logical: 0 }, author: 'client2' }
          }
        }
      ]
    };

    render(
      <ClientNode
        client={clientWithMessages}
        onSendMessage={mockOnSendMessage}
        onReceiveMessage={mockOnReceiveMessage}
        allClients={[clientWithMessages, mockOtherClient]}
        position={mockPosition}
        onUpdateClockSkew={mockOnUpdateClockSkew}
        onToggleOnline={mockOnToggleOnline}
      />
    );

    // Simulate receiving a sync response with a message that already exists
    const syncResponseMessage: Message = {
      id: 'sync-resp-2',
      from: 'client2',
      to: 'client1',
      timestamp: { physical: 1100, logical: 0 },
      content: 'sync response',
      type: 'sync-response',
      messageHistory: [
        {
          id: 'existing-msg', // Same message ID that client1 already has
          from: 'client2',
          to: 'broadcast',
          timestamp: { physical: 900, logical: 0 },
          content: 'existing message',
          type: 'update',
          entityUpdates: {
            name: { value: 'John', timestamp: { physical: 900, logical: 0 }, author: 'client2' }
          }
        }
      ]
    };

    const messageEvent = new CustomEvent('message-client1', { 
      detail: { message: syncResponseMessage } 
    });
    window.dispatchEvent(messageEvent);

    await waitFor(() => {
      // Should NOT call onReceiveMessage for existing message
      expect(mockOnReceiveMessage).not.toHaveBeenCalledWith(
        'client1',
        expect.objectContaining({
          id: 'existing-msg'
        })
      );
    });
  });

  test('should respond to sync requests with message history', async () => {
    const clientWithMessages = {
      ...mockClient,
      messages: [
        {
          id: 'msg1',
          from: 'client1',
          to: 'broadcast',
          timestamp: { physical: 900, logical: 0 },
          content: 'my message',
          type: 'update' as const,
          entityUpdates: {
            name: { value: 'Alice', timestamp: { physical: 900, logical: 0 }, author: 'client1' }
          }
        },
        {
          id: 'sync-msg',
          from: 'client2',
          to: 'client1',
          timestamp: { physical: 950, logical: 0 },
          content: 'sync request',
          type: 'sync-request' as const
        }
      ]
    };

    render(
      <ClientNode
        client={clientWithMessages}
        onSendMessage={mockOnSendMessage}
        onReceiveMessage={mockOnReceiveMessage}
        allClients={[clientWithMessages, mockOtherClient]}
        position={mockPosition}
        onUpdateClockSkew={mockOnUpdateClockSkew}
        onToggleOnline={mockOnToggleOnline}
      />
    );

    // Simulate receiving a sync request
    const syncRequestMessage: Message = {
      id: 'sync-req-1',
      from: 'client2',
      to: 'client1',
      timestamp: { physical: 1000, logical: 0 },
      content: 'sync request',
      type: 'sync-request'
    };

    const messageEvent = new CustomEvent('message-client1', { 
      detail: { message: syncRequestMessage } 
    });
    window.dispatchEvent(messageEvent);

    await waitFor(() => {
      // Should respond with sync-response containing only update messages
      expect(mockOnSendMessage).toHaveBeenCalledWith(
        'client1',
        'client2',
        { physical: 1001, logical: 0 },
        undefined,
        'sync-response',
        expect.any(Object), // entity state
        [expect.objectContaining({ type: 'update' })] // only update messages
      );
    });
  });
});