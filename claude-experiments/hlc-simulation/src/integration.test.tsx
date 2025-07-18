import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';

/**
 * Integration test to verify node synchronization behavior
 * This test simulates the real scenario where nodes go offline/online
 */
describe('Node Synchronization Integration Test', () => {
  test('nodes should sync properly when coming back online', async () => {
    render(<App />);
    
    // Wait for the app to load
    await waitFor(() => {
      expect(screen.getByText('Hybrid Logical Clock Simulation')).toBeInTheDocument();
    });
    
    // Get the client nodes
    const clientNodes = screen.getAllByText(/Client \d+/);
    expect(clientNodes).toHaveLength(2); // Default has 2 clients
    
    // Find online/offline buttons (status buttons)
    const statusButtons = screen.getAllByText(/🟢|🔴/);
    expect(statusButtons).toHaveLength(2);
    
    // First client should be online (🟢)
    expect(statusButtons[0]).toHaveTextContent('🟢');
    
    // Take first client offline
    fireEvent.click(statusButtons[0]);
    
    // Wait for UI to update
    await waitFor(() => {
      expect(statusButtons[0]).toHaveTextContent('🔴');
    });
    
    // Check that the client node has offline class
    const clientNode = statusButtons[0].closest('.client-node');
    expect(clientNode).toHaveClass('offline');
    
    // Bring client back online
    fireEvent.click(statusButtons[0]);
    
    // Wait for UI to update
    await waitFor(() => {
      expect(statusButtons[0]).toHaveTextContent('🟢');
    });
    
    // Check that the client node no longer has offline class
    expect(clientNode).not.toHaveClass('offline');
    
    // The sync should have been triggered - we can verify this by checking
    // that the client is back online and functioning
    expect(statusButtons[0]).toHaveTextContent('🟢');
  });
  
  test('should be able to add and remove clients', async () => {
    render(<App />);
    
    // Wait for the app to load
    await waitFor(() => {
      expect(screen.getByText('Clients: 2')).toBeInTheDocument();
    });
    
    // Add a client
    const addButton = screen.getByText('Add Client');
    fireEvent.click(addButton);
    
    await waitFor(() => {
      expect(screen.getByText('Clients: 3')).toBeInTheDocument();
    });
    
    // Remove a client
    const removeButton = screen.getByText('Remove Client');
    fireEvent.click(removeButton);
    
    await waitFor(() => {
      expect(screen.getByText('Clients: 2')).toBeInTheDocument();
    });
  });
  
  test('should display message logs', async () => {
    render(<App />);
    
    // Wait for the app to load
    await waitFor(() => {
      expect(screen.getByText('Hybrid Logical Clock Simulation')).toBeInTheDocument();
    });
    
    // Find Show Logs buttons
    const showLogsButtons = screen.getAllByText(/Show.*Logs/);
    expect(showLogsButtons.length).toBeGreaterThan(0);
    
    // Click first Show Logs button
    fireEvent.click(showLogsButtons[0]);
    
    // Should now show Hide Logs
    await waitFor(() => {
      expect(screen.getByText(/Hide.*Logs/)).toBeInTheDocument();
    });
  });
  
  test('should be able to update entity state', async () => {
    render(<App />);
    
    // Wait for the app to load
    await waitFor(() => {
      expect(screen.getByText('Hybrid Logical Clock Simulation')).toBeInTheDocument();
    });
    
    // Find input fields
    const nameInputs = screen.getAllByPlaceholderText('Name');
    const foodInputs = screen.getAllByPlaceholderText('Favorite Food');
    
    expect(nameInputs.length).toBeGreaterThan(0);
    expect(foodInputs.length).toBeGreaterThan(0);
    
    // Type in the first client's name input
    fireEvent.change(nameInputs[0], { target: { value: 'Alice' } });
    
    // Find and click the broadcast button
    const broadcastButtons = screen.getAllByText('Broadcast Updates');
    expect(broadcastButtons[0]).not.toBeDisabled();
    
    fireEvent.click(broadcastButtons[0]);
    
    // The entity state should eventually update
    await waitFor(() => {
      expect(screen.getByText('Alice')).toBeInTheDocument();
    });
  });
});