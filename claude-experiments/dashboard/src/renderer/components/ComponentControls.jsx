import { useState } from 'react';
import { useComponentInstances } from './ComponentRegistry';

/**
 * Controls for adding dynamic components and triggering service operations
 */
export function ComponentControls() {
  const { addInstance, getRegisteredTypes } = useComponentInstances();
  const [filePath, setFilePath] = useState('');

  const handleAddEditor = () => {
    addInstance('codemirror', {
      subscribePattern: 'file.**',
      filePath: filePath || null,
    });
  };

  const handleAddGitDiff = () => {
    addInstance('git-diff', {
      subscribePattern: 'git.**',
    });
  };

  const handleLoadFile = async () => {
    if (!filePath) return;
    await window.fileAPI.load(filePath);
  };

  const handleWatchFile = async () => {
    if (!filePath) return;
    await window.fileAPI.watch(filePath);
  };

  const handleRefreshGit = async () => {
    await window.gitAPI.refresh();
  };

  const handleStartGitPolling = async () => {
    await window.gitAPI.startPolling(2000);
  };

  const handleStopGitPolling = async () => {
    await window.gitAPI.stopPolling();
  };

  const buttonStyle = {
    padding: '6px 12px',
    fontSize: '12px',
    cursor: 'pointer',
    border: 'none',
    borderRadius: '4px',
    marginRight: '8px',
    marginBottom: '8px',
  };

  return (
    <div style={{
      background: '#f5f5f5',
      padding: '15px',
      borderRadius: '8px',
      marginBottom: '20px',
    }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Component Controls</h3>

      {/* File path input */}
      <div style={{ marginBottom: '15px' }}>
        <input
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          placeholder="File path (e.g., ./src/main/index.js)"
          style={{
            padding: '8px 12px',
            width: '300px',
            borderRadius: '4px',
            border: '1px solid #ddd',
            marginRight: '8px',
          }}
        />
        <button
          onClick={handleLoadFile}
          style={{ ...buttonStyle, background: '#2196F3', color: 'white' }}
        >
          Load File
        </button>
        <button
          onClick={handleWatchFile}
          style={{ ...buttonStyle, background: '#ff9800', color: 'white' }}
        >
          Watch File
        </button>
      </div>

      {/* Add components */}
      <div style={{ marginBottom: '15px' }}>
        <span style={{ fontSize: '12px', color: '#666', marginRight: '10px' }}>
          Add Component:
        </span>
        <button
          onClick={handleAddEditor}
          style={{ ...buttonStyle, background: '#4CAF50', color: 'white' }}
        >
          + CodeMirror Editor
        </button>
        <button
          onClick={handleAddGitDiff}
          style={{ ...buttonStyle, background: '#9c27b0', color: 'white' }}
        >
          + Git Diff Viewer
        </button>
      </div>

      {/* Git controls */}
      <div>
        <span style={{ fontSize: '12px', color: '#666', marginRight: '10px' }}>
          Git:
        </span>
        <button
          onClick={handleRefreshGit}
          style={{ ...buttonStyle, background: '#607d8b', color: 'white' }}
        >
          Refresh Status
        </button>
        <button
          onClick={handleStartGitPolling}
          style={{ ...buttonStyle, background: '#00bcd4', color: 'white' }}
        >
          Start Polling
        </button>
        <button
          onClick={handleStopGitPolling}
          style={{ ...buttonStyle, background: '#f44336', color: 'white' }}
        >
          Stop Polling
        </button>
      </div>
    </div>
  );
}
