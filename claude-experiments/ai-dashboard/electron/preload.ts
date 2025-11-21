import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Style Agent (streaming)
  generateStyle: (request: any) => {
    ipcRenderer.send('style:generate', request);
  },
  onStyleChunk: (callback: (chunk: string) => void) => {
    ipcRenderer.on('style:chunk', (_event, chunk) => callback(chunk));
  },
  onStyleComplete: (callback: (style: any) => void) => {
    ipcRenderer.on('style:complete', (_event, style) => callback(style));
  },
  onStyleError: (callback: (error: string) => void) => {
    ipcRenderer.on('style:error', (_event, error) => callback(error));
  },
  getCurrentStyle: () => ipcRenderer.invoke('style:getCurrent'),

  // Project style storage
  setProjectStyle: (projectId: string, style: any) =>
    ipcRenderer.invoke('style:setProjectStyle', projectId, style),
  getProjectStyle: (projectId: string) =>
    ipcRenderer.invoke('style:getProjectStyle', projectId),
  getAllStyles: () => ipcRenderer.invoke('style:getAllStyles'),

  // External Agents
  sendToAgent: (agentId: string, message: any) =>
    ipcRenderer.invoke('agent:send', agentId, message),
  getComponents: () => ipcRenderer.invoke('agent:getComponents'),
  getData: (dataSourceId: string) => ipcRenderer.invoke('agent:getData', dataSourceId),

  // Listen for agent messages
  onAgentRegistration: (callback: (message: any) => void) => {
    ipcRenderer.on('agent:registration', (_event, message) => callback(message));
  },
  onDataUpdate: (callback: (message: any) => void) => {
    ipcRenderer.on('agent:data-update', (_event, message) => callback(message));
  },
});

// Type definitions for TypeScript
export interface ElectronAPI {
  generateStyle: (request: any) => void;
  onStyleChunk: (callback: (chunk: string) => void) => void;
  onStyleComplete: (callback: (style: any) => void) => void;
  onStyleError: (callback: (error: string) => void) => void;
  getCurrentStyle: () => Promise<any>;
  setProjectStyle: (projectId: string, style: any) => Promise<any>;
  getProjectStyle: (projectId: string) => Promise<any>;
  getAllStyles: () => Promise<any>;
  sendToAgent: (agentId: string, message: any) => Promise<any>;
  getComponents: () => Promise<any[]>;
  getData: (dataSourceId: string) => Promise<any>;
  onAgentRegistration: (callback: (message: any) => void) => void;
  onDataUpdate: (callback: (message: any) => void) => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
