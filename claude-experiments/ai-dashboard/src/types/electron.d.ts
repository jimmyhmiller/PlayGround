// Electron API type definitions for renderer process

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
