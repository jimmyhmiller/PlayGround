const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getMessage: () => ipcRenderer.invoke('get-message'),
  increment: () => ipcRenderer.invoke('increment'),
  getCounter: () => ipcRenderer.invoke('get-counter'),
});
