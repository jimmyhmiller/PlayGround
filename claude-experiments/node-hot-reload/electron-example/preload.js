const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  getGreeting: () => ipcRenderer.invoke('get-greeting'),
  handleClick: () => ipcRenderer.invoke('handle-click'),
  getStatus: () => ipcRenderer.invoke('get-status'),
});
