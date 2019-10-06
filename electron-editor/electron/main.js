const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const os = require('os')
const url = require('url');
const { channels } = require('../src/shared/constants');

let mainWindow;


const extensionUrl = '/Users/jimmyhmiller/Library/Application Support/Google/Chrome/Default/Extensions';
const reactExtension = 'fmkadmapgofadopljbjfkapdkoienihi/4.2.0_0';
const reduxExtension = 'lmhkpmbekcpmknklioeibfkpmmfibljd/2.17.0_0';

function createWindow () {


  BrowserWindow.addDevToolsExtension(`${extensionUrl}/${reactExtension}`);
  BrowserWindow.addDevToolsExtension(`${extensionUrl}/${reduxExtension}`);
 

  const startUrl = process.env.ELECTRON_START_URL || url.format({
    pathname: path.join(__dirname, '../index.html'),
    protocol: 'file:',
    slashes: true,
  });


  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
    },
  });


  mainWindow.loadURL(startUrl);

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.on('ready', createWindow);

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', function () {
  if (mainWindow === null) {
    createWindow();
  }
});

ipcMain.on(channels.APP_INFO, (event) => {
  event.sender.send(channels.APP_INFO, { 
    appName: app.getName(),
    appVersion: app.getVersion(),
  });
});
