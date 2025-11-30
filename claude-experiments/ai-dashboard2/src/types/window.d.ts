import type { DashboardAPI, CommandAPI, ClaudeAPI, WebContentsViewAPI, ProjectAPI } from './ipc';

declare global {
  interface Window {
    dashboardAPI: DashboardAPI;
    commandAPI: CommandAPI;
    claudeAPI: ClaudeAPI;
    webContentsViewAPI: WebContentsViewAPI;
    projectAPI: ProjectAPI;
  }
}

export {};
