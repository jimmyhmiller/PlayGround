import * as fs from 'fs';
import * as path from 'path';
import { app } from 'electron';

interface GeneratedStyle {
  id: string;
  prompt: string;
  timestamp: number;
  css: string;
  svgDefs?: string;
  metadata?: any;
}

type ProjectStyles = Record<string, GeneratedStyle>;

export class ProjectStyleStorage {
  private storageDir: string;
  private storageFile: string;
  private styles: ProjectStyles = {};

  constructor() {
    // Use Electron's userData directory for persistent storage
    this.storageDir = path.join(app.getPath('userData'), 'ai-dashboard');
    this.storageFile = path.join(this.storageDir, 'project-styles.json');

    // Ensure storage directory exists
    if (!fs.existsSync(this.storageDir)) {
      fs.mkdirSync(this.storageDir, { recursive: true });
    }

    // Load existing styles
    this.load();
  }

  private load(): void {
    try {
      if (fs.existsSync(this.storageFile)) {
        const data = fs.readFileSync(this.storageFile, 'utf-8');
        this.styles = JSON.parse(data);
        console.log('[Storage] Loaded project styles:', Object.keys(this.styles));
      } else {
        console.log('[Storage] No existing styles file, starting fresh');
      }
    } catch (error) {
      console.error('[Storage] Failed to load project styles:', error);
      this.styles = {};
    }
  }

  private save(): void {
    try {
      fs.writeFileSync(this.storageFile, JSON.stringify(this.styles, null, 2), 'utf-8');
      console.log('[Storage] Saved project styles to disk');
    } catch (error) {
      console.error('[Storage] Failed to save project styles:', error);
    }
  }

  setProjectStyle(projectId: string, style: GeneratedStyle): void {
    this.styles[projectId] = style;
    this.save();
    console.log(`[Storage] Set style for project ${projectId}: ${style.prompt}`);
  }

  getProjectStyle(projectId: string): GeneratedStyle | null {
    return this.styles[projectId] || null;
  }

  getAllStyles(): ProjectStyles {
    return { ...this.styles };
  }

  deleteProjectStyle(projectId: string): void {
    delete this.styles[projectId];
    this.save();
    console.log(`[Storage] Deleted style for project ${projectId}`);
  }

  clear(): void {
    this.styles = {};
    this.save();
    console.log('[Storage] Cleared all project styles');
  }
}
