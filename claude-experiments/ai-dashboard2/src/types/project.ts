import type { Theme } from './theme';

export type ProjectType = 'node' | 'python' | 'rust' | 'other' | 'embedded' | 'standalone';

export interface Project {
  id: string;
  name: string;
  description?: string;
  path?: string;
  rootPath?: string;
  type: ProjectType;
  createdAt?: number | string;
  dashboards?: string[];
  icon?: string;
  theme?: Partial<Theme>;
  settings?: {
    allowFileAccess: boolean;
    cwd: string;
  };
}

export interface ProjectNode {
  name: string;
  path: string;
  isDirectory: boolean;
  children?: ProjectNode[];
}
