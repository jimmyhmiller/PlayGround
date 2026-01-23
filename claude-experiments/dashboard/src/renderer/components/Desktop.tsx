import { memo, useMemo, useCallback, useState, useRef, useEffect, ComponentType, ReactElement } from 'react';
import { WindowManagerProvider, WindowContainer, useWindowManager } from './WindowManager';
import { ThemeProvider } from '../theme/ThemeProvider';
import { SettingsProvider, useSettings } from '../settings/SettingsProvider';
import CommandPalette, { useCommandPalette, Command } from './CommandPalette';
import QuickSwitcher, { useQuickSwitcher } from './QuickSwitcher';
import PromptDialog, { ConfirmDialog } from './PromptDialog';
import {
  useProjectsState,
  useDashboardsState,
  useProjectDashboardTree,
  useThemeState,
} from '../hooks/useBackendState';
import CodeMirrorEditor from './CodeMirrorEditor';
import GitDiffViewer from './GitDiffViewer';
import EventLogPanel from './EventLogPanel';
import ThemeEditor from './ThemeEditor';
import SettingsEditor from './SettingsEditor';
import { GlobalUIRenderer } from '../globalUI';
import WidgetLayout from './WidgetLayout';
import { WidgetErrorBoundary } from './ErrorBoundary';
import { useCustomWidgets } from '../hooks/useCustomWidgets';
import { CodeExecutionHandler } from '../widgets/DynamicWidget';

interface ComponentConfig {
  component: ComponentType<unknown>;
  label: string;
  icon: string;
  defaultProps: Record<string, unknown>;
  width: number;
  height: number;
}

/**
 * Component registry - maps type strings to components
 * This is needed because we store component types as strings in backend state
 */
const COMPONENT_REGISTRY: Record<string, ComponentConfig> = {
  'codemirror': {
    component: CodeMirrorEditor as ComponentType<unknown>,
    label: 'Code Editor',
    icon: '📝',
    defaultProps: { subscribePattern: 'file.**' },
    width: 600,
    height: 400,
  },
  'git-diff': {
    component: GitDiffViewer as ComponentType<unknown>,
    label: 'Git Diff',
    icon: '📊',
    defaultProps: { subscribePattern: 'git.**' },
    width: 500,
    height: 400,
  },
  'event-log': {
    component: EventLogPanel as ComponentType<unknown>,
    label: 'Event Log',
    icon: '📋',
    defaultProps: { subscribePattern: '**' },
    width: 450,
    height: 350,
  },
  'theme-editor': {
    component: ThemeEditor as ComponentType<unknown>,
    label: 'Theme Editor',
    icon: '🎨',
    defaultProps: {},
    width: 320,
    height: 500,
  },
  'settings': {
    component: SettingsEditor as ComponentType<unknown>,
    label: 'Settings',
    icon: '⚙️',
    defaultProps: {},
    width: 300,
    height: 280,
  },
  'widget-layout': {
    component: WidgetLayout as ComponentType<unknown>,
    label: 'Dashboard',
    icon: '▦',
    defaultProps: {},
    width: 800,
    height: 500,
  },
};

/**
 * Window types for toolbar (derived from registry)
 */
const WINDOW_TYPES = Object.entries(COMPONENT_REGISTRY).map(([type, config]) => ({
  type,
  ...config,
}));

/**
 * Quick Switcher Controller - manages quick switcher state
 */
const DesktopQuickSwitcher = memo(function DesktopQuickSwitcher(): ReactElement {
  const { isOpen, close } = useQuickSwitcher();

  return <QuickSwitcher isOpen={isOpen} onClose={close} />;
});

// Dialog state types
interface PromptState {
  isOpen: boolean;
  title: string;
  placeholder: string;
  defaultValue: string;
  onSubmit: (value: string) => void;
}

interface ConfirmState {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
}

/**
 * Command Palette Controller - manages commands and palette state
 */
const DesktopCommandPalette = memo(function DesktopCommandPalette(): ReactElement {
  const { createWindow, windows, focusedId, updateWindow } = useWindowManager();
  const { settings } = useSettings();
  const shortcut = settings.commandPaletteShortcut || 'cmd+shift+p';
  const { isOpen, close } = useCommandPalette(shortcut);

  // Track mouse position for pinned window detection
  const mousePos = useRef({ x: 0, y: 0 });
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mousePos.current = { x: e.clientX, y: e.clientY };
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  // Dialog state
  const [promptState, setPromptState] = useState<PromptState>({
    isOpen: false,
    title: '',
    placeholder: '',
    defaultValue: '',
    onSubmit: () => {},
  });

  const [confirmState, setConfirmState] = useState<ConfirmState>({
    isOpen: false,
    title: '',
    message: '',
    onConfirm: () => {},
  });

  // Helper to show prompt dialog
  const showPrompt = useCallback(
    (title: string, placeholder: string, defaultValue: string = ''): Promise<string | null> => {
      return new Promise((resolve) => {
        setPromptState({
          isOpen: true,
          title,
          placeholder,
          defaultValue,
          onSubmit: (value) => {
            setPromptState((prev) => ({ ...prev, isOpen: false }));
            resolve(value);
          },
        });
      });
    },
    []
  );

  const closePrompt = useCallback(() => {
    setPromptState((prev) => ({ ...prev, isOpen: false }));
  }, []);

  // Helper to show confirm dialog
  const showConfirm = useCallback((title: string, message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmState({
        isOpen: true,
        title,
        message,
        onConfirm: () => {
          setConfirmState((prev) => ({ ...prev, isOpen: false }));
          resolve(true);
        },
      });
    });
  }, []);

  const closeConfirm = useCallback(() => {
    setConfirmState((prev) => ({ ...prev, isOpen: false }));
  }, []);

  // Project and dashboard state for commands
  const {
    projects,
    activeProjectId,
    activeProject,
    createProject,
    deleteProject,
    renameProject,
    setProjectTheme,
  } = useProjectsState();

  const {
    dashboards,
    activeDashboard,
    createDashboard,
    deleteDashboard,
    renameDashboard,
    saveLayout,
    setDashboardThemeOverride,
  } = useDashboardsState();

  const { tree } = useProjectDashboardTree();

  const { currentTheme, overrides } = useThemeState();

  // Build command list
  const commands: Command[] = useMemo(() => {
    const cmds: Command[] = [];

    // Window creation commands
    WINDOW_TYPES.forEach((config) => {
      cmds.push({
        id: `create-${config.type}`,
        label: `New ${config.label}`,
        description: `Open a new ${config.label.toLowerCase()} window`,
        icon: config.icon,
        category: 'Windows',
        keywords: ['create', 'new', 'open', 'window'],
        action: () => {
          createWindow({
            title: config.label,
            componentType: config.type,
            props: config.defaultProps,
            width: config.width,
            height: config.height,
          });
        },
      });
    });

    // Pin command - finds unpinned window under mouse cursor
    const unpinnedWindows = windows.filter(w => !w.pinned);
    if (unpinnedWindows.length > 0) {
      cmds.push({
        id: 'window-pin',
        label: 'Pin Window (under cursor)',
        description: 'Lock window in place (no chrome, not draggable, behind other windows)',
        category: 'Windows',
        keywords: ['pin', 'window', 'lock', 'fixed', 'background'],
        action: () => {
          const { x, y } = mousePos.current;
          // Find unpinned window under cursor (highest zIndex first)
          const sorted = [...unpinnedWindows].sort((a, b) => b.zIndex - a.zIndex);
          const windowUnderCursor = sorted.find(w => {
            return x >= w.x && x <= w.x + w.width &&
                   y >= w.y && y <= w.y + w.height;
          });
          if (windowUnderCursor) {
            updateWindow(windowUnderCursor.id, { pinned: true });
          }
        },
      });
    }

    // Unpin command - finds pinned window under mouse cursor
    const pinnedWindows = windows.filter(w => w.pinned);
    if (pinnedWindows.length > 0) {
      cmds.push({
        id: 'window-unpin',
        label: 'Unpin Window (under cursor)',
        description: 'Unpin the pinned window under your mouse cursor',
        category: 'Windows',
        keywords: ['unpin', 'window', 'unlock', 'movable'],
        action: () => {
          const { x, y } = mousePos.current;
          // Find pinned window under cursor
          const windowUnderCursor = pinnedWindows.find(w => {
            return x >= w.x && x <= w.x + w.width &&
                   y >= w.y && y <= w.y + w.height;
          });
          if (windowUnderCursor) {
            updateWindow(windowUnderCursor.id, { pinned: false });
          }
        },
      });
    }

    // Project commands
    cmds.push({
      id: 'project-create',
      label: 'Create New Project',
      description: 'Create a new project with a default dashboard',
      icon: '📁',
      category: 'Projects',
      keywords: ['project', 'create', 'new'],
      action: async () => {
        const name = await showPrompt('Create New Project', 'Enter project name');
        if (name) {
          await createProject(name);
        }
      },
    });

    if (activeProject) {
      cmds.push({
        id: 'project-rename',
        label: 'Rename Current Project',
        description: `Rename "${activeProject.name}"`,
        icon: '✏️',
        category: 'Projects',
        keywords: ['project', 'rename', 'edit'],
        action: async () => {
          const name = await showPrompt(
            'Rename Project',
            'Enter new project name',
            activeProject.name
          );
          if (name && name !== activeProject.name) {
            await renameProject(activeProject.id, name);
          }
        },
      });

      cmds.push({
        id: 'project-save-theme',
        label: 'Save Theme as Project Default',
        description: 'Set current theme as the default for this project',
        icon: '🎨',
        category: 'Theme',
        keywords: ['theme', 'save', 'project', 'default'],
        action: async () => {
          await setProjectTheme(activeProject.id, {
            current: currentTheme,
            overrides,
          });
        },
      });
    }

    if (projects.length > 1 && activeProject) {
      cmds.push({
        id: 'project-delete',
        label: 'Delete Current Project',
        description: `Delete "${activeProject.name}" and all its dashboards`,
        icon: '🗑️',
        category: 'Projects',
        keywords: ['project', 'delete', 'remove'],
        action: async () => {
          const confirmed = await showConfirm(
            'Delete Project',
            `Are you sure you want to delete project "${activeProject.name}"? This will delete all its dashboards.`
          );
          if (confirmed) {
            await deleteProject(activeProject.id);
          }
        },
      });
    }

    // Switch to other projects
    projects
      .filter((p) => p.id !== activeProjectId)
      .forEach((project) => {
        cmds.push({
          id: `project-switch-${project.id}`,
          label: `Switch to Project: ${project.name}`,
          description: 'Switch to this project',
          icon: '📂',
          category: 'Projects',
          keywords: ['project', 'switch', 'goto', project.name.toLowerCase()],
          action: async () => {
            await window.stateAPI.command('projects.switch', { id: project.id });
          },
        });
      });

    // Dashboard commands
    if (activeProject) {
      cmds.push({
        id: 'dashboard-create',
        label: 'Create New Dashboard',
        description: `Create a new dashboard in "${activeProject.name}"`,
        icon: '📋',
        category: 'Dashboards',
        keywords: ['dashboard', 'create', 'new'],
        action: async () => {
          const name = await showPrompt('Create New Dashboard', 'Enter dashboard name');
          if (name) {
            await createDashboard(activeProject.id, name);
          }
        },
      });
    }

    if (activeDashboard) {
      cmds.push({
        id: 'dashboard-rename',
        label: 'Rename Current Dashboard',
        description: `Rename "${activeDashboard.name}"`,
        icon: '✏️',
        category: 'Dashboards',
        keywords: ['dashboard', 'rename', 'edit'],
        action: async () => {
          const name = await showPrompt(
            'Rename Dashboard',
            'Enter new dashboard name',
            activeDashboard.name
          );
          if (name && name !== activeDashboard.name) {
            await renameDashboard(activeDashboard.id, name);
          }
        },
      });

      cmds.push({
        id: 'dashboard-save',
        label: 'Save Dashboard Layout',
        description: 'Save current window layout to this dashboard',
        icon: '💾',
        category: 'Dashboards',
        keywords: ['dashboard', 'save', 'layout'],
        action: async () => {
          await saveLayout();
        },
      });

      cmds.push({
        id: 'dashboard-save-theme',
        label: 'Save Theme to Dashboard',
        description: 'Save current theme as this dashboard\'s theme override',
        icon: '🎨',
        category: 'Theme',
        keywords: ['theme', 'save', 'dashboard', 'override'],
        action: async () => {
          await setDashboardThemeOverride(activeDashboard.id, {
            current: currentTheme,
            overrides,
          });
        },
      });

      // Only show reset if dashboard has a theme override
      if (activeDashboard.themeOverride) {
        cmds.push({
          id: 'dashboard-reset-theme',
          label: 'Reset Dashboard Theme',
          description: 'Use project default theme instead of dashboard override',
          icon: '↩️',
          category: 'Theme',
          keywords: ['theme', 'reset', 'dashboard', 'default'],
          action: async () => {
            await setDashboardThemeOverride(activeDashboard.id, undefined);
          },
        });
      }

      // Only show delete if there's more than one dashboard
      const projectDashboards = dashboards.filter(
        (d) => d.projectId === activeProject?.id
      );
      if (projectDashboards.length > 1) {
        cmds.push({
          id: 'dashboard-delete',
          label: 'Delete Current Dashboard',
          description: `Delete "${activeDashboard.name}"`,
          icon: '🗑️',
          category: 'Dashboards',
          keywords: ['dashboard', 'delete', 'remove'],
          action: async () => {
            const confirmed = await showConfirm(
              'Delete Dashboard',
              `Are you sure you want to delete dashboard "${activeDashboard.name}"?`
            );
            if (confirmed) {
              await deleteDashboard(activeDashboard.id);
            }
          },
        });
      }
    }

    // Switch to other dashboards (grouped by project)
    tree.forEach(({ project, dashboards: projectDashboards }) => {
      projectDashboards
        .filter((d) => d.id !== activeDashboard?.id)
        .forEach((dashboard) => {
          cmds.push({
            id: `dashboard-switch-${dashboard.id}`,
            label: `Switch to: ${project.name} / ${dashboard.name}`,
            description: 'Switch to this dashboard',
            icon: '📄',
            category: 'Dashboards',
            keywords: [
              'dashboard',
              'switch',
              'goto',
              project.name.toLowerCase(),
              dashboard.name.toLowerCase(),
            ],
            action: async () => {
              await window.stateAPI.command('dashboards.switch', { id: dashboard.id });
            },
          });
        });
    });

    // Git commands
    cmds.push({
      id: 'git-refresh',
      label: 'Git Refresh',
      description: 'Refresh git status',
      icon: '🔄',
      category: 'Git',
      keywords: ['git', 'refresh', 'status', 'update'],
      action: async () => {
        await window.gitAPI.refresh();
      },
    });

    cmds.push({
      id: 'git-poll',
      label: 'Start Git Polling',
      description: 'Poll git status every 2 seconds',
      icon: '📡',
      category: 'Git',
      keywords: ['git', 'poll', 'watch', 'auto'],
      action: async () => {
        await window.gitAPI.startPolling(2000);
      },
    });

    // Dashboard loading commands
    cmds.push({
      id: 'load-dashboard-json',
      label: 'Load Dashboard from JSON',
      description: 'Load a dashboard layout from a JSON file',
      icon: '▦',
      category: 'Dashboards',
      keywords: ['load', 'dashboard', 'json', 'file', 'open', 'layout'],
      action: async () => {
        const filePath = await showPrompt(
          'Load Dashboard',
          'Enter path to dashboard JSON file',
          ''
        );
        if (filePath) {
          try {
            const result = await window.fileAPI.load(filePath);
            const config = JSON.parse(result.content);
            const layout = config.layout || config;

            // Handle window-group specially - spawn windows directly
            if (layout.type === 'window-group') {
              const scope = layout.scope || `wg-${Date.now()}`;
              for (const win of layout.windows) {
                createWindow({
                  title: win.title,
                  componentType: 'widget-layout',
                  props: {
                    scope,
                    padding: 0,
                    background: 'transparent',
                    config: win.widget,
                  },
                  x: win.x,
                  y: win.y,
                  width: win.width ?? 400,
                  height: win.height ?? 300,
                });
              }
            } else {
              // Regular dashboard - create single window
              const title = config.name || 'Dashboard';
              createWindow({
                title,
                componentType: 'widget-layout',
                props: { configPath: filePath },
                width: 800,
                height: 600,
              });
            }
          } catch (err) {
            console.error('Failed to load dashboard:', err);
          }
        }
      },
    });

    cmds.push({
      id: 'load-dashboard-inline',
      label: 'Load Dashboard from Clipboard',
      description: 'Create a dashboard from JSON in clipboard',
      icon: '📋',
      category: 'Dashboards',
      keywords: ['paste', 'dashboard', 'json', 'clipboard', 'inline'],
      action: async () => {
        try {
          const text = await navigator.clipboard.readText();
          const config = JSON.parse(text);
          const title = config.name || 'Dashboard';
          createWindow({
            title,
            componentType: 'widget-layout',
            props: { config: config.layout || config },
            width: 800,
            height: 600,
          });
        } catch (err) {
          console.error('Failed to parse clipboard JSON:', err);
        }
      },
    });

    // Flask Demo - Spawn as independent floating panes
    cmds.push({
      id: 'flask-demo-panes',
      label: 'Flask Demo (Floating Panes)',
      description: 'Open Flask demo as independent draggable panes',
      icon: '🐍',
      category: 'Demos',
      keywords: ['flask', 'demo', 'floating', 'panes', 'light table'],
      action: async () => {
        const sharedScope = 'flask-demo';

        // File loader (hidden, just emits events)
        createWindow({
          title: 'Flask Loader',
          componentType: 'widget-layout',
          props: {
            scope: sharedScope,
            padding: 4,
            config: {
              type: 'layout',
              direction: 'vertical',
              gap: 4,
              children: [
                {
                  type: 'process-runner',
                  props: {
                    id: 'flask-server',
                    command: './venv/bin/python',
                    args: ['app.py'],
                    cwd: 'examples/flask-app',
                    title: 'Server',
                    startLabel: 'Start',
                    stopLabel: 'Stop',
                    showOutput: false,
                  },
                },
                {
                  type: 'file-loader',
                  props: {
                    files: ['examples/flask-app/app.py', 'examples/flask-app/models.py'],
                    channel: '$scope.routes',
                    transform: "(content, filePath, allFiles) => { if (!filePath.endsWith('app.py')) return null; const modelsContent = allFiles && allFiles[1] ? allFiles[1].content : ''; const routes = []; const lines = content.split('\\n'); let i = 0; while (i < lines.length) { const line = lines[i]; const match = line.match(/@app\\.route\\(['\"]([^'\"]+)['\"](?:,\\s*methods=\\[([^\\]]+)\\])?\\)/); if (match) { const path = match[1]; const methods = match[2] ? match[2].replace(/['\"/]/g, '').split(',').map(m => m.trim()) : ['GET']; let codeEnd = i; let template = null; let usesTask = false; let usesNote = false; for (let j = i + 1; j < lines.length; j++) { if (lines[j].match(/^@app\\.route|^if __name__/)) { codeEnd = j - 1; break; } if (j === lines.length - 1) codeEnd = j; const tmplMatch = lines[j].match(/render_template\\(['\"]([^'\"]+)['\"]/); if (tmplMatch) template = 'examples/flask-app/templates/' + tmplMatch[1]; if (lines[j].includes('Task')) usesTask = true; if (lines[j].includes('Note')) usesNote = true; } const code = lines.slice(i, codeEnd + 1).join('\\n'); let modelCode = ''; if (modelsContent && (usesTask || usesNote)) { const modelLines = modelsContent.split('\\n'); let inClass = false; let className = ''; let classStart = -1; for (let k = 0; k < modelLines.length; k++) { const classMatch = modelLines[k].match(/^class (Task|Note)/); if (classMatch) { if (inClass && ((className === 'Task' && usesTask) || (className === 'Note' && usesNote))) { modelCode += modelLines.slice(classStart, k).join('\\n') + '\\n\\n'; } inClass = true; className = classMatch[1]; classStart = k; } } if (inClass && ((className === 'Task' && usesTask) || (className === 'Note' && usesNote))) { let endIdx = modelLines.length; for (let k = classStart + 1; k < modelLines.length; k++) { if (modelLines[k].match(/^class |^# Seed|^Task\\(|^Note\\(/)) { endIdx = k; break; } } modelCode += modelLines.slice(classStart, endIdx).join('\\n'); } } routes.push({ id: path + methods[0], path, methods, code, template, modelCode: modelCode.trim(), file: filePath }); } i++; } return routes; }",
                  },
                },
              ],
            },
          },
          width: 200,
          height: 100,
          x: 20,
          y: 20,
        });

        // Routes selector
        createWindow({
          title: 'Routes',
          componentType: 'widget-layout',
          props: {
            scope: sharedScope,
            padding: 0,
            background: 'transparent',
            config: {
              type: 'selector',
              props: {
                subscribePattern: 'loaded.$scope.routes',
                dataPath: '0.data',
                labelTemplate: '${methods} ${path}',
                idKey: 'id',
                channel: '$scope.route',
                direction: 'vertical',
              },
            },
          },
          width: 200,
          height: 400,
          x: 20,
          y: 130,
        });

        // Controller code
        createWindow({
          title: 'Controller',
          componentType: 'widget-layout',
          props: {
            scope: sharedScope,
            padding: 0,
            background: 'transparent',
            config: {
              type: 'code-block',
              props: {
                subscribePattern: 'selection.$scope.route',
                codeKey: 'code',
                lineNumbers: true,
                language: 'Python',
              },
            },
          },
          width: 450,
          height: 300,
          x: 240,
          y: 20,
        });

        // Model code
        createWindow({
          title: 'Model',
          componentType: 'widget-layout',
          props: {
            scope: sharedScope,
            padding: 0,
            background: 'transparent',
            config: {
              type: 'code-block',
              props: {
                subscribePattern: 'selection.$scope.route',
                codeKey: 'modelCode',
                lineNumbers: true,
                language: 'Python',
              },
            },
          },
          width: 450,
          height: 300,
          x: 240,
          y: 340,
        });

        // Template code
        createWindow({
          title: 'Template',
          componentType: 'widget-layout',
          props: {
            scope: sharedScope,
            padding: 0,
            background: 'transparent',
            config: {
              type: 'code-block',
              props: {
                filePattern: 'selection.$scope.route',
                fileKey: 'template',
                lineNumbers: true,
                language: 'HTML',
              },
            },
          },
          width: 450,
          height: 300,
          x: 710,
          y: 20,
        });

        // Live preview
        createWindow({
          title: 'Live Preview',
          componentType: 'widget-layout',
          props: {
            scope: sharedScope,
            padding: 0,
            background: 'transparent',
            config: {
              type: 'webview',
              props: {
                url: 'http://localhost:5001',
                subscribePattern: 'selection.$scope.route',
                pathKey: 'path',
              },
            },
          },
          width: 450,
          height: 400,
          x: 710,
          y: 340,
        });
      },
    });

    return cmds;
  }, [
    createWindow,
    windows,
    focusedId,
    updateWindow,
    projects,
    activeProjectId,
    activeProject,
    createProject,
    deleteProject,
    renameProject,
    setProjectTheme,
    dashboards,
    activeDashboard,
    createDashboard,
    deleteDashboard,
    renameDashboard,
    saveLayout,
    setDashboardThemeOverride,
    tree,
    currentTheme,
    overrides,
    showPrompt,
    showConfirm,
  ]);

  const handleExecute = useCallback((command: Command) => {
    command.action();
  }, []);

  // Claude status state
  const [claudeStatus, setClaudeStatus] = useState<{
    active: boolean;
    query: string;
    status: 'connecting' | 'working' | 'done' | 'error';
    message?: string;
  }>({ active: false, query: '', status: 'connecting' });

  // Handle freeform queries to Claude - runs in background with status indicator
  const handleAskClaude = useCallback(async (query: string) => {
    setClaudeStatus({ active: true, query, status: 'connecting' });

    try {
      // Check if ACP is connected, if not spawn and initialize
      const isConnected = await window.acpAPI.isConnected();
      if (!isConnected) {
        await window.acpAPI.spawn();
        await window.acpAPI.initialize();
      }

      setClaudeStatus({ active: true, query, status: 'working' });

      // Create a new session
      const { sessionId } = await window.acpAPI.newSession('/');

      // Build a prompt that gives Claude context about the dashboard
      const prompt = `You are helping modify a React/Electron dashboard application. The user wants to make changes to their dashboard.

The dashboard has these key features:
- Window management system (create/move/resize floating windows)
- Widget-based layouts (widgets can be arranged in grids/layouts)
- Custom widget support (widgets are React components)
- Theme system with CSS variables
- Command palette for actions
- Projects and dashboards for organization

The user's request: "${query}"

Please implement this request by:
1. Creating or modifying the necessary widget components
2. Using the dashboard's MCP tools to register widgets or update the UI
3. If this is a new widget, register it using the dashboard_register_widget tool
4. Keep the code simple and focused on the request

Start by understanding what files need to be created or modified, then implement the changes.`;

      // Send the prompt to Claude
      await window.acpAPI.prompt(sessionId, prompt);

      setClaudeStatus({ active: true, query, status: 'done', message: 'Complete!' });

      // Auto-hide after 3 seconds
      setTimeout(() => {
        setClaudeStatus(prev => prev.status === 'done' ? { ...prev, active: false } : prev);
      }, 3000);

    } catch (error) {
      console.error('[Desktop] Failed to send query to Claude:', error);
      setClaudeStatus({
        active: true,
        query,
        status: 'error',
        message: (error as Error).message || 'Unknown error',
      });
    }
  }, []);

  return (
    <>
      <CommandPalette
        isOpen={isOpen}
        onClose={close}
        commands={commands}
        onExecute={handleExecute}
        onAskClaude={handleAskClaude}
      />

      {/* Claude Status Indicator */}
      {claudeStatus.active && (
        <div
          style={{
            position: 'fixed',
            bottom: 16,
            right: 16,
            background: 'var(--theme-bg-elevated, #252540)',
            border: '1px solid var(--theme-border-primary, #333)',
            borderRadius: 8,
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            zIndex: 9999,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            maxWidth: 300,
          }}
        >
          {claudeStatus.status === 'connecting' && (
            <div style={{ width: 16, height: 16, borderRadius: '50%', background: '#f59e0b', animation: 'pulse 1s infinite' }} />
          )}
          {claudeStatus.status === 'working' && (
            <div style={{ width: 16, height: 16, borderRadius: '50%', border: '2px solid #3b82f6', borderTopColor: 'transparent', animation: 'spin 1s linear infinite' }} />
          )}
          {claudeStatus.status === 'done' && (
            <div style={{ width: 16, height: 16, borderRadius: '50%', background: '#22c55e' }} />
          )}
          {claudeStatus.status === 'error' && (
            <div style={{ width: 16, height: 16, borderRadius: '50%', background: '#ef4444' }} />
          )}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{
              fontSize: 12,
              color: 'var(--theme-text-muted, #888)',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}>
              {claudeStatus.query}
            </div>
            <div style={{
              fontSize: 11,
              color: claudeStatus.status === 'error' ? '#ef4444' : 'var(--theme-text-secondary, #aaa)',
            }}>
              {claudeStatus.status === 'connecting' && 'Connecting to Claude...'}
              {claudeStatus.status === 'working' && 'Working...'}
              {claudeStatus.status === 'done' && (claudeStatus.message || 'Done!')}
              {claudeStatus.status === 'error' && (claudeStatus.message || 'Error')}
            </div>
          </div>
          <button
            onClick={() => setClaudeStatus(prev => ({ ...prev, active: false }))}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--theme-text-muted, #888)',
              cursor: 'pointer',
              padding: 4,
              fontSize: 14,
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* CSS for animations */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>

      <PromptDialog
        isOpen={promptState.isOpen}
        title={promptState.title}
        placeholder={promptState.placeholder}
        defaultValue={promptState.defaultValue}
        onSubmit={promptState.onSubmit}
        onCancel={closePrompt}
      />
      <ConfirmDialog
        isOpen={confirmState.isOpen}
        title={confirmState.title}
        message={confirmState.message}
        onConfirm={confirmState.onConfirm}
        onCancel={closeConfirm}
      />
    </>
  );
});

/**
 * CustomWidgetsLoader - Loads custom widgets from state on startup
 * This component renders nothing but initializes the custom widget system.
 */
const CustomWidgetsLoader = memo(function CustomWidgetsLoader(): null {
  // This hook loads custom widgets from backend state and
  // subscribes to registration/update/unregistration events
  useCustomWidgets();
  return null;
});

/**
 * Desktop environment with windowing and theming
 */
function Desktop(): ReactElement {
  const [leftPanelVisible, setLeftPanelVisible] = useState(false);
  const [rightPanelVisible, setRightPanelVisible] = useState(false);
  const [sharedPanelWidth, setSharedPanelWidth] = useState(200);

  // Listen for panel visibility changes
  useEffect(() => {
    const unsubLeft = window.eventAPI.subscribe('globalUI.leftPanel.toggle', (event) => {
      setLeftPanelVisible(event.payload as boolean);
    });
    const unsubRight = window.eventAPI.subscribe('globalUI.rightPanel.toggle', (event) => {
      setRightPanelVisible(event.payload as boolean);
    });
    
    // Listen for width changes from EITHER side - they both set the same shared width
    const unsubLeftWidth = window.eventAPI.subscribe('globalUI.leftPanel.width', (event) => {
      const width = event.payload as number;
      if (width > 0) setSharedPanelWidth(width);
    });
    const unsubRightWidth = window.eventAPI.subscribe('globalUI.rightPanel.width', (event) => {
      const width = event.payload as number;
      if (width > 0) setSharedPanelWidth(width);
    });
    
    // Listen for panel moves - when moving, update visibility for both sides
    const unsubMove = window.eventAPI.subscribe('globalUI.panel.moveTo', (event) => {
      const newSide = event.payload as string;
      if (newSide === 'left') {
        setLeftPanelVisible(true);
        setRightPanelVisible(false);
      } else {
        setLeftPanelVisible(false);
        setRightPanelVisible(true);
      }
    });
    
    return () => {
      unsubLeft();
      unsubRight();
      unsubLeftWidth();
      unsubRightWidth();
      unsubMove();
    };
  }, []);

  return (
    <SettingsProvider>
      <ThemeProvider>
        <WindowManagerProvider componentRegistry={COMPONENT_REGISTRY}>
          {/* Initialize custom widgets system */}
          <CustomWidgetsLoader />
          {/* Handle code execution requests from MCP server */}
          <CodeExecutionHandler />
          <div
            className="desktop"
            style={{
              display: 'flex',
              flexDirection: 'column',
              height: '100vh',
              background: 'var(--theme-bg-primary)',
              fontFamily: 'var(--theme-font-family)',
              color: 'var(--theme-text-primary)',
              position: 'relative',
            }}
          >
            {/* Drag region for frameless window */}
            <div
              className="window-drag-region"
              style={{
                position: 'absolute',
                top: 0,
                left: 80, // Leave space for traffic lights
                right: 0,
                height: '38px',
                // @ts-expect-error - webkit property for electron
                WebkitAppRegion: 'drag',
                zIndex: 1000,
              }}
            />
            {/* Background layer for pattern overlays (like ai-dashboard2) */}
            <div
              className="bg-layer"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 0,
                opacity: 'var(--theme-bg-layer-opacity)',
                background: 'var(--theme-bg-layer-background)',
                backgroundSize: 'var(--theme-bg-layer-background-size)',
                filter: 'var(--theme-bg-layer-filter)',
              }}
            />
            <div
              className="desktop-workspace"
              style={{
                flex: 1,
                position: 'relative',
                zIndex: 1,
                marginLeft: leftPanelVisible ? sharedPanelWidth : 0,
                marginRight: rightPanelVisible ? sharedPanelWidth : 0,
              }}
            >
              <WindowContainer />
            </div>
            {/* Global UI layer - widgets in fixed positions */}
            <WidgetErrorBoundary>
              <GlobalUIRenderer />
            </WidgetErrorBoundary>
            <DesktopCommandPalette />
            <DesktopQuickSwitcher />
          </div>
        </WindowManagerProvider>
      </ThemeProvider>
    </SettingsProvider>
  );
}

export default Desktop;
