import { memo, useMemo, useCallback, useState, ComponentType, ReactElement } from 'react';
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
} from '../hooks/useBackendState';
import CodeMirrorEditor from './CodeMirrorEditor';
import GitDiffViewer from './GitDiffViewer';
import EventLogPanel from './EventLogPanel';
import ThemeEditor from './ThemeEditor';
import SettingsEditor from './SettingsEditor';

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
  const { createWindow } = useWindowManager();
  const { settings } = useSettings();
  const shortcut = settings.commandPaletteShortcut || 'cmd+shift+p';
  const { isOpen, close } = useCommandPalette(shortcut);

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
  } = useProjectsState();

  const {
    dashboards,
    activeDashboard,
    createDashboard,
    deleteDashboard,
    renameDashboard,
    saveLayout,
  } = useDashboardsState();

  const { tree } = useProjectDashboardTree();

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

    return cmds;
  }, [
    createWindow,
    projects,
    activeProjectId,
    activeProject,
    createProject,
    deleteProject,
    renameProject,
    dashboards,
    activeDashboard,
    createDashboard,
    deleteDashboard,
    renameDashboard,
    saveLayout,
    tree,
    showPrompt,
    showConfirm,
  ]);

  const handleExecute = useCallback((command: Command) => {
    command.action();
  }, []);

  return (
    <>
      <CommandPalette
        isOpen={isOpen}
        onClose={close}
        commands={commands}
        onExecute={handleExecute}
      />
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
 * Desktop environment with windowing and theming
 */
function Desktop(): ReactElement {
  return (
    <SettingsProvider>
      <ThemeProvider>
        <WindowManagerProvider componentRegistry={COMPONENT_REGISTRY}>
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
              style={{ flex: 1, position: 'relative', zIndex: 1 }}
            >
              <WindowContainer />
            </div>
            <DesktopCommandPalette />
            <DesktopQuickSwitcher />
          </div>
        </WindowManagerProvider>
      </ThemeProvider>
    </SettingsProvider>
  );
}

export default Desktop;
