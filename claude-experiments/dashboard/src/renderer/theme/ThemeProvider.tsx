import { createContext, useContext, useCallback, useEffect, useRef, ReactNode, ReactElement } from 'react';
import { useThemeState } from '../hooks/useBackendState';
import './theme.css';

/**
 * Theme Context
 *
 * Provides theme management with backend-driven state:
 * - Switch between preset themes
 * - Override individual CSS variables
 * - Get current theme values
 */

interface ThemeVariable {
  var: string;
  label: string;
  type: string;
}

interface ThemeExport {
  baseTheme: string;
  variables: Record<string, string>;
  overrides: Record<string, string>;
}

interface ThemeContextValue {
  currentTheme: string;
  presetThemes: string[];
  setTheme: (theme: string) => void;
  setVariable: (varName: string, value: string) => void;
  getVariable: (varName: string) => string;
  resetVariable: (varName: string) => void;
  resetAllOverrides: () => void;
  overrides: Record<string, string>;
  exportTheme: () => ThemeExport;
  loading: boolean;
}

interface ThemeProviderProps {
  children: ReactNode;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

const PRESET_THEMES = [
  'dark',
  'light',
  'flat',
  'tui',
  'tui-amber',
  'monokai',
  'nord',
  'solarized',
  'glass',
  'high-contrast',
  'architect',
  'dream',
  'cyber',
  'github',
];

// CSS variable categories for the theme editor
export const THEME_CATEGORIES: Record<string, ThemeVariable[]> = {
  'Structure': [
    { var: '--theme-window-radius', label: 'Window Radius', type: 'text' },
    { var: '--theme-window-border-width', label: 'Border Width', type: 'text' },
    { var: '--theme-window-border-style', label: 'Border Style', type: 'text' },
    { var: '--theme-window-clip-path', label: 'Clip Path', type: 'text' },
    { var: '--theme-window-shadow', label: 'Shadow', type: 'text' },
    { var: '--theme-radius-sm', label: 'Radius Small', type: 'text' },
    { var: '--theme-radius-lg', label: 'Radius Large', type: 'text' },
    { var: '--theme-toolbar-button-radius', label: 'Button Radius', type: 'text' },
  ],
  'Background': [
    { var: '--theme-bg-primary', label: 'Primary', type: 'color' },
    { var: '--theme-bg-secondary', label: 'Secondary', type: 'color' },
    { var: '--theme-bg-tertiary', label: 'Tertiary', type: 'color' },
    { var: '--theme-bg-elevated', label: 'Elevated', type: 'color' },
    { var: '--theme-bg-input', label: 'Input', type: 'color' },
  ],
  'Background Layer': [
    { var: '--theme-bg-layer-opacity', label: 'Opacity', type: 'text' },
    { var: '--theme-bg-layer-background', label: 'Pattern', type: 'text' },
    { var: '--theme-bg-layer-background-size', label: 'Pattern Size', type: 'text' },
    { var: '--theme-bg-layer-filter', label: 'Filter', type: 'text' },
  ],
  'Text': [
    { var: '--theme-text-primary', label: 'Primary', type: 'color' },
    { var: '--theme-text-secondary', label: 'Secondary', type: 'color' },
    { var: '--theme-text-muted', label: 'Muted', type: 'color' },
    { var: '--theme-text-disabled', label: 'Disabled', type: 'color' },
  ],
  'Borders': [
    { var: '--theme-border-primary', label: 'Primary', type: 'color' },
    { var: '--theme-border-secondary', label: 'Secondary', type: 'color' },
    { var: '--theme-window-border-color', label: 'Window', type: 'color' },
    { var: '--theme-toolbar-border-color', label: 'Toolbar', type: 'color' },
  ],
  'Accents': [
    { var: '--theme-accent-primary', label: 'Primary', type: 'color' },
    { var: '--theme-accent-secondary', label: 'Secondary', type: 'color' },
    { var: '--theme-accent-success', label: 'Success', type: 'color' },
    { var: '--theme-accent-warning', label: 'Warning', type: 'color' },
    { var: '--theme-accent-error', label: 'Error', type: 'color' },
    { var: '--theme-accent-info', label: 'Info', type: 'color' },
  ],
  'Window': [
    { var: '--theme-window-bg', label: 'Background', type: 'text' },
    { var: '--theme-window-header-bg', label: 'Header', type: 'text' },
    { var: '--theme-window-header-inactive-bg', label: 'Header Inactive', type: 'text' },
    { var: '--theme-window-title-transform', label: 'Title Transform', type: 'text' },
    { var: '--theme-window-title-weight', label: 'Title Weight', type: 'text' },
  ],
  'Toolbar': [
    { var: '--theme-toolbar-bg', label: 'Background', type: 'color' },
    { var: '--theme-toolbar-button-bg', label: 'Button', type: 'color' },
    { var: '--theme-toolbar-button-hover-bg', label: 'Button Hover', type: 'color' },
  ],
  'Command Palette': [
    { var: '--theme-palette-bg', label: 'Background', type: 'text' },
    { var: '--theme-palette-border', label: 'Border', type: 'color' },
    { var: '--theme-palette-shadow', label: 'Shadow', type: 'text' },
    { var: '--theme-palette-input-bg', label: 'Input BG', type: 'color' },
    { var: '--theme-palette-input-border', label: 'Input Border', type: 'color' },
    { var: '--theme-palette-item-hover', label: 'Item Hover', type: 'color' },
    { var: '--theme-palette-item-selected', label: 'Selected', type: 'color' },
    { var: '--theme-palette-overlay', label: 'Overlay', type: 'text' },
    { var: '--theme-palette-radius', label: 'Radius', type: 'text' },
  ],
  'Code': [
    { var: '--theme-code-bg', label: 'Background', type: 'color' },
    { var: '--theme-code-text', label: 'Text', type: 'color' },
    { var: '--theme-code-keyword', label: 'Keyword', type: 'color' },
    { var: '--theme-code-string', label: 'String', type: 'color' },
    { var: '--theme-code-comment', label: 'Comment', type: 'color' },
  ],
  'Typography': [
    { var: '--theme-font-family', label: 'Font Family', type: 'text' },
    { var: '--theme-font-mono', label: 'Mono Font', type: 'text' },
  ],
  'Events': [
    { var: '--theme-event-user', label: 'User', type: 'color' },
    { var: '--theme-event-data', label: 'Data', type: 'color' },
    { var: '--theme-event-file', label: 'File', type: 'color' },
    { var: '--theme-event-git', label: 'Git', type: 'color' },
    { var: '--theme-event-system', label: 'System', type: 'color' },
  ],
};

export function ThemeProvider({ children }: ThemeProviderProps): ReactElement {
  const {
    currentTheme,
    overrides,
    setTheme: backendSetTheme,
    setVariable: backendSetVariable,
    resetVariable: backendResetVariable,
    resetOverrides: backendResetOverrides,
    loading,
  } = useThemeState();

  // Track previous overrides to clean up removed ones
  const prevOverridesRef = useRef<Record<string, string>>({});

  // Apply theme class to root
  useEffect(() => {
    if (loading) return;

    const root = document.documentElement;

    // Remove all theme classes
    PRESET_THEMES.forEach((theme) => {
      root.classList.remove(`theme-${theme}`);
    });

    // Add current theme class (dark is default, no class needed)
    if (currentTheme !== 'dark') {
      root.classList.add(`theme-${currentTheme}`);
    }
  }, [currentTheme, loading]);

  // Apply overrides as inline styles on root
  useEffect(() => {
    if (loading) return;

    const root = document.documentElement;
    const prevOverrides = prevOverridesRef.current;

    // Remove properties that are no longer in overrides
    Object.keys(prevOverrides).forEach((varName) => {
      if (!(varName in overrides)) {
        root.style.removeProperty(varName);
      }
    });

    // Set current overrides
    Object.entries(overrides).forEach(([varName, value]) => {
      if (value) {
        root.style.setProperty(varName, value);
      } else {
        root.style.removeProperty(varName);
      }
    });

    prevOverridesRef.current = { ...overrides };
  }, [overrides, loading]);

  const setTheme = useCallback((theme: string): void => {
    if (PRESET_THEMES.includes(theme)) {
      // Clear inline styles first
      const root = document.documentElement;
      Object.keys(overrides).forEach((varName) => {
        root.style.removeProperty(varName);
      });
      // Backend will reset overrides when theme changes
      backendSetTheme(theme);
      backendResetOverrides();
    }
  }, [overrides, backendSetTheme, backendResetOverrides]);

  const setVariable = useCallback((varName: string, value: string): void => {
    backendSetVariable(varName, value);
  }, [backendSetVariable]);

  const getVariable = useCallback((varName: string): string => {
    // Check overrides first
    if (overrides[varName]) {
      return overrides[varName];
    }
    // Get computed value from CSS
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }, [overrides]);

  const resetVariable = useCallback((varName: string): void => {
    backendResetVariable(varName);
    document.documentElement.style.removeProperty(varName);
  }, [backendResetVariable]);

  const resetAllOverrides = useCallback((): void => {
    Object.keys(overrides).forEach((varName) => {
      document.documentElement.style.removeProperty(varName);
    });
    backendResetOverrides();
  }, [overrides, backendResetOverrides]);

  const exportTheme = useCallback((): ThemeExport => {
    const allVars: Record<string, string> = {};
    Object.values(THEME_CATEGORIES).flat().forEach(({ var: varName }) => {
      allVars[varName] = getVariable(varName);
    });
    return {
      baseTheme: currentTheme,
      variables: allVars,
      overrides,
    };
  }, [currentTheme, overrides, getVariable]);

  const value: ThemeContextValue = {
    currentTheme,
    presetThemes: PRESET_THEMES,
    setTheme,
    setVariable,
    getVariable,
    resetVariable,
    resetAllOverrides,
    overrides,
    exportTheme,
    loading,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}
