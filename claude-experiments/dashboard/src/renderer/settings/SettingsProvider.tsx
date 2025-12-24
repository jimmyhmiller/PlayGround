import { createContext, useContext, useEffect, ReactNode, ReactElement } from 'react';
import { useSettingsState } from '../hooks/useBackendState';
import type { SettingsState, CommandResult } from '../../types/state';

/**
 * Settings Context
 *
 * Global settings that persist on the backend.
 * These apply on top of any theme and don't reset when switching themes.
 */

interface FontSizePreset {
  xs: number;
  sm: number;
  md: number;
  lg: number;
}

interface ShortcutPreset {
  label: string;
  value: string;
}

interface SettingsContextValue {
  settings: SettingsState;
  updateSetting: (key: string, value: unknown) => Promise<CommandResult>;
  resetSettings: () => Promise<CommandResult>;
  fontSizeOptions: string[];
  spacingOptions: string[];
  shortcutPresets: ShortcutPreset[];
  loading: boolean;
}

interface SettingsProviderProps {
  children: ReactNode;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

// Font size presets (base sizes in px)
const FONT_SIZE_PRESETS: Record<string, FontSizePreset> = {
  small: { xs: 9, sm: 10, md: 11, lg: 12 },
  medium: { xs: 10, sm: 11, md: 12, lg: 14 },
  large: { xs: 11, sm: 12, md: 14, lg: 16 },
  xlarge: { xs: 12, sm: 14, md: 16, lg: 18 },
};

// Spacing presets (multiplier)
const SPACING_PRESETS: Record<string, number> = {
  compact: 0.75,
  normal: 1.0,
  relaxed: 1.25,
};

// Common keyboard shortcut presets for command palette
export const SHORTCUT_PRESETS: ShortcutPreset[] = [
  { label: '⌘⇧P', value: 'cmd+shift+p' },
  { label: '⌘K', value: 'cmd+k' },
  { label: '⌘P', value: 'cmd+p' },
  { label: '⌘/', value: 'cmd+/' },
];

export function SettingsProvider({ children }: SettingsProviderProps): ReactElement {
  const { settings, updateSetting, resetSettings, loading } = useSettingsState();

  // Apply settings as CSS variables (these override theme values)
  useEffect(() => {
    if (loading) return;

    const root = document.documentElement;

    // Apply font sizes
    const fontSizes = FONT_SIZE_PRESETS[settings.fontSize] ?? FONT_SIZE_PRESETS.medium ?? { xs: 10, sm: 11, md: 12, lg: 14 };
    const scale = settings.fontScale || 1.0;

    root.style.setProperty('--settings-font-size-xs', `${Math.round(fontSizes.xs * scale)}px`);
    root.style.setProperty('--settings-font-size-sm', `${Math.round(fontSizes.sm * scale)}px`);
    root.style.setProperty('--settings-font-size-md', `${Math.round(fontSizes.md * scale)}px`);
    root.style.setProperty('--settings-font-size-lg', `${Math.round(fontSizes.lg * scale)}px`);

    // Apply spacing multiplier
    const spacingMult = SPACING_PRESETS[settings.spacing] || 1.0;
    root.style.setProperty('--settings-spacing-mult', spacingMult.toString());

    // Override theme font sizes with settings values
    root.style.setProperty('--theme-font-size-xs', 'var(--settings-font-size-xs)');
    root.style.setProperty('--theme-font-size-sm', 'var(--settings-font-size-sm)');
    root.style.setProperty('--theme-font-size-md', 'var(--settings-font-size-md)');
    root.style.setProperty('--theme-font-size-lg', 'var(--settings-font-size-lg)');
  }, [settings, loading]);

  const value: SettingsContextValue = {
    settings,
    updateSetting,
    resetSettings,
    fontSizeOptions: Object.keys(FONT_SIZE_PRESETS),
    spacingOptions: Object.keys(SPACING_PRESETS),
    shortcutPresets: SHORTCUT_PRESETS,
    loading,
  };

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings must be used within SettingsProvider');
  }
  return context;
}
