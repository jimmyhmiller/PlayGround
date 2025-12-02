import React, { createContext, useContext } from 'react';
import type { Theme } from '../types/theme';
import type { LayoutSettings } from '../types/dashboard';

export interface DashboardContextValue {
  theme: Theme;
  layout?: LayoutSettings;
  nestingDepth: number;
  widgetPath: string;
}

const DashboardContext = createContext<DashboardContextValue | null>(null);

export interface DashboardProviderProps {
  theme: Theme;
  layout?: LayoutSettings;
  nestingDepth?: number;
  widgetPath?: string;
  children: React.ReactNode;
}

export const DashboardProvider: React.FC<DashboardProviderProps> = ({
  theme,
  layout,
  nestingDepth = 0,
  widgetPath = '',
  children,
}) => {
  const value: DashboardContextValue = {
    theme,
    layout,
    nestingDepth,
    widgetPath,
  };

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
};

export const useDashboardContext = (): DashboardContextValue | null => {
  return useContext(DashboardContext);
};
