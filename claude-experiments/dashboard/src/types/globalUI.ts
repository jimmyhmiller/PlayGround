/**
 * Global UI Types
 *
 * Type definitions for the global UI widget system - state-connected components
 * that render in fixed positions outside of windows.
 */

import type { ComponentType, ReactNode, ReactElement } from 'react';
import type { CommandResult } from './state';

// ========== Slot System ==========

/**
 * Corner position identifiers
 */
export type CornerPosition = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

/**
 * Bar edge identifiers
 */
export type BarEdge = 'top' | 'bottom';

/**
 * Panel side identifiers
 */
export type PanelSide = 'left' | 'right';

/**
 * Slot position configuration - determines how a slot is rendered
 */
export type SlotPosition =
  | { type: 'corner'; corner: CornerPosition }
  | { type: 'bar'; edge: BarEdge }
  | { type: 'panel'; side: PanelSide; width?: number; collapsible?: boolean }
  | { type: 'custom'; render: (children: ReactNode) => ReactElement };

/**
 * Slot definition - a named position where widgets can be placed
 */
export interface SlotDefinition {
  /** Unique identifier for this slot */
  id: string;
  /** How this slot is positioned and rendered */
  position: SlotPosition;
  /** Z-index layer for this slot (defaults based on position type) */
  zIndex?: number;
}

// ========== Widget System ==========

/**
 * Command definition for widget actions
 */
export interface CommandDefinition {
  /** Command type string (e.g., 'dashboards.switch') */
  type: string;
  /** Optional payload transformer */
  transform?: (payload: unknown) => unknown;
}

/**
 * Widget definition - declarative configuration for a state-connected component
 */
export interface WidgetDefinition<
  TState = unknown,
  TSelected = unknown,
  TCommands extends Record<string, CommandDefinition> = Record<string, never>
> {
  /** Unique identifier for this widget type */
  id: string;
  /** Human-readable display name */
  displayName: string;
  /** State subscription configuration */
  state: {
    /** State path to subscribe to (e.g., 'projects', 'dashboards') */
    path: string;
    /** Selector function to extract specific data from state */
    select: (state: TState) => TSelected;
    /** Optional equality function for performance optimization */
    equals?: (a: TSelected, b: TSelected) => boolean;
  };
  /** Command definitions for interactive widgets */
  commands?: TCommands;
}

/**
 * Props injected into widget components by createStateWidget
 */
export interface WidgetProps<
  TSelected,
  TCommands extends Record<string, CommandDefinition> = Record<string, never>
> {
  /** Selected state data */
  data: TSelected;
  /** Whether state is still loading */
  loading: boolean;
  /** Dispatch functions for each defined command */
  actions: {
    [K in keyof TCommands]: (payload?: unknown) => Promise<CommandResult>;
  };
}

/**
 * Widget component type - what developers implement
 */
export type WidgetComponent<
  TSelected,
  TCommands extends Record<string, CommandDefinition> = Record<string, never>
> = ComponentType<WidgetProps<TSelected, TCommands>>;

// ========== Widget Registry ==========

/**
 * Widget registration - places a widget component in a slot
 */
export interface WidgetRegistration {
  /** Unique ID for this widget instance */
  id: string;
  /** Slot ID where this widget should render */
  slot: string;
  /** The widget component to render */
  component: ComponentType;
  /** Priority for ordering within a slot (higher = rendered later) */
  priority?: number;
  /** Whether this widget is currently visible */
  visible?: boolean;
}

/**
 * Complete registry configuration
 */
export interface GlobalUIRegistry {
  /** Slot definitions */
  slots: SlotDefinition[];
  /** Widget registrations */
  widgets: WidgetRegistration[];
}

// ========== Z-Index Layers ==========

/**
 * Default z-index values for slot types
 */
export const SLOT_Z_INDEX: Record<string, number> = {
  bar: 100,
  corner: 200,
  panel: 100,
  custom: 150,
  overlay: 1000,
};
